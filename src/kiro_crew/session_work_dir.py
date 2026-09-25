"""Reclaim the per-run work directories that subagent and cron sessions leave behind.

Every ACP session runs in a work directory. A dashboard or channel session gets a
long-lived one keyed on its slot; a subagent or a stateless cron run gets one keyed
on its own session key -- ``workspace_root()/subagent_<id>``,
``workspace_root()/cron_<job>_<run>`` -- that nothing ever revisits once the run
ends. Preparing the spawn writes one file into it, ``.kiro/settings/cli.json``
(the skill-projection overlay, plus the lock sidecar that serializes writers), so a
host that runs many subagents or a stateless cron every minute accumulates
thousands of directories holding exactly that file. Two things pay for them: the
directory listing itself, and the skill-view aliases in ``~/.kiro/agents`` whose
work directory they are: a reclaim that keyed on the directory's existence could
never retire one, because the directory never went away.

This module removes such a directory when it holds ONLY that residue. The rule is
the safety argument, so it is stated once: the only files this module ever unlinks
are the two names Crew itself writes, inside the two directories Crew itself
creates, and every directory is removed with ``rmdir`` -- which fails on anything
non-empty. A run that wrote a report into its cwd, a resumed conversation's
project checkout, an operator's own file: none of it is reachable by this code,
because a directory holding it is simply not empty. Refusing is the default
answer; reclaiming needs the whole tree to be exactly the residue.

Provenance is written, never inferred. The provider that DERIVED a directory
from a one-run session key drops a marker file into it before the first spawn
(:func:`mark_run_dir`); that marker, and only that marker, says "Crew made this
directory for one run" and names the gateway process that made it. Once the ACP
root is spawned, the provider adds that process's pid and persistent start identity.
A name under
the workspace root says nothing -- a person can make ``subagent_deadbeef`` there
-- and the contents of ``cli.json`` say nothing either, because the projection
merges its keys INTO a file that already exists, so a Crew key in the file proves
Crew touched it, not that Crew owns the directory. Directories older builds left
behind carry no marker and are never reclaimed automatically; ``kirocrew doctor``
counts them and names the manual remedy, the same way it treats the alias backlog.

Two callers. The session's own provider reclaims its directory at shutdown,
which covers the ordinary end of every run; its provenance is the factory flag,
so an absent marker is allowed. When a marker is present, its gateway must be this
process or confirmed dead and its recorded ACP root must be confirmed gone. The
gateway's hourly maintenance wake sweeps the workspace root for the rest -- runs
that ended abnormally, or under a gateway that restarted -- taking only MARKED
directories whose gateway and recorded child identities are confirmed gone,
skipping any a live session in this process still names and any younger than a
grace window. A spawn whose provider is not yet
registered cannot lose its directory between the mkdir and the first turn, and a
second gateway sharing the default workspace root cannot sweep the first one's
live run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from kiro_crew import pinned_fs, platform_compat
from kiro_crew.workspace_cli_settings import CLI_SETTINGS_LOCK_NAME

logger = logging.getLogger(__name__)

#: The one directory chain Crew creates inside a work dir, root first.
_RESIDUE_DIRS: tuple[str, ...] = (".kiro", "settings")
#: The only files Crew writes at the leaf. Anything else is somebody's data.
_RESIDUE_FILES: frozenset[str] = frozenset({"cli.json", CLI_SETTINGS_LOCK_NAME})
#: The provenance marker, at the work dir's root. Written by :func:`mark_run_dir`
#: for a DERIVED one-run directory only, never for a caller's cwd. Line one is
#: the gateway pid; once spawned, line two records the ACP root pid and start token.
RUN_DIR_MARKER = ".kirocrew-run-dir"
#: Session-key shapes that mint ONE-RUN directories. Subagent ids are the hex
#: run ids ``SubagentManager._mint_agent_id`` draws -- sixteen characters on
#: this tree, eight on the shipped builds whose directories the sweep also has
#: to recognise; cron job and run ids are the eight-hex UUID prefixes minted in
#: ``cron.py``. A named agent never reaches a key in these shapes: the stable
#: sequence key ``cron:<job>:<agent>`` carries a name, not eight hex digits.
#: Both the key predicate and the directory-name filter are generated from this
#: table so their shapes cannot diverge.
_DERIVED_SESSION_SHAPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("subagent", (16,)),
    ("subagent", (8,)),
    ("cron", (8, 8)),
)


def _derived_shape(separator: str) -> str:
    return "|".join(
        re.escape(prefix) + separator + separator.join(rf"[0-9a-f]{{{width}}}" for width in widths)
        for prefix, widths in _DERIVED_SESSION_SHAPES
    )


_DISPOSABLE_SESSION_KEY_RE = re.compile(rf"^(?:{_derived_shape(':')})$")
#: Used as the sweep's candidate filter and to COUNT unmarked leftovers for the
#: doctor; it authorizes no deletion on its own.
DERIVED_NAME_RE = re.compile(rf"^(?:{_derived_shape('_')})$")

#: Mirrors agent_scratch's bounded owner-marker discipline. The marker is inside
#: a directory the spawned process can write, so it is never read without a cap.
_OWNER_MARKER_MAX_BYTES = 4096
_PID_MAX = 2**31 - 1

#: A swept directory must be idle this long. Not a liveness signal -- the live
#: set is -- but the window between a spawn's ``mkdir`` and its provider showing
#: up in the registry, which the live set cannot see.
SWEEP_MIN_AGE_SECS = 3600.0
#: Bounds on one sweep. The root can hold tens of thousands of entries; a wake
#: that could not finish leaves the rest for the next one.
SWEEP_MAX_ENTRIES = 5000
SWEEP_MAX_SECONDS = 20.0


def is_disposable_session_key(session_key: str | None) -> bool:
    """Whether a work directory DERIVED from *session_key* is one run's and disposable.

    Subagent sessions use ``subagent:<id>``. A cron's stable
    ``cron:<job_id>`` key belongs to its persistent session; only the stateless
    ``cron:<job_id>:<run_id>`` key is a one-run directory. Only a derived
    directory qualifies: a caller-supplied ``cwd`` is always the user's.
    """
    if not isinstance(session_key, str):
        return False
    return _DISPOSABLE_SESSION_KEY_RE.fullmatch(session_key) is not None


def _is_disposable_dir_name(name: str) -> bool:
    """Whether *name* has the exact one-run directory shape.

    This is only a sweep candidate filter; :data:`RUN_DIR_MARKER` remains the
    provenance required before deletion.
    """
    return DERIVED_NAME_RE.match(name) is not None


@dataclass(frozen=True)
class _RunDirMarker:
    gateway_pid: int
    child_pid: int | None = None
    child_start_token: str | None = None


def _valid_pid(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("run-directory marker names a non-integer pid")
    if isinstance(value, str) and (not value or not value.isascii() or not value.isdigit()):
        raise ValueError("run-directory marker names a non-integer pid")
    pid = int(value)
    if pid < 1 or pid > _PID_MAX:
        raise ValueError("run-directory marker names an impossible pid")
    return pid


def _marker_from_fd(fd: int) -> _RunDirMarker:
    """Read one bounded marker from an already-open, no-follow descriptor."""
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_size > _OWNER_MARKER_MAX_BYTES:
        raise ValueError("run-directory owner marker is not a bounded regular file")
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = _OWNER_MARKER_MAX_BYTES + 1
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) > _OWNER_MARKER_MAX_BYTES:
        raise ValueError("run-directory owner marker is oversized")
    lines = data.decode("utf-8").splitlines()
    if len(lines) not in {1, 2}:
        raise ValueError("run-directory owner marker has an unknown shape")
    gateway_pid = _valid_pid(lines[0])
    if len(lines) == 1:
        return _RunDirMarker(gateway_pid)
    child = json.loads(lines[1])
    if not isinstance(child, dict) or set(child) != {"pid", "start"}:
        raise ValueError("run-directory child marker has an unknown shape")
    child_pid = _valid_pid(child["pid"])
    start = child["start"]
    if start is not None and (not isinstance(start, str) or not start):
        raise ValueError("run-directory child marker has an invalid start token")
    return _RunDirMarker(gateway_pid, child_pid, start)


def _write_marker_to_fd(fd: int, marker: _RunDirMarker) -> None:
    data = str(marker.gateway_pid)
    if marker.child_pid is not None:
        child = json.dumps(
            {"pid": marker.child_pid, "start": marker.child_start_token},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        data = f"{data}\n{child}"
    encoded = data.encode("ascii")
    if len(encoded) > _OWNER_MARKER_MAX_BYTES:
        raise ValueError("run-directory owner marker is oversized")
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    view = memoryview(encoded)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _claim_owner_fd(fd: int, *, created: bool, identity_is_current: Callable[[], bool]) -> bool:
    """Claim a locked marker unless a live or unverifiable owner already has it."""
    owner = os.getpid()
    try:
        with platform_compat.file_lock(fd, exclusive=True, timeout=2.0):
            if not created:
                existing = _marker_from_fd(fd)
                if existing.gateway_pid == owner:
                    return identity_is_current()
                if not _marker_allows_reclaim(existing):
                    return False
            _write_marker_to_fd(fd, _RunDirMarker(owner))
            return identity_is_current()
    except (OSError, RecursionError, UnicodeError, ValueError):
        return False


def _open_marker_at(root_fd: int) -> tuple[int, bool]:
    flags = os.O_RDWR | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return (
            os.open(
                RUN_DIR_MARKER,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=root_fd,
            ),
            True,
        )
    except FileExistsError:
        return os.open(RUN_DIR_MARKER, flags, dir_fd=root_fd), False


def _open_marker_by_name(marker: Path) -> tuple[int, bool]:
    flags = os.O_RDWR | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(marker, flags | os.O_CREAT | os.O_EXCL, 0o600), True
    except FileExistsError:
        return os.open(marker, flags), False


def mark_run_dir(work_dir: Path) -> bool:
    """Record this gateway as *work_dir*'s owner without stealing a live mark.

    A restart may replace a predecessor only after its gateway and recorded child
    identities are confirmed gone. A marker naming this process is idempotent; live,
    incomplete, or malformed foreign evidence is left untouched. The shutdown path
    still keys on the
    factory flag, so a refused marker never fails session start.
    """
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    if not pinned_fs.supports_pinned_walk():
        if platform_compat.is_link_or_junction(work_dir):
            return False
        marker = work_dir / RUN_DIR_MARKER
        if platform_compat.is_link_or_junction(marker):
            return False
        try:
            marker_fd, created = _open_marker_by_name(marker)
        except OSError:
            return False
        try:
            opened = os.fstat(marker_fd)

            def current() -> bool:
                if platform_compat.is_link_or_junction(marker):
                    return False
                info = pinned_fs.lstat_by_name(marker)
                return info is not None and (info.st_dev, info.st_ino) == (
                    opened.st_dev,
                    opened.st_ino,
                )

            return _claim_owner_fd(marker_fd, created=created, identity_is_current=current)
        finally:
            os.close(marker_fd)
    try:
        root_fd = pinned_fs.open_dir_pinned(work_dir, what="session work directory")
    except (pinned_fs.PinnedPathRefusal, OSError):
        return False
    try:
        fd, created = _open_marker_at(root_fd)
    except OSError:
        os.close(root_fd)
        return False
    opened = os.fstat(fd)
    try:

        def current() -> bool:
            info = pinned_fs.stat_at(root_fd, RUN_DIR_MARKER)
            return info is not None and (info.st_dev, info.st_ino) == (
                opened.st_dev,
                opened.st_ino,
            )

        return _claim_owner_fd(fd, created=created, identity_is_current=current)
    finally:
        os.close(fd)
        os.close(root_fd)


def _record_child_fd(
    fd: int,
    *,
    created: bool,
    child_pid: int,
    child_start_token: str | None,
    identity_is_current: Callable[[], bool],
) -> bool:
    """Add the spawned root identity without changing a foreign marker."""
    try:
        child = _RunDirMarker(
            os.getpid(),
            _valid_pid(child_pid),
            child_start_token if isinstance(child_start_token, str) and child_start_token else None,
        )
        with platform_compat.file_lock(fd, exclusive=True, timeout=2.0):
            if not created and _marker_from_fd(fd).gateway_pid != os.getpid():
                return False
            _write_marker_to_fd(fd, child)
            return identity_is_current()
    except (OSError, RecursionError, UnicodeError, ValueError):
        return False


def record_run_dir_child(work_dir: Path, child_pid: int, child_start_token: str | None) -> bool:
    """Record the spawned ACP root identity in this gateway's run-dir marker."""
    if not pinned_fs.supports_pinned_walk():
        if platform_compat.is_link_or_junction(work_dir):
            return False
        marker = work_dir / RUN_DIR_MARKER
        if platform_compat.is_link_or_junction(marker):
            return False
        try:
            marker_fd, created = _open_marker_by_name(marker)
        except OSError:
            return False
        try:
            opened = os.fstat(marker_fd)

            def current() -> bool:
                if platform_compat.is_link_or_junction(marker):
                    return False
                info = pinned_fs.lstat_by_name(marker)
                return info is not None and (info.st_dev, info.st_ino) == (
                    opened.st_dev,
                    opened.st_ino,
                )

            return _record_child_fd(
                marker_fd,
                created=created,
                child_pid=child_pid,
                child_start_token=child_start_token,
                identity_is_current=current,
            )
        finally:
            os.close(marker_fd)
    try:
        root_fd = pinned_fs.open_dir_pinned(work_dir, what="session work directory")
    except (pinned_fs.PinnedPathRefusal, OSError):
        return False
    try:
        fd, created = _open_marker_at(root_fd)
    except OSError:
        os.close(root_fd)
        return False
    opened = os.fstat(fd)
    try:

        def current() -> bool:
            info = pinned_fs.stat_at(root_fd, RUN_DIR_MARKER)
            return info is not None and (info.st_dev, info.st_ino) == (
                opened.st_dev,
                opened.st_ino,
            )

        return _record_child_fd(
            fd,
            created=created,
            child_pid=child_pid,
            child_start_token=child_start_token,
            identity_is_current=current,
        )
    finally:
        os.close(fd)
        os.close(root_fd)


def _read_run_dir_marker_at(root_fd: int) -> _RunDirMarker | None:
    """Read the marker relative to a pinned work-directory descriptor."""
    try:
        fd = os.open(
            RUN_DIR_MARKER,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=root_fd,
        )
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
        with platform_compat.file_lock(fd, exclusive=False, timeout=2.0):
            info = pinned_fs.stat_at(root_fd, RUN_DIR_MARKER)
            if info is None or (info.st_dev, info.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                return None
            return _marker_from_fd(fd)
    except (OSError, RecursionError, UnicodeError, ValueError):
        return None
    finally:
        os.close(fd)


def _read_run_dir_marker(work_dir: Path) -> _RunDirMarker | None:
    """Read a marker through a bounded regular-file descriptor; fail closed."""
    if not pinned_fs.supports_pinned_walk():
        marker = work_dir / RUN_DIR_MARKER
        if platform_compat.is_link_or_junction(work_dir) or platform_compat.is_link_or_junction(
            marker
        ):
            return None
        try:
            fd = os.open(
                marker,
                os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError:
            return None
        try:
            opened = os.fstat(fd)
            with platform_compat.file_lock(fd, exclusive=False, timeout=2.0):
                info = pinned_fs.lstat_by_name(marker)
                if info is None or (info.st_dev, info.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    return None
                return _marker_from_fd(fd)
        except (OSError, RecursionError, UnicodeError, ValueError):
            return None
        finally:
            os.close(fd)
    try:
        root_fd = pinned_fs.open_dir_pinned(work_dir, what="session work directory")
    except (pinned_fs.PinnedPathRefusal, OSError):
        return None
    try:
        return _read_run_dir_marker_at(root_fd)
    finally:
        os.close(root_fd)


def _marker_allows_reclaim(marker: _RunDirMarker) -> bool:
    """Whether both the gateway and recorded child-tree identities are gone."""
    gateway_is_ours = marker.gateway_pid == os.getpid()
    if (
        not gateway_is_ours
        and platform_compat.pid_liveness(marker.gateway_pid) != platform_compat.PID_DEAD
    ):
        return False
    if marker.child_pid is None:
        # This process knows an owner-only marker precedes its own spawn. For a
        # dead gateway, the same shape can mean it crashed after spawn but before
        # recording the child, so only the local owner may authorize reclaim.
        return gateway_is_ours
    child_liveness = platform_compat.pid_liveness(marker.child_pid)
    if child_liveness == platform_compat.PID_DEAD:
        return True
    if child_liveness != platform_compat.PID_ALIVE or marker.child_start_token is None:
        return False
    current_start = platform_compat.get_process_start_id(marker.child_pid)
    return current_start is not None and current_start != marker.child_start_token


def reclaim_session_work_dir(
    work_dir: Path,
    *,
    min_age_secs: float = 0.0,
    require_marker: bool = False,
    now: float | None = None,
) -> bool:
    """Remove *work_dir* when it holds only Crew's own residue; otherwise leave it.

    ``True`` means the directory is gone. ``False`` covers every other outcome and
    is never an error: the directory was already gone, holds something that is
    not residue, is younger than *min_age_secs* (judged by the newest mtime in the
    residue tree), sits behind a link, or the filesystem refused. Nothing here
    raises to the caller -- reclaiming is hygiene, and hygiene must never be the
    thing that fails a session's shutdown.

    With *require_marker*, :data:`RUN_DIR_MARKER` must be present at the root:
    the sweep passes it because it knows the directory only by its name, and a
    name is not provenance. The shutdown path does not require one, because the
    factory that derived the name is its provenance. When either path finds a
    marker, malformed evidence, a live foreign gateway, or a live recorded child
    refuses the reclaim. A live child pid permits reclaim only when its readable
    start token proves that the pid was recycled.

    Every step addresses the tree through descriptors where the platform allows,
    so a component swapped for a link after the check is refused rather than
    followed. What ``rmdir`` alone cannot rule out is a swap for another EMPTY
    directory between two adjacent syscalls, whose worst outcome is that empty
    directory removed; the two file names it unlinks are Crew's own, so a swap
    there can cost a file of that name and nothing else.
    """
    if not pinned_fs.supports_pinned_walk():
        return _reclaim_by_name(
            work_dir, min_age_secs=min_age_secs, require_marker=require_marker, now=now
        )
    try:
        root_fd = pinned_fs.open_dir_pinned(work_dir, what="session work directory")
    except (pinned_fs.PinnedPathRefusal, OSError):
        return False
    opened = [root_fd]
    try:
        newest = os.fstat(root_fd).st_mtime
        fd = root_fd
        # Walk down the one chain Crew creates. Each level may hold ONLY the next
        # level (or, at the leaf, only the residue files); anything else ends the
        # reclaim before a single unlink.
        marked = False
        for depth, child in enumerate(_RESIDUE_DIRS):
            entries = list(os.scandir(fd))
            names = {entry.name for entry in entries}
            if depth == 0 and RUN_DIR_MARKER in names:
                marker_info = pinned_fs.stat_at(fd, RUN_DIR_MARKER)
                if marker_info is None or not stat.S_ISREG(marker_info.st_mode):
                    return False
                marker = _read_run_dir_marker_at(fd)
                if marker is None or not _marker_allows_reclaim(marker):
                    return False
                newest = max(newest, marker_info.st_mtime)
                marked = True
                names.discard(RUN_DIR_MARKER)
            if not names:
                # An already-emptied prefix of the chain: still ours to remove.
                break
            if names != {child}:
                return False
            entry_info = pinned_fs.stat_at(fd, child)
            if entry_info is None or not stat.S_ISDIR(entry_info.st_mode):
                return False
            newest = max(newest, entry_info.st_mtime)
            try:
                fd = os.open(child, pinned_fs.dir_flags(), dir_fd=fd)
            except OSError:
                return False
            opened.append(fd)
            if depth == len(_RESIDUE_DIRS) - 1:
                leaf_entries = list(os.scandir(fd))
                for entry in leaf_entries:
                    if entry.name not in _RESIDUE_FILES:
                        return False
                    info = pinned_fs.stat_at(fd, entry.name)
                    if info is None or not stat.S_ISREG(info.st_mode):
                        return False
                    newest = max(newest, info.st_mtime)
        if require_marker and not marked:
            return False
        current = time.time() if now is None else now
        if min_age_secs > 0 and current - newest < min_age_secs:
            return False
        # Unlink the residue files at the deepest opened level (only the leaf has
        # any), then rmdir each level from the leaf up through the parent's
        # descriptor. A directory that gained an entry meanwhile fails its rmdir
        # and the reclaim stops there, leaving what arrived in place.
        leaf_fd = opened[-1]
        if len(opened) == len(_RESIDUE_DIRS) + 1:
            for entry in list(os.scandir(leaf_fd)):
                if entry.name not in _RESIDUE_FILES or not pinned_fs.is_regular_at(
                    leaf_fd, entry.name
                ):
                    return False
                try:
                    os.unlink(entry.name, dir_fd=leaf_fd)
                except OSError:
                    return False
        for level in range(len(opened) - 1, 0, -1):
            try:
                os.rmdir(_RESIDUE_DIRS[level - 1], dir_fd=opened[level - 1])
            except OSError:
                return False
        if marked:
            try:
                os.unlink(RUN_DIR_MARKER, dir_fd=root_fd)
            except OSError:
                return False
        parent = os.path.realpath(work_dir.parent)
        try:
            parent_fd = pinned_fs.pin_parent(parent, what="session work directory")
        except (pinned_fs.PinnedPathRefusal, OSError):
            return False
        try:
            os.rmdir(work_dir.name, dir_fd=parent_fd)
        except OSError:
            return False
        finally:
            os.close(parent_fd)
        return True
    except OSError:
        return False
    finally:
        pinned_fs.close_all(opened)


def _reclaim_by_name(
    work_dir: Path, *, min_age_secs: float, require_marker: bool, now: float | None
) -> bool:
    """The by-name form for platforms without descriptor-relative opens (Windows).

    Same rule, same refusals; what it lacks is the pin, which the platform's
    mandatory file locking partly stands in for -- an open ``cli.json`` cannot be
    unlinked there, so a live writer is refused by the OS itself.
    """
    if platform_compat.is_link_or_junction(work_dir):
        return False
    info = pinned_fs.lstat_by_name(work_dir)
    if info is None or not stat.S_ISDIR(info.st_mode):
        return False
    newest = info.st_mtime
    levels = [work_dir]
    marked = False
    try:
        for depth, child in enumerate(_RESIDUE_DIRS):
            names = set(os.listdir(levels[-1]))
            if depth == 0 and RUN_DIR_MARKER in names:
                marker = work_dir / RUN_DIR_MARKER
                marker_info = pinned_fs.lstat_by_name(marker)
                if (
                    platform_compat.is_link_or_junction(marker)
                    or marker_info is None
                    or not stat.S_ISREG(marker_info.st_mode)
                ):
                    return False
                marker_record = _read_run_dir_marker(work_dir)
                if marker_record is None or not _marker_allows_reclaim(marker_record):
                    return False
                newest = max(newest, marker_info.st_mtime)
                marked = True
                names.discard(RUN_DIR_MARKER)
            if not names:
                break
            if names != {child}:
                return False
            path = levels[-1] / child
            if platform_compat.is_link_or_junction(path):
                return False
            child_info = pinned_fs.lstat_by_name(path)
            if child_info is None or not stat.S_ISDIR(child_info.st_mode):
                return False
            newest = max(newest, child_info.st_mtime)
            levels.append(path)
            if depth == len(_RESIDUE_DIRS) - 1:
                for name in os.listdir(path):
                    if name not in _RESIDUE_FILES or platform_compat.is_link_or_junction(
                        path / name
                    ):
                        return False
                    file_info = pinned_fs.lstat_by_name(path / name)
                    if file_info is None or not stat.S_ISREG(file_info.st_mode):
                        return False
                    newest = max(newest, file_info.st_mtime)
        if require_marker and not marked:
            return False
        current = time.time() if now is None else now
        if min_age_secs > 0 and current - newest < min_age_secs:
            return False
        if len(levels) == len(_RESIDUE_DIRS) + 1:
            for name in os.listdir(levels[-1]):
                if name not in _RESIDUE_FILES:
                    return False
                (levels[-1] / name).unlink()
        for path in reversed(levels[1:]):
            path.rmdir()
        if marked:
            (work_dir / RUN_DIR_MARKER).unlink()
        work_dir.rmdir()
    except OSError:
        return False
    return True


def sweep_disposable_work_dirs(
    root: Path,
    *,
    live_work_dirs: Collection[str],
    min_age_secs: float = SWEEP_MIN_AGE_SECS,
    max_entries: int = SWEEP_MAX_ENTRIES,
    max_seconds: float = SWEEP_MAX_SECONDS,
    now: float | None = None,
) -> int:
    """Reclaim the residue-only run directories under *root*; return how many went.

    *live_work_dirs* are the work directories of every session this process
    currently holds, as the provider spells them; an entry whose path is in that
    set is skipped whatever it holds. The rest are judged by
    :func:`reclaim_session_work_dir` with *min_age_secs* and with the marker
    required, so a directory a spawn created moments ago is left for the next
    wake even when it is empty, and a directory Crew did not mark -- a person's,
    or an older build's -- is never residue however old it is.

    Bounded twice, by entries examined and by wall clock, because the root this
    exists for holds tens of thousands of entries and a maintenance wake must
    not turn into a minutes-long scan -- and bounded in what it RETAINS: the
    listing is streamed, never collected, so a root of any size costs the sweep
    a few counters and nothing that grows with the backlog. Whatever a bound
    leaves is reached by a later wake: the walk starts at a random offset into
    the directory's own order (counted in one streaming pass, then skipped in
    the next) and wraps, so an unreclaimable prefix cannot hide the rest forever.
    """
    if platform_compat.is_link_or_junction(root):
        return 0
    deadline = time.monotonic() + max_seconds
    try:
        total = 0
        with os.scandir(root) as it:
            for entry in it:
                if _is_disposable_dir_name(entry.name):
                    total += 1
                if time.monotonic() >= deadline:
                    # A root too large to even count inside the budget is left
                    # for a wake with more of it; nothing is judged blind.
                    logger.debug("session work dirs: sweep budget spent counting %d", total)
                    return 0
    except OSError:
        return 0
    if not total:
        return 0
    live = {os.path.normcase(str(path)) for path in live_work_dirs}
    offset = secrets.randbelow(total)
    examined = 0
    reclaimed = 0
    bounded = False

    def judge(name: str) -> None:
        nonlocal examined, reclaimed
        examined += 1
        candidate = root / name
        if os.path.normcase(str(candidate)) in live:
            return
        marker = _read_run_dir_marker(candidate)
        if marker is None or not _marker_allows_reclaim(marker):
            return
        if reclaim_session_work_dir(
            candidate, min_age_secs=min_age_secs, require_marker=True, now=now
        ):
            reclaimed += 1

    # Two streaming passes over the same order: the tail from the offset, then
    # the head up to it. Each pass is its own listing, so an entry the first
    # pass removed is simply not there for the second.
    for tail in (True, False):
        position = 0
        try:
            with os.scandir(root) as it:
                for entry in it:
                    if not _is_disposable_dir_name(entry.name):
                        continue
                    position += 1
                    if tail != (position > offset):
                        continue
                    if examined >= max_entries or time.monotonic() >= deadline:
                        bounded = True
                        break
                    judge(entry.name)
        except OSError:
            break
        if bounded:
            break
    if bounded:
        logger.debug(
            "session work dirs: sweep bound reached after %d of %d entries", examined, total
        )
    if reclaimed:
        logger.info("session work dirs: reclaimed %d idle run director(ies)", reclaimed)
    return reclaimed


def count_unmarked_run_dirs(root: Path, *, max_entries: int = 100000) -> tuple[int, bool]:
    """How many run-shaped directories under *root* carry no marker, and whether the count is a floor.

    Read-only, for ``kirocrew doctor``: the directories older builds left behind
    are exactly the ones the sweep will never touch, so the operator has to hear
    about them somewhere. Counts names matching :data:`DERIVED_NAME_RE` without
    :data:`RUN_DIR_MARKER` inside; stops at *max_entries* names examined and
    says so, since a root this exists for can hold tens of thousands.
    """
    if platform_compat.is_link_or_junction(root):
        return 0, False
    unmarked = 0
    examined = 0
    try:
        with os.scandir(root) as it:
            for entry in it:
                if not DERIVED_NAME_RE.match(entry.name):
                    continue
                examined += 1
                if examined > max_entries:
                    return unmarked, True
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    marker = pinned_fs.lstat_by_name(root / entry.name / RUN_DIR_MARKER)
                except OSError:
                    continue
                if marker is None:
                    unmarked += 1
    except OSError:
        return unmarked, False
    return unmarked, False
