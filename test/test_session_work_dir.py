"""Per-run work directories are reclaimed when they hold only Crew's residue.

Hole 1 of the skill-view growth: every subagent and stateless cron run got a
``workspace_root()/<key>`` directory holding one file and nothing ever removed
it. These tests pin the rule that makes the removal safe -- only the two names
Crew writes, inside the two directories Crew creates, and ``rmdir`` for every
directory -- plus the two callers: the provider at shutdown and the bounded
sweep for what a crash or an older build left behind.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import unittest.mock
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import requires_symlinks
from kiro_crew import session_work_dir
from kiro_crew.acp.types import ACP_BACKEND_KIRO
from kiro_crew.config.loader import KiroCrewConfig
from kiro_crew.workspace_cli_settings import CLI_SETTINGS_LOCK_NAME


def _residue_dir(
    root: Path,
    name: str,
    *,
    age_secs: float = 0.0,
    marked: bool = True,
    owner_pid: int | str | None = None,
    child_pid: int | None = None,
    child_start_token: str | None = None,
) -> Path:
    """A run directory exactly as a marked spawn leaves it.

    *marked* False is the directory an older build left: same residue, no
    provenance marker, which only the shutdown path (whose provenance is the
    factory flag) may reclaim.
    """
    work_dir = root / name
    settings = work_dir / ".kiro" / "settings"
    settings.mkdir(parents=True)
    (settings / "cli.json").write_text(json.dumps({"chat.modelDefaults": {}}), encoding="utf-8")
    (settings / CLI_SETTINGS_LOCK_NAME).write_bytes(b"")
    if marked:
        owner = os.getpid() if owner_pid is None else owner_pid
        marker = str(owner)
        if child_pid is not None:
            marker += "\n" + json.dumps(
                {"pid": child_pid, "start": child_start_token},
                separators=(",", ":"),
                sort_keys=True,
            )
        (work_dir / session_work_dir.RUN_DIR_MARKER).write_text(marker, encoding="ascii")
    if age_secs:
        old = time.time() - age_secs
        for path in (
            work_dir,
            work_dir / ".kiro",
            settings,
            *settings.iterdir(),
            *work_dir.iterdir(),
        ):
            if path.exists():
                os.utime(path, (old, old))
    return work_dir


class TestDisposableSessionKey:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("cron:aaaa1111:bbbb2222", True),
            ("cron:aaaa1111:myagent", False),
            ("cron:aaaa1111", False),
            ("subagent:0123456789abcdef", True),
            ("subagent:0123abcd", True),
            ("subagent:0123abc", False),
            ("subagent:notes", False),
        ],
    )
    def test_only_generated_one_run_key_shapes_are_disposable(
        self, key: str, expected: bool
    ) -> None:
        assert session_work_dir.is_disposable_session_key(key) is expected
        name = key.replace(":", "_")
        assert (session_work_dir.DERIVED_NAME_RE.fullmatch(name) is not None) is expected


class TestReclaimRule:
    def test_a_residue_only_directory_is_removed(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        assert session_work_dir.reclaim_session_work_dir(work_dir) is True
        assert not work_dir.exists()

    def test_an_emptied_prefix_of_the_chain_is_still_removed(self, tmp_path: Path) -> None:
        """A run whose overlay was never written, or already cleared, is residue too."""
        bare = tmp_path / "subagent_00000001"
        bare.mkdir()
        (tmp_path / "subagent_00000002" / ".kiro").mkdir(parents=True)
        assert session_work_dir.reclaim_session_work_dir(bare) is True
        assert session_work_dir.reclaim_session_work_dir(tmp_path / "subagent_00000002") is True
        assert not bare.exists() and not (tmp_path / "subagent_00000002").exists()

    @pytest.mark.parametrize(
        "plant",
        [
            lambda d: (d / "report.md").write_text("the run's output", encoding="utf-8"),
            lambda d: (d / "src").mkdir(),
            lambda d: (d / ".kiro" / "steering").mkdir(),
            lambda d: (d / ".kiro" / "settings" / "mcp.json").write_text("{}", encoding="utf-8"),
            lambda d: (d / ".git").mkdir(),
        ],
    )
    def test_anything_that_is_not_residue_keeps_the_whole_directory(
        self, tmp_path: Path, plant
    ) -> None:
        """THE safety property: a file the run wrote is never reachable by this code."""
        work_dir = _residue_dir(tmp_path, "subagent_cafef00d")
        plant(work_dir)
        before = sorted(str(p.relative_to(work_dir)) for p in work_dir.rglob("*"))
        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        after = sorted(str(p.relative_to(work_dir)) for p in work_dir.rglob("*"))
        assert after == before, "a refusal must leave the tree exactly as it was"

    def test_a_missing_directory_is_a_quiet_no(self, tmp_path: Path) -> None:
        assert session_work_dir.reclaim_session_work_dir(tmp_path / "gone") is False

    def test_a_young_directory_is_kept_when_an_age_is_required(self, tmp_path: Path) -> None:
        """The sweep's grace: a spawn between mkdir and registration is not dead."""
        fresh = _residue_dir(tmp_path, "cron_aaaa_bbbb")
        assert session_work_dir.reclaim_session_work_dir(fresh, min_age_secs=3600) is False
        assert fresh.exists()
        old = _residue_dir(tmp_path, "cron_cccc_dddd", age_secs=7200)
        assert session_work_dir.reclaim_session_work_dir(old, min_age_secs=3600) is True

    def test_a_fresh_overlay_write_renews_an_old_directory(self, tmp_path: Path) -> None:
        """Age is the NEWEST mtime in the tree: a live run rewriting cli.json is young."""
        work_dir = _residue_dir(tmp_path, "subagent_01234567", age_secs=7200)
        (work_dir / ".kiro" / "settings" / "cli.json").write_text("{}", encoding="utf-8")
        assert session_work_dir.reclaim_session_work_dir(work_dir, min_age_secs=3600) is False

    @requires_symlinks
    def test_a_linked_work_directory_is_refused(self, tmp_path: Path) -> None:
        real = _residue_dir(tmp_path, "elsewhere")
        link = tmp_path / "subagent_11111111"
        link.symlink_to(real, target_is_directory=True)
        assert session_work_dir.reclaim_session_work_dir(link) is False
        assert real.exists() and (real / ".kiro" / "settings" / "cli.json").exists()

    @requires_symlinks
    def test_a_link_inside_the_chain_is_refused(self, tmp_path: Path) -> None:
        """A run could plant ``.kiro/settings -> <somewhere>``; nothing there is unlinked."""
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "cli.json").write_text("precious", encoding="utf-8")
        work_dir = tmp_path / "subagent_22222222"
        (work_dir / ".kiro").mkdir(parents=True)
        (work_dir / ".kiro" / "settings").symlink_to(victim, target_is_directory=True)
        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert (victim / "cli.json").read_text(encoding="utf-8") == "precious"

    def test_the_marker_is_the_only_provenance_the_sweep_accepts(self, tmp_path: Path) -> None:
        """A run-prefixed name under the workspace root is not provenance.

        Someone can make ``workspace_root()/subagent_deadbeef`` and put their own
        ``.kiro/settings/cli.json`` in it; by name and shape it is residue. No
        marker means Crew never derived it, so the sweep must leave it however
        old it is -- and so must the reclaim of any directory an older build made.
        """
        theirs = _residue_dir(tmp_path, "subagent_deadbeef", age_secs=10**6, marked=False)
        assert session_work_dir.reclaim_session_work_dir(theirs, require_marker=True) is False
        assert (theirs / ".kiro" / "settings" / "cli.json").exists()
        # The shutdown path's provenance is the factory flag, so it does not
        # need the file.
        assert session_work_dir.reclaim_session_work_dir(theirs) is True

    def test_shutdown_reclaim_refuses_a_live_foreign_owner(self, tmp_path: Path) -> None:
        foreign_pid = os.getppid()
        assert foreign_pid != os.getpid()
        assert (
            session_work_dir.platform_compat.pid_liveness(foreign_pid)
            != session_work_dir.platform_compat.PID_DEAD
        )
        work_dir = _residue_dir(
            tmp_path,
            "subagent_00000003",
            owner_pid=foreign_pid,
        )

        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert (work_dir / session_work_dir.RUN_DIR_MARKER).exists()
        assert (work_dir / ".kiro" / "settings" / "cli.json").exists()

    def test_dead_gateway_without_a_child_record_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_00000004", owner_pid=12345)
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_DEAD,
        )

        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert work_dir.exists()

    def test_dead_gateway_and_dead_child_are_reclaimed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(
            tmp_path,
            "subagent_00000008",
            owner_pid=12345,
            child_pid=23456,
            child_start_token="child-start",
        )
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_DEAD,
        )

        assert session_work_dir.reclaim_session_work_dir(work_dir) is True
        assert not work_dir.exists()

    def test_dead_gateway_and_live_child_are_kept_by_reclaim_and_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_pid = os.getpid()
        child_start = session_work_dir.platform_compat.get_process_start_id(child_pid)
        assert child_start is not None
        direct = _residue_dir(
            tmp_path,
            "subagent_0000000000000008",
            age_secs=7200,
            owner_pid=12345,
            child_pid=child_pid,
            child_start_token=child_start,
        )
        swept = _residue_dir(
            tmp_path,
            "subagent_0000000000000009",
            age_secs=7200,
            owner_pid=12345,
            child_pid=child_pid,
            child_start_token=child_start,
        )
        real_liveness = session_work_dir.platform_compat.pid_liveness
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: (
                session_work_dir.platform_compat.PID_DEAD if pid == 12345 else real_liveness(pid)
            ),
        )

        assert session_work_dir.reclaim_session_work_dir(direct) is False
        assert session_work_dir.sweep_disposable_work_dirs(tmp_path, live_work_dirs=[]) == 0
        assert direct.exists() and swept.exists()

    def test_recycled_child_pid_allows_reclaim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        child_pid = os.getpid()
        work_dir = _residue_dir(
            tmp_path,
            "subagent_0000000a",
            owner_pid=12345,
            child_pid=child_pid,
            child_start_token="a-different-process",
        )
        real_liveness = session_work_dir.platform_compat.pid_liveness
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: (
                session_work_dir.platform_compat.PID_DEAD if pid == 12345 else real_liveness(pid)
            ),
        )

        assert session_work_dir.reclaim_session_work_dir(work_dir) is True

    def test_own_gateway_with_dead_child_is_reclaimed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(
            tmp_path,
            "subagent_0000000b",
            child_pid=23456,
            child_start_token="gone",
        )
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_DEAD,
        )
        assert session_work_dir.reclaim_session_work_dir(work_dir) is True

    def test_malformed_child_record_fails_closed(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_0000000c")
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        marker.write_text(f"{os.getpid()}\n{{not-json", encoding="ascii")

        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert marker.exists()

    def test_live_child_without_start_token_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(
            tmp_path,
            "subagent_0000000d",
            owner_pid=12345,
            child_pid=os.getpid(),
            child_start_token=None,
        )
        real_liveness = session_work_dir.platform_compat.pid_liveness
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: (
                session_work_dir.platform_compat.PID_DEAD if pid == 12345 else real_liveness(pid)
            ),
        )
        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert work_dir.exists()

    def test_child_record_does_not_replace_a_foreign_marker(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_0000000e", owner_pid=os.getppid())
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        original = marker.read_bytes()

        assert session_work_dir.record_run_dir_child(work_dir, os.getpid(), "token") is False
        assert marker.read_bytes() == original

    def test_shutdown_reclaim_accepts_its_own_owner(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_00000005", owner_pid=os.getpid())

        assert session_work_dir.reclaim_session_work_dir(work_dir) is True
        assert not work_dir.exists()

    def test_shutdown_reclaim_refuses_a_malformed_owner(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_00000006", owner_pid="not-a-pid")

        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert (work_dir / session_work_dir.RUN_DIR_MARKER).exists()
        assert (work_dir / ".kiro" / "settings" / "cli.json").exists()

    def test_shutdown_reclaim_still_accepts_no_marker(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_00000007", marked=False)

        assert session_work_dir.reclaim_session_work_dir(work_dir) is True
        assert not work_dir.exists()

    def test_a_marked_directory_is_reclaimed_marker_included(self, tmp_path: Path) -> None:
        marked = _residue_dir(tmp_path, "subagent_0badf00d", age_secs=7200)
        assert session_work_dir.reclaim_session_work_dir(marked, require_marker=True) is True
        assert not marked.exists()

    def test_a_marked_directory_that_gained_a_file_is_kept(self, tmp_path: Path) -> None:
        marked = _residue_dir(tmp_path, "subagent_0badf00d", age_secs=7200)
        (marked / "notes.txt").write_text("mine", encoding="utf-8")
        assert session_work_dir.reclaim_session_work_dir(marked, require_marker=True) is False
        assert (marked / "notes.txt").exists() and (
            marked / session_work_dir.RUN_DIR_MARKER
        ).exists()

    @requires_symlinks
    def test_a_linked_marker_is_not_a_marker(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_00000009", age_secs=7200, marked=False)
        real = tmp_path / "real-marker"
        real.write_bytes(b"")
        (work_dir / session_work_dir.RUN_DIR_MARKER).symlink_to(real)
        assert session_work_dir.reclaim_session_work_dir(work_dir, require_marker=True) is False
        assert work_dir.exists()

    def test_mark_run_dir_creates_and_is_idempotent(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "subagent_11111111"
        assert session_work_dir.mark_run_dir(work_dir) is True
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        assert marker.is_file() and marker.read_text(encoding="ascii") == str(os.getpid())
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert marker.read_text(encoding="ascii") == str(os.getpid())
        child_pid = os.getpid()
        child_start = session_work_dir.platform_compat.get_process_start_id(child_pid)
        assert child_start is not None
        assert session_work_dir.record_run_dir_child(work_dir, child_pid, child_start) is True
        child_record = marker.read_text(encoding="ascii")
        assert child_record.splitlines() == [
            str(os.getpid()),
            json.dumps(
                {"pid": child_pid, "start": child_start},
                separators=(",", ":"),
                sort_keys=True,
            ),
        ]
        assert len(child_record.encode("ascii")) <= 4096
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert marker.read_text(encoding="ascii") == child_record

    def test_mark_run_dir_does_not_steal_a_live_foreign_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_11111111", owner_pid=12345)
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_ALIVE,
        )
        assert session_work_dir.mark_run_dir(work_dir) is False
        assert marker.read_text(encoding="ascii") == "12345"

    def test_mark_run_dir_replaces_only_a_confirmed_dead_owner_and_child(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(
            tmp_path,
            "subagent_11111111",
            owner_pid=12345,
            child_pid=23456,
            child_start_token="gone",
        )
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_DEAD,
        )
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert marker.read_text(encoding="ascii") == str(os.getpid())

    def test_mark_run_dir_keeps_a_dead_gateway_without_child_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_11111112", owner_pid=12345)
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_DEAD,
        )
        assert session_work_dir.mark_run_dir(work_dir) is False
        assert marker.read_text(encoding="ascii") == "12345"

    @pytest.mark.parametrize("owner", ["garbled", "", "0", str(2**31)])
    def test_mark_run_dir_refuses_an_unverifiable_owner(self, tmp_path: Path, owner: str) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_11111111", owner_pid=owner)
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        assert session_work_dir.mark_run_dir(work_dir) is False
        assert marker.read_text(encoding="ascii") == owner

    @requires_symlinks
    def test_mark_run_dir_refuses_a_linked_directory(self, tmp_path: Path) -> None:
        real = tmp_path / "elsewhere"
        real.mkdir()
        link = tmp_path / "subagent_22222222"
        link.symlink_to(real, target_is_directory=True)
        assert session_work_dir.mark_run_dir(link) is False
        assert not (real / session_work_dir.RUN_DIR_MARKER).exists()

    def test_counting_unmarked_leftovers_for_the_doctor(self, tmp_path: Path) -> None:
        _residue_dir(tmp_path, "subagent_0000000000000001", marked=False)
        _residue_dir(tmp_path, "cron_aaaaaaaa_bbbbbbbb", marked=False)
        _residue_dir(tmp_path, "subagent_0000000000000002")
        _residue_dir(tmp_path, "subagent_notes", marked=False)
        (tmp_path / "subagent_0000000000000003").write_text("a file, not a dir", encoding="utf-8")
        assert session_work_dir.count_unmarked_run_dirs(tmp_path) == (2, False)
        assert session_work_dir.count_unmarked_run_dirs(tmp_path, max_entries=1) == (1, True)
        assert session_work_dir.count_unmarked_run_dirs(tmp_path / "gone") == (0, False)


class TestByNameForm:
    """The platform form without descriptor-relative opens (Windows), driven here.

    Same rule, same refusals as the pinned walk; these tests drive it directly
    on a POSIX host so the contract is pinned on every platform the suite runs.
    """

    @pytest.fixture(autouse=True)
    def _unpinned(self, monkeypatch):
        monkeypatch.setattr(session_work_dir.pinned_fs, "supports_pinned_walk", lambda: False)

    def test_marked_residue_is_removed_marker_included(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_0000000a", age_secs=7200)
        assert session_work_dir.reclaim_session_work_dir(work_dir, require_marker=True) is True
        assert not work_dir.exists()

    def test_unmarked_residue_is_kept_when_the_marker_is_required(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_0000000b", age_secs=7200, marked=False)
        assert session_work_dir.reclaim_session_work_dir(work_dir, require_marker=True) is False
        assert (work_dir / ".kiro" / "settings" / "cli.json").exists()
        assert session_work_dir.reclaim_session_work_dir(work_dir) is True

    @pytest.mark.parametrize(
        "plant",
        [
            lambda d: (d / "report.md").write_text("output", encoding="utf-8"),
            lambda d: (d / ".kiro" / "steering").mkdir(),
            lambda d: (d / ".kiro" / "settings" / "mcp.json").write_text("{}", encoding="utf-8"),
        ],
    )
    def test_anything_that_is_not_residue_keeps_the_directory(self, tmp_path: Path, plant):
        work_dir = _residue_dir(tmp_path, "subagent_0000000c", age_secs=7200)
        plant(work_dir)
        before = sorted(str(p.relative_to(work_dir)) for p in work_dir.rglob("*"))
        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert sorted(str(p.relative_to(work_dir)) for p in work_dir.rglob("*")) == before

    def test_emptied_prefixes_missing_and_young_directories(self, tmp_path: Path) -> None:
        bare = tmp_path / "subagent_0000000d"
        bare.mkdir()
        assert session_work_dir.reclaim_session_work_dir(bare) is True
        (tmp_path / "subagent_0000000e" / ".kiro").mkdir(parents=True)
        assert session_work_dir.reclaim_session_work_dir(tmp_path / "subagent_0000000e") is True
        assert session_work_dir.reclaim_session_work_dir(tmp_path / "gone") is False
        (tmp_path / "a-file").write_text("x", encoding="utf-8")
        assert session_work_dir.reclaim_session_work_dir(tmp_path / "a-file") is False
        young = _residue_dir(tmp_path, "subagent_0000000f")
        assert session_work_dir.reclaim_session_work_dir(young, min_age_secs=3600) is False

    @requires_symlinks
    def test_links_are_refused_at_every_level(self, tmp_path: Path) -> None:
        real = _residue_dir(tmp_path, "elsewhere", age_secs=7200)
        link = tmp_path / "subagent_00000010"
        link.symlink_to(real, target_is_directory=True)
        assert session_work_dir.reclaim_session_work_dir(link) is False
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "cli.json").write_text("precious", encoding="utf-8")
        work_dir = tmp_path / "subagent_00000011"
        (work_dir / ".kiro").mkdir(parents=True)
        (work_dir / ".kiro" / "settings").symlink_to(victim, target_is_directory=True)
        assert session_work_dir.reclaim_session_work_dir(work_dir) is False
        assert (victim / "cli.json").read_text(encoding="utf-8") == "precious"
        linked_marker = _residue_dir(tmp_path, "subagent_00000012", age_secs=7200, marked=False)
        (tmp_path / "real-marker").write_bytes(b"")
        (linked_marker / session_work_dir.RUN_DIR_MARKER).symlink_to(tmp_path / "real-marker")
        assert (
            session_work_dir.reclaim_session_work_dir(linked_marker, require_marker=True) is False
        )

    def test_mark_run_dir_by_name_creates_and_is_idempotent(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "subagent_00000013"
        assert session_work_dir.mark_run_dir(work_dir) is True
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        assert marker.read_text(encoding="ascii") == str(os.getpid())
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert session_work_dir.reclaim_session_work_dir(work_dir, require_marker=True) is True

    def test_record_run_dir_child_by_name_preserves_the_record(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "subagent_00000016"
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert session_work_dir.record_run_dir_child(work_dir, 45678, "start-45678") is True
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        recorded = marker.read_text(encoding="ascii")
        assert recorded.splitlines() == [
            str(os.getpid()),
            '{"pid":45678,"start":"start-45678"}',
        ]
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert marker.read_text(encoding="ascii") == recorded

    def test_mark_run_dir_by_name_protects_live_and_replaces_dead_foreign_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(
            tmp_path,
            "subagent_00000013",
            owner_pid=12345,
            child_pid=23456,
            child_start_token="gone",
        )
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_ALIVE,
        )
        original = marker.read_text(encoding="ascii")
        assert session_work_dir.mark_run_dir(work_dir) is False
        assert marker.read_text(encoding="ascii") == original
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_DEAD,
        )
        assert session_work_dir.mark_run_dir(work_dir) is True
        assert marker.read_text(encoding="ascii") == str(os.getpid())

    def test_sweep_by_name_keeps_a_live_foreign_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        work_dir = _residue_dir(
            tmp_path,
            "subagent_0000000000000013",
            age_secs=7200,
            owner_pid=12345,
        )
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: session_work_dir.platform_compat.PID_ALIVE,
        )
        assert session_work_dir.sweep_disposable_work_dirs(tmp_path, live_work_dirs=[]) == 0
        assert work_dir.exists()

    @requires_symlinks
    def test_mark_run_dir_by_name_refuses_links(self, tmp_path: Path) -> None:
        real = tmp_path / "elsewhere"
        real.mkdir()
        link = tmp_path / "subagent_00000014"
        link.symlink_to(real, target_is_directory=True)
        assert session_work_dir.mark_run_dir(link) is False
        work_dir = tmp_path / "subagent_00000015"
        work_dir.mkdir()
        (tmp_path / "real-marker").write_bytes(b"")
        (work_dir / session_work_dir.RUN_DIR_MARKER).symlink_to(tmp_path / "real-marker")
        assert session_work_dir.mark_run_dir(work_dir) is False


class TestSweepBounds:
    def test_a_root_too_large_to_count_inside_the_budget_sweeps_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        for n in range(3):
            _residue_dir(tmp_path, f"subagent_{n:016x}", age_secs=7200)
        ticks = iter([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        monkeypatch.setattr(session_work_dir.time, "monotonic", lambda: next(ticks, 10.0))
        assert (
            session_work_dir.sweep_disposable_work_dirs(
                tmp_path, live_work_dirs=[], max_seconds=1.0
            )
            == 0
        )
        assert all((tmp_path / f"subagent_{n:016x}").exists() for n in range(3))

    def test_a_linked_root_and_an_unlistable_root_sweep_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setattr(session_work_dir.platform_compat, "is_link_or_junction", lambda p: True)
        assert session_work_dir.sweep_disposable_work_dirs(tmp_path, live_work_dirs=[]) == 0
        monkeypatch.setattr(
            session_work_dir.platform_compat, "is_link_or_junction", lambda p: False
        )
        assert (
            session_work_dir.sweep_disposable_work_dirs(tmp_path / "gone", live_work_dirs=[]) == 0
        )
        assert session_work_dir.count_unmarked_run_dirs(tmp_path / "gone") == (0, False)


class TestSweep:
    def test_reclaims_idle_marked_residue_and_skips_live_young_foreign_and_unmarked(
        self, tmp_path: Path
    ) -> None:
        idle = _residue_dir(tmp_path, "subagent_0000000000000001", age_secs=7200)
        idle_cron = _residue_dir(tmp_path, "cron_11112222_33334444", age_secs=7200)
        persistent_cron = _residue_dir(tmp_path, "cron_11112222", age_secs=7200)
        live = _residue_dir(tmp_path, "subagent_0000000000000002", age_secs=7200)
        young = _residue_dir(tmp_path, "subagent_0000000000000003")
        dashboard = _residue_dir(tmp_path, "dashboard_slot-1", age_secs=7200)
        data = _residue_dir(tmp_path, "subagent_0000000000000004", age_secs=7200)
        (data / "out.txt").write_text("kept", encoding="utf-8")
        legacy = _residue_dir(tmp_path, "subagent_000000000000abcd", age_secs=10**6, marked=False)
        theirs = _residue_dir(tmp_path, "subagent_project", age_secs=10**6, marked=False)

        reclaimed = session_work_dir.sweep_disposable_work_dirs(
            tmp_path, live_work_dirs=[str(live)]
        )

        assert reclaimed == 2
        assert not idle.exists() and not idle_cron.exists()
        assert persistent_cron.exists(), "a persistent cron directory was swept"
        assert live.exists(), "a directory a registered session names was swept"
        assert young.exists(), "a directory inside the grace window was swept"
        assert dashboard.exists(), "a long-lived session's directory was swept"
        assert (data / "out.txt").exists()
        assert legacy.exists(), "an unmarked directory from an older build was swept"
        assert (
            theirs / ".kiro" / "settings" / "cli.json"
        ).exists(), "a person's own directory under a run prefix was swept"

    def test_sweep_keeps_live_foreign_and_garbled_owners_but_reclaims_dead_and_own(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        own = _residue_dir(tmp_path, "subagent_0000000000000011", age_secs=7200)
        live_foreign = _residue_dir(
            tmp_path,
            "subagent_0000000000000012",
            age_secs=7200,
            owner_pid=12345,
        )
        dead_foreign = _residue_dir(
            tmp_path,
            "subagent_0000000000000013",
            age_secs=7200,
            owner_pid=23456,
            child_pid=34567,
            child_start_token="gone",
        )
        garbled = _residue_dir(
            tmp_path,
            "subagent_0000000000000014",
            age_secs=7200,
            owner_pid="not-a-pid",
        )
        monkeypatch.setattr(
            session_work_dir.platform_compat,
            "pid_liveness",
            lambda pid: (
                session_work_dir.platform_compat.PID_ALIVE
                if pid == 12345
                else session_work_dir.platform_compat.PID_DEAD
            ),
        )

        assert session_work_dir.sweep_disposable_work_dirs(tmp_path, live_work_dirs=[]) == 2
        assert not own.exists() and not dead_foreign.exists()
        assert live_foreign.exists() and garbled.exists()

    def test_the_sweep_is_bounded_and_finishes_on_later_wakes(self, tmp_path: Path) -> None:
        dirs = [_residue_dir(tmp_path, f"subagent_{n:016x}", age_secs=7200) for n in range(10)]
        first = session_work_dir.sweep_disposable_work_dirs(
            tmp_path, live_work_dirs=[], max_entries=4
        )
        assert first == 4
        assert sum(d.exists() for d in dirs) == 6
        total = first
        for _ in range(3):
            total += session_work_dir.sweep_disposable_work_dirs(
                tmp_path, live_work_dirs=[], max_entries=4
            )
        assert total == 10 and not any(d.exists() for d in dirs)

    def test_a_missing_or_linked_root_sweeps_nothing(self, tmp_path: Path) -> None:
        assert session_work_dir.sweep_disposable_work_dirs(tmp_path / "no", live_work_dirs=[]) == 0


def _cfg(tmp_path: Path) -> KiroCrewConfig:
    cfg_file = tmp_path / "kirocrew.json"
    cfg_file.write_text(
        json.dumps({"agent": {"provider": "acp", "acp_backend": ACP_BACKEND_KIRO}}),
        encoding="utf-8",
    )
    with unittest.mock.patch("kiro_crew.config.loader.config_path", return_value=cfg_file):
        return KiroCrewConfig.load()


class TestFactoryMarksOnlyDerivedOneRunDirectories:
    """The provider learns from the factory which directories are its to reclaim."""

    @staticmethod
    def _captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **factory_kwargs) -> dict:
        import kiro_crew.providers.acp as acp_mod

        monkeypatch.setenv("KIROCREW_WORKSPACE", str(tmp_path / "ws"))
        seen: list[dict] = []

        class _FakeProvider:
            def __init__(self, **kwargs: object) -> None:
                seen.append(kwargs)

        with unittest.mock.patch.object(acp_mod, "AcpProvider", _FakeProvider):
            _cfg(tmp_path).create_provider_factory()(**factory_kwargs)
        return seen[0]

    @pytest.mark.parametrize("key", ["subagent:0123456789abcdef", "cron:aaaa1111:bbbb2222"])
    def test_a_derived_one_run_directory_is_marked(self, tmp_path, monkeypatch, key) -> None:
        got = self._captured(tmp_path, monkeypatch, session_key=key)
        assert got["disposable_work_dir"] is True
        assert Path(got["work_dir"]).parent == Path(os.path.realpath(tmp_path / "ws"))

    @pytest.mark.parametrize(
        "key",
        [
            "cron:aaaa1111",
            "cron:aaaa1111:myagent",
            "subagent:notes",
            "dashboard:slot-1",
            "slack:C123:456.789",
            None,
        ],
    )
    def test_a_long_lived_session_directory_is_not_marked(self, tmp_path, monkeypatch, key):
        got = self._captured(tmp_path, monkeypatch, session_key=key)
        assert got["disposable_work_dir"] is False

    def test_an_explicit_cwd_is_never_marked_whatever_the_key(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        got = self._captured(
            tmp_path, monkeypatch, session_key="subagent:0123abcd", cwd=str(project)
        )
        assert got["disposable_work_dir"] is False
        assert Path(got["work_dir"]) == project


class TestProviderReclaimsAtShutdown:
    @staticmethod
    def _provider(work_dir: Path, *, disposable: bool, claimed: bool = True):
        from kiro_crew.providers.acp import AcpProvider

        with unittest.mock.patch("kiro_crew.providers.acp.AcpClient"):
            provider = AcpProvider(acp_backend=ACP_BACKEND_KIRO, disposable_work_dir=disposable)
        provider._client = MagicMock()
        provider._client.backend = ACP_BACKEND_KIRO
        provider._client._work_dir = work_dir
        provider._client.session_id = None
        provider._client.process_tree_confirmed_dead = True
        provider._client.shutdown = AsyncMock()
        if claimed:

            @asynccontextmanager
            async def claim():
                yield True

            provider.set_work_dir_claim_probe(claim)
        return provider

    @pytest.mark.asyncio
    async def test_a_run_directory_disappears_when_its_run_ends(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        provider = self._provider(work_dir, disposable=True)
        await provider.shutdown()
        provider._client.shutdown.assert_awaited_once()
        assert not work_dir.exists()

    @pytest.mark.asyncio
    async def test_a_surviving_process_tree_keeps_the_run_directory(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        provider = self._provider(work_dir, disposable=True)
        provider._client.process_tree_confirmed_dead = False
        await provider.shutdown()
        assert work_dir.exists()

    @pytest.mark.asyncio
    async def test_an_unknown_process_tree_keeps_the_run_directory(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        provider = self._provider(work_dir, disposable=True)
        provider._client.process_tree_confirmed_dead = None
        await provider.shutdown()
        assert work_dir.exists()

    @pytest.mark.asyncio
    async def test_a_provider_without_a_registry_claim_keeps_the_run_directory(
        self, tmp_path: Path
    ) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        provider = self._provider(work_dir, disposable=True, claimed=False)
        await provider.shutdown()
        assert work_dir.exists()

    @pytest.mark.asyncio
    async def test_a_broken_registry_claim_keeps_the_run_directory(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        provider = self._provider(work_dir, disposable=True, claimed=False)

        @asynccontextmanager
        async def broken_claim():
            raise RuntimeError("registry unavailable")
            yield True  # pragma: no cover - makes this an async context manager

        provider.set_work_dir_claim_probe(broken_claim)
        await provider.shutdown()
        assert work_dir.exists()

    @pytest.mark.asyncio
    async def test_a_run_that_wrote_a_file_keeps_its_directory(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        (work_dir / "result.json").write_text("{}", encoding="utf-8")
        provider = self._provider(work_dir, disposable=True)
        await provider.shutdown()
        assert (work_dir / "result.json").exists()

    @pytest.mark.asyncio
    async def test_an_unmarked_directory_is_untouched(self, tmp_path: Path) -> None:
        work_dir = _residue_dir(tmp_path, "dashboard_slot-1")
        provider = self._provider(work_dir, disposable=False)
        await provider.shutdown()
        assert (work_dir / ".kiro" / "settings" / "cli.json").exists()

    @pytest.mark.asyncio
    async def test_start_marks_a_derived_directory_before_any_writer(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The marker is written on start, so a crash after it leaves a sweepable dir."""
        work_dir = tmp_path / "subagent_deadbeef"
        provider = self._provider(work_dir, disposable=True)
        provider._client.ensure_ready = AsyncMock()
        provider._client.memory_mode = "persistent"
        provider._client._pid = 44444
        provider._client._start_time = "start-44444"
        monkeypatch.setattr(provider, "_apply_effort_overlay", lambda: None)
        monkeypatch.setattr(provider, "_apply_tool_search_overlay", lambda: None)
        monkeypatch.setattr(type(provider), "is_acp_runtime_backend", property(lambda self: False))
        monkeypatch.setattr(provider, "_apply_initial_effort", AsyncMock())
        await provider.start()
        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        assert marker.read_text(encoding="ascii").splitlines() == [
            str(os.getpid()),
            '{"pid":44444,"start":"start-44444"}',
        ]
        # A caller's cwd is never marked.
        project = tmp_path / "project"
        project.mkdir()
        other = self._provider(project, disposable=False)
        other._client.ensure_ready = AsyncMock()
        other._client.memory_mode = "persistent"
        monkeypatch.setattr(other, "_apply_effort_overlay", lambda: None)
        monkeypatch.setattr(other, "_apply_tool_search_overlay", lambda: None)
        monkeypatch.setattr(other, "_apply_initial_effort", AsyncMock())
        await other.start()
        assert not (project / session_work_dir.RUN_DIR_MARKER).exists()

    @pytest.mark.asyncio
    async def test_start_records_the_shared_runtime_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kiro_crew.acp.session_provider import AcpSessionProvider

        work_dir = tmp_path / "subagent_feedface"
        provider = self._provider(work_dir, disposable=True)
        provider._client.memory_mode = "persistent"
        runtime = MagicMock()
        runtime.pid = 55555
        runtime._start_time = "start-55555"
        runtime._work_dir = work_dir
        inner = AcpSessionProvider(MagicMock(), runtime, owns_runtime=True)

        async def start_runtime() -> None:
            provider._client = inner

        monkeypatch.setattr(type(provider), "is_acp_runtime_backend", property(lambda self: True))
        monkeypatch.setattr(provider, "_start_kiro_runtime", start_runtime)
        monkeypatch.setattr(provider, "_apply_effort_overlay", lambda: None)
        monkeypatch.setattr(provider, "_apply_tool_search_overlay", lambda: None)
        monkeypatch.setattr(provider, "_apply_initial_effort", AsyncMock())

        await provider.start()

        marker = work_dir / session_work_dir.RUN_DIR_MARKER
        assert marker.read_text(encoding="ascii").splitlines() == [
            str(os.getpid()),
            '{"pid":55555,"start":"start-55555"}',
        ]

    @pytest.mark.asyncio
    async def test_a_reclaim_failure_never_fails_the_shutdown(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        work_dir = _residue_dir(tmp_path, "subagent_deadbeef")
        provider = self._provider(work_dir, disposable=True)

        def boom(*args, **kwargs):
            raise RuntimeError("filesystem said no")

        import kiro_crew.providers.acp as acp_mod

        monkeypatch.setattr(acp_mod, "reclaim_session_work_dir", boom)
        await provider.shutdown()
        assert work_dir.exists()


class TestRegistryLeavesASiblingsDirectory:
    """Two providers for one KEY derive one directory; the discarded one leaves it.

    The factory flag says "derived from a one-run key", not "this instance is
    the one the registry kept". Where the registry shuts down a provider whose
    key another live provider holds -- the loser of a cold-start race, the
    predecessor of a recycle whose successor is already registered, the
    bootstrap provider that borrowed a parent's key -- it first tells that
    provider the directory is not its to reclaim. The live sibling still
    reclaims at its own shutdown, which is what the ``close_all`` at the end
    of each test pins.
    """

    KEY = "subagent:0123abcd"

    @staticmethod
    def _factory(work_dir: Path, made: list, gate: "asyncio.Event | None" = None):
        def factory(session_key=None, agent=None, channel_id=None, **kwargs):
            provider = TestProviderReclaimsAtShutdown._provider(work_dir, disposable=True)
            provider._client._session_id = ""
            provider._client._pid = None
            provider._client._runtime = None

            async def _start() -> None:
                if gate is not None:
                    await gate.wait()

            provider.start = _start  # type: ignore[method-assign]
            provider.is_process_alive = lambda: True  # type: ignore[method-assign]
            provider.is_alive = lambda: True  # type: ignore[method-assign]
            provider.context_usage_pct = lambda: 0.0  # type: ignore[method-assign]
            provider.context_window_tokens = lambda: 0  # type: ignore[method-assign]
            provider.has_active_turn = lambda: False  # type: ignore[method-assign]
            provider.runtime_info = lambda: (None, None)  # type: ignore[method-assign]
            made.append(provider)
            return provider

        return factory

    @pytest.mark.asyncio
    async def test_the_loser_of_a_cold_start_race_leaves_the_winners_directory(
        self, tmp_path: Path
    ) -> None:
        from kiro_crew.session import SessionManager

        work_dir = _residue_dir(tmp_path, "subagent_0123abcd")
        made: list = []
        gate = asyncio.Event()
        mgr = SessionManager(_cfg(tmp_path), provider_factory=self._factory(work_dir, made, gate))
        first = asyncio.create_task(mgr.get_or_create(self.KEY))
        second = asyncio.create_task(mgr.get_or_create(self.KEY))
        await asyncio.sleep(0.05)
        gate.set()
        done, pending = await asyncio.wait({first, second}, timeout=3.0)
        assert len(done) == 1 and len(pending) == 1, "exactly one cold start wins the key"
        await asyncio.sleep(0.05)

        winner = mgr._sessions[mgr._fold_key(self.KEY)].provider
        (loser,) = [p for p in made if p is not winner]
        loser._client.shutdown.assert_awaited_once()
        assert work_dir.exists(), "the discarded provider reclaimed the live sibling's cwd"
        assert loser._disposable_work_dir is False
        assert winner._disposable_work_dir is True

        mgr.release(self.KEY)
        await asyncio.wait_for(next(iter(pending)), timeout=3.0)
        mgr.release(self.KEY)
        await mgr.close_all()
        assert not work_dir.exists(), "the winner still owned the directory at its shutdown"

    @pytest.mark.asyncio
    async def test_a_recycled_session_leaves_its_registered_successors_directory(
        self, tmp_path: Path
    ) -> None:
        from kiro_crew.session import SessionManager

        work_dir = _residue_dir(tmp_path, "subagent_0123abcd")
        made: list = []
        mgr = SessionManager(_cfg(tmp_path), provider_factory=self._factory(work_dir, made))
        key = mgr._fold_key(self.KEY)
        await mgr.get_or_create(self.KEY)
        predecessor = mgr._sessions[key]
        mgr.release(self.KEY)
        # A successor registers while the predecessor is being recycled: the
        # marker is what lets allocation treat the key as free.
        mgr._recycling[key] = predecessor
        await mgr.get_or_create(self.KEY)
        mgr.release(self.KEY)
        mgr._recycling.pop(key, None)
        successor = mgr._sessions[key]
        assert successor is not predecessor

        await mgr._recycle_held(key, predecessor, 95.0)

        predecessor.provider._client.shutdown.assert_awaited_once()
        assert mgr._sessions[key] is successor
        assert work_dir.exists(), "the recycled predecessor reclaimed its successor's cwd"
        assert predecessor.provider._disposable_work_dir is False
        assert successor.provider._disposable_work_dir is True
        await mgr.close_all()
        assert not work_dir.exists()

    @pytest.mark.asyncio
    async def test_a_successor_registering_during_shutdown_keeps_its_directory(
        self, tmp_path: Path
    ) -> None:
        """The final registry claim spans process teardown and filesystem reclaim."""
        from kiro_crew.session import SessionManager

        work_dir = _residue_dir(tmp_path, "subagent_0123abcd")
        made: list = []
        mgr = SessionManager(_cfg(tmp_path), provider_factory=self._factory(work_dir, made))
        key = mgr._fold_key(self.KEY)
        await mgr.get_or_create(self.KEY)
        predecessor = mgr._sessions[key]
        mgr.release(self.KEY)

        async def register_successor() -> None:
            await mgr.get_or_create(self.KEY)
            mgr.release(self.KEY)

        predecessor.provider._client.shutdown = AsyncMock(side_effect=register_successor)
        await mgr._recycle_held(key, predecessor, 95.0)

        successor = mgr._sessions[key]
        assert successor is not predecessor
        assert predecessor.provider._disposable_work_dir is True
        assert work_dir.exists(), "the predecessor reclaimed a successor registered mid-shutdown"
        await mgr.close_all()
        assert not work_dir.exists()

    @pytest.mark.asyncio
    async def test_a_recycle_with_no_successor_still_reclaims(self, tmp_path: Path) -> None:
        """The disown is keyed on a successor being registered, not on recycling."""
        from kiro_crew.session import SessionManager

        work_dir = _residue_dir(tmp_path, "subagent_0123abcd")
        made: list = []
        mgr = SessionManager(_cfg(tmp_path), provider_factory=self._factory(work_dir, made))
        key = mgr._fold_key(self.KEY)
        await mgr.get_or_create(self.KEY)
        session = mgr._sessions[key]
        await mgr._recycle_held(key, session, 95.0)
        assert key not in mgr._sessions
        assert not work_dir.exists()
        await mgr.close_all()

    @pytest.mark.asyncio
    async def test_a_runtime_bootstrap_provider_leaves_the_parents_directory(
        self, tmp_path: Path
    ) -> None:
        """A bootstrap provider borrows the PARENT's key, so it derives the parent's cwd."""
        from kiro_crew.session import SessionManager

        work_dir = _residue_dir(tmp_path, "subagent_0123abcd")
        made: list = []
        mgr = SessionManager(_cfg(tmp_path), provider_factory=self._factory(work_dir, made))
        adopted = MagicMock()
        mgr.get_subagent_runtime = AsyncMock(return_value=adopted)  # type: ignore[method-assign]

        # ``_runtime`` is None on the bootstrap provider's client, so no runtime
        # is adopted and the provider is shut down instead.
        assert await mgr._get_or_bootstrap_run_runtime(self.KEY, agent="kirocrew") is adopted

        (bootstrap,) = made
        bootstrap._client.shutdown.assert_awaited_once()
        assert bootstrap._disposable_work_dir is False
        assert work_dir.exists(), "the bootstrap provider reclaimed the parent's cwd"
