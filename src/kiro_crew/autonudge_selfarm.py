"""Authenticated record of which nudge loops a crew/member session armed ITSELF.

``NudgeLoop.self_armed`` is the one bit that relaxes the crew/member
external-arm refusal at fire time (``GatewayOrchestrator._fire_dashboard_nudge``),
and the loop store it lives in (``autonudge.json``) is agent-writable: an
agent -- or a prompt injected into one -- can write ``"self_armed": true`` on a
loop it did not arm from that session, and a restart would restore it. The
persisted bit alone is therefore not authorization; it is a hint that must
AGREE with a record the agent cannot forge.

That record lives here, in a GATEWAY-ONLY directory of its own under the data
home, ``autonudge-trust/`` (:func:`self_arm_record_path`), with all three ways
of writing it closed on that one name: the agent's file tools are fenced by
``security._CREW_SECRET_LEAVES``, a sandboxed shell -- a command that builds the
path at runtime, which no text matcher sees -- is bind-masked by
``sandbox._CREW_HIDDEN_LEAVES`` (pre-created before every spawn so the mask has a
name to cover on a fresh install, and refused when aliased by a link), and no
in-sandbox code opens it: only gateway code opening the path directly -- the
authorizer at the moment it admits an arm, the owner's switch, the store's
remove -- ever reads or writes it. NOT under ``trust/``: that directory is a
declared sandbox READ-WRITE exception (in-sandbox MCP servers append to the
audit log there and ``verify_session_pid`` reads the SEL key), so a record
under it stayed writable by a same-UID sandboxed command, and an entry is the
whole of an owner arm's fire-time admission. A record an upgraded install still
has at the ``trust/`` layout is DISCARDED on first access, not migrated
(:func:`_retire_legacy_record`): its entries may be the sandbox's, so the loops
behind them are refused until armed again. The fire-time guard admits a crew/member wake
only when BOTH hold: ``loop.self_armed is True`` on the record it loaded AND
``is_recorded_self_arm(loop.id, loop.slot_key)`` here. A forged bit with no
trust entry refuses; a stale trust entry with no bit refuses.

One flat JSON object ``{loop_id: {"slot_key": ..., "armed_ts": ...}}`` under a
SEAL: an HMAC over the entries keyed by the gateway's ``token_signing.key``
(:func:`_seal`), a secret every shipping build already masks from the sandbox
and fences from the agent's file tools. The directory mask above closes the
record to the sandbox from the boot that first masks it; it says nothing about
bytes planted at the same path BEFORE that boot, when the name was an ordinary
writable child of the data home. The seal does: a record this gateway did not
write under its current key cannot carry a valid seal, so a pre-upgrade plant
reads as nothing recorded (refuse) and is moved aside by the next writer
(:func:`_quarantine_unsealed_record`). A token-key rotation breaks every seal
the same way -- the safe side; the loops behind them are armed again. An entry
is written by exactly one path (the authorizer admitting a self-arm) and REVOKED
by exactly one path: its loop leaving the store (``AutoNudgeService.remove_sync``
calls :func:`forget_self_arm` on remove and on replacement), so a removed loop's
id cannot keep its authorization and the file cannot grow past the loops that
were armed and not yet removed. A write never prunes against a caller-supplied
view of the store -- see :func:`record_self_arm` for the race that would open.
Every read-modify-write runs under an exclusive file lock so two concurrent
self-arms cannot drop each other's entries. Every reader is TOTAL: a
missing, unreadable, malformed or unsealed file reads as "not recorded", which is the
refusing answer.

Blocking file IO throughout -- async callers offload via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

from kiro_crew import platform_compat
from kiro_crew.atomic_write import atomic_write
from kiro_crew.config.paths import data_home

logger = logging.getLogger(__name__)

SELF_ARM_RECORD_NAME = "autonudge-self-armed.json"
_LOCK_NAME = SELF_ARM_RECORD_NAME + ".lock"

#: The record's own directory under the data home. A direct child of the data
#: home (the sandbox pre-create is a plain ``mkdir`` of that shape), named for
#: what it holds and for nothing else, so no other writer has a reason to share
#: it: ``sandbox._CREW_HIDDEN_LEAVES`` masks it whole, ``_CREW_NO_ALIAS_LEAVES``
#: refuses a link at the name, and ``security._CREW_SECRET_LEAVES`` fences it.
ARM_RECORD_DIRNAME = "autonudge-trust"

#: The directory an older layout kept the record in. Read only by
#: :func:`_retire_legacy_record`, which deletes what it finds there.
_LEGACY_DIRNAME = "trust"

#: ``armed_by`` values an entry may carry. Absent reads as ``ARMED_BY_SELF``.
ARMED_BY_SELF = "self"
ARMED_BY_OWNER = "owner"

#: Domain separator for the record's seal, so a value minted here can never
#: validate as a dashboard token, a tag-grant certificate, or any other HMAC
#: the same key signs -- and vice versa.
_SEAL_DOMAIN = b"kiro-crew:autonudge-trust:seal:v1\x00"

#: Where :func:`_quarantine_unsealed_record` moves a parseable record that does
#: not carry this gateway's seal: beside the record, same masked directory, as
#: evidence rather than deleted. One name, so a second quarantine replaces the
#: first and the directory cannot grow with plants.
_UNSEALED_SUFFIX = ".unsealed"


def _seal_secret() -> bytes:
    """The key the seal is minted under: the dashboard's ``token_signing.key``.

    Chosen because it is the one gateway secret that every build shipping this
    record already keeps out of both agent planes (``sandbox._CREW_HIDDEN_LEAVES``
    masks it from spawned commands, ``security._CREW_SECRET_LEAVES`` fences it
    from the file tools), so a plant written before THIS leaf was masked could
    not have read it. Lazy import: ``token_secret`` must not be loaded -- and
    must not create its key file -- on import of this module.
    """
    from kiro_crew.dashboard import token_secret

    return token_secret._get_secret()


def _seal(entries: dict[str, Any]) -> str:
    """HMAC-SHA256 over the canonical JSON of *entries* under :func:`_seal_secret`."""
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hmac.new(
        _seal_secret(), _SEAL_DOMAIN + canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _is_sealed(raw: dict[str, Any], entries: dict[str, Any]) -> bool:
    """Whether *raw* carries a seal that verifies over *entries*.

    Total: a missing or non-string seal, or a key that cannot be loaded, is
    ``False`` -- the refusing answer -- never an exception into a reader.
    """
    seal = raw.get("seal")
    if not isinstance(seal, str):
        return False
    try:
        return hmac.compare_digest(seal, _seal(entries))
    except Exception:
        logger.warning("autonudge trust record seal could not be checked", exc_info=True)
        return False


@contextlib.contextmanager
def _record_lock() -> Iterator[None]:
    """Exclusive lock spanning one read-modify-write transaction.

    Two self-arms committing at once (two members arming in the same second)
    would otherwise each read the pre-transaction file and the second write
    would drop the first's entry -- a loop the authorizer reported as armed
    that the fire-time guard then refuses. A sibling lock file rather than the
    record itself, because ``atomic_write`` replaces the record's inode.
    """
    path = self_arm_record_path()
    _ensure_record_dir(path.parent)
    lock_path = path.parent / _LOCK_NAME
    with open(lock_path, "a+", encoding="utf-8") as fh:
        with platform_compat.file_lock(fh.fileno(), exclusive=True):
            _retire_legacy_record()
            _quarantine_unsealed_record()
            yield


def self_arm_record_path() -> Path:
    """Absolute path of the record: ``<data home>/autonudge-trust/<name>``."""
    return data_home() / ARM_RECORD_DIRNAME / SELF_ARM_RECORD_NAME


def _legacy_record_path() -> Path:
    return data_home() / _LEGACY_DIRNAME / SELF_ARM_RECORD_NAME


def _ensure_record_dir(directory: Path) -> None:
    """Create the record's directory owner-only; tighten best-effort if present.

    The sandbox pre-create (``sandbox._CREW_PRECREATE_HIDDEN_DIR_LEAVES``) makes
    the same directory at 0o700 before every spawn; this is the gateway's own
    path to it for the first write on a host that never spawned a sandbox. A
    ``parents=True`` mkdir could leave a default-mode directory behind, so the
    mode is tightened after -- the fence and the mask are the real boundary,
    the mode is hygiene.
    """
    directory.mkdir(parents=True, exist_ok=True)
    try:
        platform_compat.restrict_dir_to_owner(directory)
    except OSError:
        logger.debug("could not tighten mode on %s", directory, exc_info=True)


def _retire_legacy_record() -> None:
    """Delete a record left at the ``trust/`` layout. It is NOT migrated.

    That file sat in a directory the sandbox keeps read-write, so anything in
    it may have been written by a sandboxed command rather than by the
    authorizer -- carrying it into the masked leaf would launder a forged entry
    into a trusted one, and there is no way to tell the two apart after the
    fact. So the whole file is discarded, and every loop armed before the
    upgrade is refused at fire time (the safe side, audited by the fire guard)
    until it is armed again: the owner's switch goes OFF then ON, a crewmate's
    own loop is re-armed from its turn. One-time and gateway-only, called under
    the record lock by every writer and (lockless) by a reader that finds the
    new path absent; idempotent once the file is gone. Best-effort: an
    ``OSError`` is logged and the caller proceeds on the new path.
    """
    legacy = _legacy_record_path()
    try:
        if not legacy.is_file():
            return
        legacy.unlink()
        logger.warning(
            "legacy autonudge trust record at %s discarded (that directory is writable "
            "from the agent sandbox, so its entries cannot be trusted); loops armed before "
            "this upgrade must be armed again",
            legacy,
        )
        with contextlib.suppress(OSError):
            (legacy.parent / _LOCK_NAME).unlink()
    except OSError:
        logger.warning("could not discard the legacy autonudge trust record", exc_info=True)


def _load_record_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """``(raw, entries)`` of a parseable record of the expected shape, else ``None``.

    Raises what ``read_text``/``json.loads`` raise; the callers decide what an
    unreadable or mis-shaped file means to them (the total readers say ``{}``,
    the strict one raises). A ``None`` is the mis-shaped case only.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("loops") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return None
    return raw, entries


def _quarantine_unsealed_record() -> None:
    """Move a parseable record that does not carry this gateway's seal aside.

    Called under :func:`_record_lock` by every writer, so no reader can be
    quarantining a file a writer just sealed: the writer holding the lock is
    the only mover. A file that does not PARSE is left where it is -- the
    strict reader refuses the write on it and keeps it as evidence, since a
    torn write of the gateway's own is a possibility there; a file that parses
    to the right shape but carries no valid seal has exactly one story (not
    written by this gateway under its current key -- a plant from before the
    leaf was masked, or a record from before a token-key rotation) and no
    reading under which its entries may be kept. Best-effort: an ``OSError``
    is logged and the strict reader still refuses its content.
    """
    path = self_arm_record_path()
    try:
        loaded = _load_record_file(path)
    except (OSError, ValueError):
        return
    if loaded is None or _is_sealed(*loaded):
        return
    try:
        os.replace(path, path.with_name(path.name + _UNSEALED_SUFFIX))
        logger.warning(
            "autonudge trust record at %s carries no valid seal (not written by this gateway "
            "under its current key) and was moved aside; loops armed under it must be armed again",
            path,
        )
    except OSError:
        logger.warning("could not quarantine the unsealed autonudge trust record", exc_info=True)


def _read_record() -> dict[str, dict[str, Any]]:
    """Return the record's entries, or ``{}`` for absent/unreadable/malformed/unsealed."""
    path = self_arm_record_path()
    if not path.exists():
        # A reader on an upgraded install before any writer ran. Not under the
        # record lock: a reader may run inside a writer's locked section
        # (``restore_arm_party_if_token`` reads before it writes) and the lock
        # is not re-entrant; the unlink is idempotent, so a racing writer's own
        # retirement simply wins and this one logs and falls through to a read.
        _retire_legacy_record()
    try:
        loaded = _load_record_file(path)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("autonudge self-arm record unreadable; treating as empty", exc_info=True)
        return {}
    if loaded is None:
        return {}
    raw, entries = loaded
    if not _is_sealed(raw, entries):
        logger.warning("autonudge trust record carries no valid seal; treating as empty")
        return {}
    return {
        str(loop_id): entry
        for loop_id, entry in entries.items()
        if isinstance(entry, dict) and isinstance(entry.get("slot_key"), str)
    }


def _read_record_strict_raw() -> dict[str, Any]:
    """The record's ``loops`` map VERBATIM, for the writers.

    Unlike :func:`_read_record`, which is the readers' total view, this one
    keeps every entry as stored -- a malformed sibling included -- and RAISES
    ``OSError`` when the file exists but cannot be read or is not the expected
    shape. A writer that read a corrupt file as ``{}`` would then write its one
    entry over every sibling's, turning one bad byte into every other member's
    loop being refused at its next wake; refusing the write keeps the file as
    evidence and the caller fails closed. A missing file is an empty map: nothing
    to lose.

    A file that parses but carries no valid seal is NOT indeterminate: it was
    provably not written by this gateway under its current key, so its entries
    are nobody's and it reads as the empty map -- the certain "nothing
    recorded" answer, which lets a writer proceed (the lock quarantines the file
    first) rather than leaving the switch refused until an operator intervenes.
    """
    path = self_arm_record_path()
    if not path.exists():
        _retire_legacy_record()  # idempotent; a no-op once the legacy file is gone
    try:
        loaded = _load_record_file(path)
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise OSError(f"autonudge trust record unreadable: {path}") from exc
    if loaded is None:
        raise OSError(f"autonudge trust record malformed: {path}")
    raw, entries = loaded
    if not _is_sealed(raw, entries):
        logger.warning("autonudge trust record carries no valid seal; treating as empty")
        return {}
    return {str(loop_id): entry for loop_id, entry in entries.items()}


def _write_record(entries: dict[str, Any]) -> None:
    path = self_arm_record_path()
    _ensure_record_dir(path.parent)
    atomic_write(
        path,
        json.dumps(
            {"version": 2, "loops": entries, "seal": _seal(entries)},
            ensure_ascii=False,
            sort_keys=True,
        ),
        fsync=True,
    )


def record_self_arm(loop_id: str, slot_key: str) -> None:
    """Record that *loop_id* on *slot_key* was armed by that session's own turn.

    A pure UPSERT of one entry: every other entry is preserved verbatim. The
    write deliberately does NOT prune against a "live loop ids" set supplied
    by the caller -- the authorizer takes that snapshot outside this lock, and
    two crew/member sessions arming in the same second (the crew-boot case the
    exception exists for) would race it: the arm whose snapshot predates the
    other's ``svc.add`` but acquires the lock LAST would prune the sibling's
    freshly written entry, and that sibling's loop -- reported as armed --
    would be refused at every fire. Removing entries is revocation's job
    (:func:`forget_self_arm`), reached from every path a loop leaves the store
    (``AutoNudgeService.remove_sync``), so the record cannot grow past the
    loops that were ever armed and not yet removed. Raises ``OSError`` on a
    failed write: the caller (the authorizer) treats that as fail-closed -- a
    self-armed loop that cannot be recorded would never be allowed to fire, so
    it must not be reported as armed.
    """
    _record_arm(loop_id, slot_key, ARMED_BY_SELF)


def record_owner_arm(loop_id: str, slot_key: str, *, txn: str = "") -> None:
    """Record that *loop_id* on *slot_key* was armed by the dashboard OWNER.

    The owner's Perpetual mode switch on the Crew Members page arms a member's
    own thread from OUTSIDE that thread's turn, which the self-arm exception
    does not cover. It is admitted on the owner-gated member route only, and
    recorded here under its own ``armed_by`` so the fire-time guard can vouch
    for it without the loop ever claiming to be self-armed: a self-arm entry
    never satisfies :func:`is_recorded_owner_arm` and an owner entry never
    satisfies :func:`is_recorded_self_arm`. Same lock, same upsert-only write,
    same revocation path (``forget_self_arm`` on removal) and the same
    fail-closed ``OSError`` contract as :func:`record_self_arm`.

    PRECEDENCE: one entry per loop, one party per entry. An owner TAKEOVER of
    a loop the member armed itself (the owner turning Perpetual mode on over
    a stopped self-armed loop) rewrites that loop's entry to ``owner``; the
    loop is then governed by the owner's rules (caps refused to the member,
    its stop retained) and the store's ``self_armed`` bit, which nothing
    rewrites, is inert because :func:`is_recorded_self_arm` does not vouch
    for an owner entry. The caller that takes over reads the party first
    (:func:`read_arm_party_strict`) so a failed takeover can put it back,
    and stamps its own ``txn`` token on the entry so that restore touches
    only the entry IT wrote: "still says owner" is not an identity once a
    second takeover can write owner too. An arm-time owner entry (no
    takeover) carries no token.
    """
    _record_arm(loop_id, slot_key, ARMED_BY_OWNER, txn=txn)


def _record_arm(loop_id: str, slot_key: str, armed_by: str, *, txn: str = "") -> None:
    with _record_lock():
        # Strict: a corrupt file refuses the write (OSError, fail closed for
        # the arm) rather than being replaced by a map holding only this entry.
        entries = _read_record_strict_raw()
        entry: dict[str, Any] = {
            "slot_key": str(slot_key),
            "armed_ts": time.time(),
            "armed_by": armed_by,
        }
        if txn:
            entry["txn"] = str(txn)
        entries[str(loop_id)] = entry
        _write_record(entries)


def restore_arm_party_if_token(
    loop_id: str, slot_key: str, expected_txn: str, prior_party: str
) -> bool:
    """Undo one takeover's owner entry, atomically, and only if it is still that takeover's.

    One locked read-modify-write: under :func:`_record_lock` the record is read
    strictly and verbatim, the entry for *loop_id* must name *slot_key*, say
    ``owner`` and carry exactly *expected_txn*; then it is rewritten to a self
    entry (``prior_party == "self"``) or removed (``prior_party == ""``), and
    every sibling is written back byte-for-byte. Returns ``True`` when it
    restored, ``False`` when the entry is not this takeover's any more (a later
    takeover's token, another party, no entry) -- in which case nothing is
    written. Because the compare and the write share the lock, an entry cannot
    change between them; the caller never needs a second read. A
    ``prior_party`` of ``owner`` or a blank token means nothing was changed and
    nothing is restored (``False``). Raises ``OSError`` when the file or the
    entry is malformed (fail closed, file left intact) or the write fails.
    """
    if not expected_txn or prior_party == ARMED_BY_OWNER:
        return False
    if prior_party not in (ARMED_BY_SELF, ""):
        raise ValueError(f"unknown prior party {prior_party!r}")
    with _record_lock():
        entries = _read_record_strict_raw()
        entry = entries.get(str(loop_id))
        if entry is None:
            return False
        if not isinstance(entry, dict) or not isinstance(entry.get("slot_key"), str):
            raise OSError(f"autonudge trust record entry for {loop_id} is malformed")
        if entry["slot_key"] != str(slot_key) or entry.get("armed_by") != ARMED_BY_OWNER:
            return False
        if entry.get("txn") != expected_txn:
            return False
        if prior_party == ARMED_BY_SELF:
            entries[str(loop_id)] = {
                "slot_key": str(slot_key),
                "armed_ts": time.time(),
                "armed_by": ARMED_BY_SELF,
            }
        else:
            del entries[str(loop_id)]
        _write_record(entries)
        return True


def forget_self_arm(loop_id: str) -> None:
    """Drop *loop_id* from the record. Best-effort; never raises."""
    try:
        revoke_arm(loop_id)
    except OSError:
        logger.warning("could not revoke self-arm record for %s", loop_id, exc_info=True)


def revoke_arm(loop_id: str) -> None:
    """Drop *loop_id* from the record, whichever party wrote it. STRICT: an
    unreadable or unwritable record raises ``OSError`` so the caller can report
    that the authorization is still standing. Siblings are kept verbatim.

    The entry is the whole of an owner arm's fire-time authorization (there is
    no store bit beside it), so a loop the owner turned OFF must lose it: the
    loop store is agent-writable, and a retained entry would let a forged
    ``active: true`` on the paused record resume wakes the owner stopped. The
    owner's next ON records a fresh entry through the takeover path.
    """
    with _record_lock():
        entries = _read_record_strict_raw()
        if str(loop_id) in entries:
            del entries[str(loop_id)]
            _write_record(entries)


def is_recorded_self_arm(loop_id: str, slot_key: str) -> bool:
    """Whether the trust record vouches that *loop_id* self-armed on *slot_key*.

    Total: any failure to read is ``False`` (refuse). Both the id AND the slot
    must match, so a forged loop that reuses a recorded id on a different slot
    does not inherit the authorization.
    """
    return _armed_by_of(loop_id, slot_key) == ARMED_BY_SELF


def is_recorded_owner_arm(loop_id: str, slot_key: str) -> bool:
    """Whether the trust record vouches that the OWNER armed *loop_id* on *slot_key*.

    Total like :func:`is_recorded_self_arm`, and disjoint from it: an entry
    vouches for exactly one arming party.
    """
    return _armed_by_of(loop_id, slot_key) == ARMED_BY_OWNER


def read_arm_party_strict(loop_id: str, slot_key: str) -> str:
    """The recorded arming party for *loop_id* on *slot_key* -- ``"self"``,
    ``"owner"`` or ``""`` for none -- RAISING ``OSError`` when the record
    exists and cannot be read, instead of answering ``""``.

    For the callers whose refusing answer is not the safe one: an applier
    deciding whether a member may cap or remove its own loop must fail CLOSED
    on an unreadable record (refuse the cap, keep the record), and a total
    read would hand it the permissive ``""`` there. So every INDETERMINATE
    state raises: an unreadable or mis-shaped file, and an entry for this loop
    that is present but malformed (not a dict, no string ``slot_key``, an
    ``armed_by`` that names no known party, a non-string takeover token). Only
    two states answer ``""``, and both are certain: no file at all, or no entry
    for this loop -- nothing was ever recorded. An entry naming ANOTHER slot is
    a certain answer too: whoever armed that loop did not arm it on this slot.
    """
    entries = _read_record_strict_raw()
    if str(loop_id) not in entries:
        return ""
    entry = entries[str(loop_id)]
    if not isinstance(entry, dict) or not isinstance(entry.get("slot_key"), str):
        raise OSError(f"autonudge trust record entry for {loop_id} is malformed")
    if entry["slot_key"] != str(slot_key):
        return ""
    armed_by = entry.get("armed_by", ARMED_BY_SELF)
    if armed_by not in (ARMED_BY_SELF, ARMED_BY_OWNER):
        raise OSError(f"autonudge trust record entry for {loop_id} names an unknown party")
    if not isinstance(entry.get("txn", ""), str):
        raise OSError(f"autonudge trust record entry for {loop_id} carries a malformed token")
    return str(armed_by)


def _armed_by_of(loop_id: str, slot_key: str) -> str:
    """The recorded arming party for *loop_id* on *slot_key*, or ``""``.

    An entry with no ``armed_by`` predates the field and was written by the
    only writer that existed then -- the self-arm path -- so it reads as
    ``"self"``. Any other spelling reads as nobody, which refuses.
    """
    entry = _read_record().get(str(loop_id))
    return _party_of_entry(entry, slot_key)


def _party_of_entry(entry: dict[str, Any] | None, slot_key: str) -> str:
    if entry is None or entry.get("slot_key") != str(slot_key):
        return ""
    armed_by = entry.get("armed_by", ARMED_BY_SELF)
    if armed_by in (ARMED_BY_SELF, ARMED_BY_OWNER):
        return str(armed_by)
    return ""


def recorded_arm_parties() -> dict[tuple[str, str], str]:
    """All sealed arm parties, read and verified once.

    The members roster projects every crewmate in one response. Reading the
    same sealed file once per row would block the event loop and repeat its
    JSON parse and HMAC check. This total, fail-closed snapshot lets that
    caller offload one read, then use only in-memory lookups per row.
    """
    parties: dict[tuple[str, str], str] = {}
    for loop_id, entry in _read_record().items():
        if not isinstance(entry, dict):
            continue
        slot_key = entry.get("slot_key")
        if not isinstance(slot_key, str):
            continue
        party = _party_of_entry(entry, slot_key)
        if party:
            parties[(str(loop_id), slot_key)] = party
    return parties


async def await_thread_to_completion(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """``asyncio.to_thread`` that a cancellation cannot leave running unjoined.

    For the trust-record writes (``record_owner_arm``, the takeover restore):
    the thread is started as its own future and awaited through a shield; if
    the awaiting task is cancelled, the thread is awaited AGAIN -- shielded,
    in a loop, so a SECOND cancellation does not abandon it either -- until
    the thread future is done, and only then does the cancellation propagate.
    By the time the caller unwinds (releasing a lock, letting a shutdown
    drain's join report done) the write has finished. The wait is bounded by
    the write itself (one locked file rewrite) and, at shutdown, by the
    drain's own join timeout, which gives up on the TASK and logs so; the
    thread still runs to its end on its own. The thread's own exception, when
    the caller was cancelled, is dropped here: the caller is unwinding on the
    cancel, and the write's outcome is re-read by whoever acts next.
    """
    fut = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        while not fut.done():
            try:
                await asyncio.shield(fut)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - the write's own error is not this cancel's
                break
        raise


#: PER-SLOT LOCK for the owner's Perpetual mode on one member slot. Every
#: transition that pairs the arm record with the loop's state takes it: the
#: switch's mutations in ``handlers.members`` (identity re-check, loop re-read,
#: takeover, OFF's pause + revoke), the member's own ``autonudge_stop`` /
#: ``monitor_stop`` and its ``monitor_update`` cap check in
#: ``session_directive_apply``, AND the fire path's admission on a member slot
#: (``gateway._dashboard_mode_admits`` through turn publication). Without the
#: last one the timer could read the owner entry, lose the CPU to an OFF that
#: pauses the loop and revokes the entry, then publish a wake on a loop whose
#: authorization is gone; without the cap one the member could read "self-armed",
#: lose the CPU to a takeover that resumes the loop uncapped, then write a
#: finite cap onto the owner's loop. ``monitor_update`` of non-cap fields and
#: the service's own bookkeeping go through the nudge service's locks, not this
#: one. Process-local like the service. Entries are dropped once no holder or
#: waiter remains (``_PERPETUAL_LOCK_USERS``), so the map is bounded by the
#: operations in flight, not by the members ever touched. Lives HERE, next to
#: the record, because the gateway, the handler and the directive consumer all
#: import this leaf and none of them may import each other for it.
_PERPETUAL_LOCKS: dict[str, asyncio.Lock] = {}
_PERPETUAL_LOCK_USERS: dict[str, int] = {}


class perpetual_slot_lock:
    """``async with perpetual_slot_lock(slot_key):`` -- the per-slot lock, refcounted."""

    def __init__(self, slot_key: str) -> None:
        self._key = slot_key
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _PERPETUAL_LOCKS.get(self._key)
        if lock is None:
            lock = _PERPETUAL_LOCKS[self._key] = asyncio.Lock()
        _PERPETUAL_LOCK_USERS[self._key] = _PERPETUAL_LOCK_USERS.get(self._key, 0) + 1
        self._lock = lock
        try:
            await lock.acquire()
        except BaseException:
            self._release_use()
            raise

    async def __aexit__(self, *_exc: object) -> None:
        assert self._lock is not None
        self._lock.release()
        self._release_use()

    def _release_use(self) -> None:
        left = _PERPETUAL_LOCK_USERS.get(self._key, 1) - 1
        if left <= 0:
            _PERPETUAL_LOCK_USERS.pop(self._key, None)
            _PERPETUAL_LOCKS.pop(self._key, None)
        else:
            _PERPETUAL_LOCK_USERS[self._key] = left
