"""Queue storage and delivery-ledger operations for dashboard chat slots."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

# The MODULE, not its names: the proof producers stay behind a path the argv
# floor's credential-mint rule reads (``queue_origin_token``); re-exporting them
# here would put a mint under a path that rule does not cover.
from kiro_crew.dashboard import queue_origin_token

logger = logging.getLogger(__name__)

# This is well above the slot queue's legitimate in-flight set.  Eviction only
# bounds orphaned bookkeeping; an evicted agent remains recoverable on restart.
MAX_PENDING_SUBAGENT_DELIVERIES = 128

#: How many queued user prompts one session's metadata line carries, and how
#: many a restore admits back. Front-first, because the front is what runs
#: first: an over-cap queue keeps the entries closest to delivery.
MAX_DURABLE_QUEUE_ENTRIES = 32

#: Byte budget for the whole serialized ``queued_prompts`` value. The metadata
#: line is ONE json line every reader of the session parses, so an unbounded
#: set of long prompts would make it expensive for every reader rather than
#: only for the queue. Entries are admitted front-first until the budget is
#: spent; a prompt is never truncated, because a shortened prompt replayed as
#: the user's own words is worse than one reported as not carried.
MAX_DURABLE_QUEUE_BYTES = 256_000

#: How many raw entries a restore INSPECTS, as distinct from how many it keeps.
#: A retention cap bounds this gateway's own writes; it does not bound a line
#: that was edited or corrupted outside the gateway into carrying a million
#: entries. The apply phase that calls :func:`sanitize_restored_queue` is
#: loop-affine (``_apply_recent_session``), so a scan proportional to the file's
#: content — even a cheap per-item scan — is startup work the event loop cannot
#: shed, and it blocks chat and heartbeat while it runs. Comfortably above the
#: retention cap so an ordinary line with a few dropped entries still restores
#: everything it should; entries past it are reported as not restored, never
#: silently ignored.
MAX_DURABLE_QUEUE_SCAN = 4 * MAX_DURABLE_QUEUE_ENTRIES

#: Queue-entry keys the durable copy carries. Everything else on an entry is
#: process-local plumbing (retry callbacks, synthetic payloads) or is
#: deliberately excluded — see :func:`durable_queue_entries`.
#:
#: ``_directive_user_origin`` and ``_directive_channel_origin`` are NOT here, and
#: their absence is a security property rather than an omission. They record that
#: an entry's words arrived from an authenticated human, and the drain turns that
#: into real authority: ``chat_runner`` reduces the consumed entries' flags into
#: ``producer_is_user_facing``, which admits user-surface and self-arming
#: directives and exempts them from the LINKED containment constraint. The
#: transcript this value is written to is an ordinary readable-writable file in
#: the crew home, not a write-protected one, so anything restored from it is
#: attacker-supplied for the purposes of that decision. Persisting provenance
#: that a later reader cannot distinguish from a hand-edited line would hand
#: human authority to whoever can write the file, so provenance is not carried
#: across a restart at all and restored entries fail closed to non-directive.
#: Marks an entry this process RESTORED rather than accepted. Process-local by
#: construction -- the writer emits only :data:`_DURABLE_QUEUE_KEYS`, so a
#: hand-edited line cannot clear it and cannot forge it either.
#:
#: It exists because dropping provenance is only half a fail-closed rule. A
#: restored entry carries no actor, and "no actor" resolves to ``user`` in the
#: drain -- which is exactly the arm ``model.route`` admits. So the absence has to
#: be readable as "unknown" rather than as "the person's", and this key is what a
#: consumer asks instead of trying to tell the two apart from the actor alone.
RESTORED_QUEUE_KEY = "_restored_from_disk"

#: Field of the DURABLE queue record (the persisted ``queued_prompts`` line, not
#: the live entry) carrying the gateway's own attestation of the entry's
#: PROVENANCE: where it came from -- the dashboard (the composer, an app, a
#: recovery of a dashboard turn) or a named channel conversation that handed it
#: off -- and the containment that held when it was accepted. It is the one
#: provenance a restored entry may carry across a restart, because it is the one
#: provenance a restart cannot forge: an HMAC over the slot key, the queue id,
#: the content, the entry's channel address (``channel_busy``'s stamp; none for
#: dashboard text) and its admission-time containment snapshot, keyed from the
#: gateway's fenced signing secret (``token_signing.key``, which no agent file
#: tool can read or write), under its own domain tag so a queue proof is never
#: also a valid auth token (:mod:`kiro_crew.dashboard.queue_origin_token`).
#:
#: The rule it serves is DEFAULT DENY, in three parts. A restored entry carries no
#: command authority: the drain treats it as channel text -- no composer command
#: word, a leading ``/workflow`` refused with the usual notice -- unless an
#: ADDRESS-LESS proof verifies against it (:func:`dashboard_origin_proven`). It
#: carries no admission snapshot -- it is re-checked against every constraint
#: that holds NOW, and on a linked or mirrored slot that drops it -- unless the
#: proof verifies over the snapshot the record carries. And it names no channel
#: conversation -- a released binding neither drops nor reports it -- unless the
#: proof verifies over the address the record carries. All three are read back
#: by :func:`restore_queue_provenance` from one verification, so a persisted
#: entry can be edited in any way an editor likes (its stamps removed or
#: rewritten, its content rewritten, a proof transplanted from another entry or
#: slot) and comes back as unproven prose that fails closed; a dashboard entry
#: keeps its command word and its admission through a restart because its proof
#: still verifies; and a channel hand-off comes back exactly as it was queued --
#: channel text, bound to the conversation that sent it, admitted under the
#: containment it was accepted with -- so an entry whose binding still holds
#: drains and one whose binding changed is dropped WITH the notice to its
#: sender. Compare :data:`RESTORED_QUEUE_KEY`: that key says the entry was
#: restored, this one says what the gateway knew about it when it was accepted.
#:
#: Live, the proof lives BESIDE the queue -- ``owner._origin_proofs``, keyed by
#: queue id, like ``_last_enqueue_ts`` -- and never on the entry: queue dicts are
#: compared wholesale on the wire and across the suite (the facade hands back the
#: caller's own ``meta`` object), so the board and every ``to_dict`` projection
#: see exactly what was enqueued. The durable writer joins the proof onto the
#: record it emits (the address and the snapshot already ride the record inside
#: ``meta``) and the restore verifies it and lifts it back into the sidecar.
ORIGIN_PROOF_KEY = "origin_proof"


def _origin_proofs_of(owner: Any) -> dict[str, str]:
    """The owner's proof sidecar, created on first use for a bare test double."""
    proofs = getattr(owner, "_origin_proofs", None)
    if not isinstance(proofs, dict):
        proofs = {}
        owner._origin_proofs = proofs
    return proofs


def _provenance_parts(meta: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The two signed parts of an entry's ``meta``: its channel address and its
    admission-time containment snapshot, each ``None`` unless a mapping.

    Read off untrusted plumbing (a live entry's meta, or a durable record's), so
    any other shape is "absent" -- and absent is what the proof is then computed
    over, which is why a stamp hand-written as anything but a mapping verifies
    as nothing rather than as a stamp.
    """
    if not isinstance(meta, dict):
        return None, None
    # Local import: session_control reaches this module through state, and
    # channel_busy through chat_delivery, so taking either key at module level
    # would close an import cycle.
    from kiro_crew.dashboard.channel_busy import CHANNEL_ORIGIN_META_KEY
    from kiro_crew.dashboard.session_control import QUEUED_CONTAINMENT_META_KEY

    address = meta.get(CHANNEL_ORIGIN_META_KEY)
    admission = meta.get(QUEUED_CONTAINMENT_META_KEY)
    return (
        address if isinstance(address, dict) else None,
        admission if isinstance(admission, dict) else None,
    )


def dashboard_origin_proven(owner: Any, entry: Any) -> bool:
    """Whether *entry* of *owner*'s queue carries the gateway's proof of DASHBOARD
    origin: the proof over the entry as it stands (owner key, id, content, its
    admission snapshot) and NO channel address.

    Read from the sidecar by the entry's id. False for a missing proof, a
    rewritten content or snapshot, a proof moved between entries or slots, a
    non-string tag -- and for a channel hand-off, whose proof is over its
    address and so never verifies as address-less. Never raises.
    """
    if not isinstance(entry, dict):
        return False
    entry_id = entry.get("id")
    if not isinstance(entry_id, str):
        return False
    proofs = getattr(owner, "_origin_proofs", None)
    tag = proofs.get(entry_id) if isinstance(proofs, dict) else None
    address, admission = _provenance_parts(entry.get("meta"))
    if address is not None:
        # A channel hand-off is not dashboard text, whatever its proof says.
        return False
    return queue_origin_token.queue_provenance_matches(
        str(getattr(owner, "key", "") or ""), entry_id, entry.get("content"), None, admission, tag
    )


def _stamp_origin(owner: Any, item: dict[str, Any], *, directive_channel_origin: bool) -> None:
    """Record the provenance proof for *item* -- or withhold it.

    Called at every point the repository accepts or rewrites an entry, because
    the proof is over the content and the stamps beside it: an edit that changes
    the words must re-sign them. Writes the owner's sidecar only -- never
    ``item`` -- and prunes the sidecar down to the ids the queue holds, so a
    consumed or removed entry does not leave its proof behind.

    Only an entry a restart can hand back gets a proof -- a cron notice, a
    recovery payload or a callback-bearing entry dies with its process. Channel
    text whose entry names no conversation (``directive_channel_origin`` with no
    address stamped) gets none either: the proof of a channel hand-off IS the
    proof of its address, and channel text the gateway cannot place comes back
    as unproven prose that fails closed, the pre-proof reading. A dashboard edit
    of a stamped hand-off keeps the address on the entry, so its proof stays a
    channel proof: restored, the edited words drain with channel authority, the
    narrower of the two -- the one restart cost this design accepts.
    """
    proofs = _origin_proofs_of(owner)
    queue = getattr(owner, "_queue", None) or ()
    live = {entry.get("id") for entry in queue if isinstance(entry, dict)} | {item.get("id")}
    for stale in [entry_id for entry_id in proofs if entry_id not in live]:
        del proofs[stale]
    entry_id = item.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        return
    if not _is_durable_queue_entry(item):
        proofs.pop(entry_id, None)
        return
    address, admission = _provenance_parts(item.get("meta"))
    if directive_channel_origin and address is None:
        proofs.pop(entry_id, None)
        return
    try:
        proofs[entry_id] = queue_origin_token.queue_provenance_proof(
            str(getattr(owner, "key", "") or ""),
            entry_id,
            str(item.get("content") or ""),
            channel_address=address,
            admission=admission,
        )
    except (TypeError, ValueError):
        # A stamp json cannot emit would also fail the durable write of this
        # entry's meta; with no proof the entry restores as unproven prose.
        proofs.pop(entry_id, None)


_DURABLE_QUEUE_KEYS: tuple[str, ...] = (
    "id",
    "content",
    "meta",
)


def _is_durable_queue_entry(item: Any) -> bool:
    """True when *item* is a plain user prompt a restart may hand back.

    The queue holds two populations and only one of them survives its process.
    A plain user prompt is the user's own words, waiting for the running turn
    to end: nothing outside the queue holds it, so losing it loses speech.
    Everything else in the queue is a SYSTEM entry whose meaning is bound to
    live state this process is about to lose:

    - an entry carrying ``_on_consumed`` / ``_on_irreversibly_consumed``
      acknowledges the exact automatic payload that failed, and the callback
      does not survive the restart — replaying the text without it
      acknowledges nothing;
    - an entry carrying a ``payload`` is a synthetic recovery continuation
      (``is_synthetic_payload_item``), which dispatches an action a dead turn
      announced;
    - an entry carrying a ``kind`` is an injection whose producer is gone: a
      cron notification, a subagent completion, a plan approval. Its content
      names an event, and a restart is not that event happening again.
    """
    if not isinstance(item, dict):
        return False
    content = item.get("content")
    if not isinstance(content, str) or not content:
        return False
    if item.get("_on_consumed") is not None or item.get("_on_irreversibly_consumed") is not None:
        return False
    if item.get("payload"):
        return False
    if item.get("kind"):
        return False
    return True


def durable_queue_entries(
    queue: list[dict[str, Any]], proofs: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """The json-safe copies of *queue* a metadata writer may persist.

    Copies rather than aliases, so a later in-memory mutation cannot rewrite a
    dict a writer is holding. ``meta`` rides along VERBATIM (through one json
    round-trip that also proves the writer can serialize it): it carries the
    admission-time containment snapshot the drain re-validates against, and an
    entry without one fails closed into the full current-constraint set — so
    dropping it would make a restored prompt refusable for a boundary its
    author was never subject to.

    *proofs* is the owner's origin-proof sidecar (``owner._origin_proofs``): an
    entry it names gets the proof joined onto its RECORD as
    :data:`ORIGIN_PROOF_KEY`, which is how the gateway's attestation reaches the
    line without ever sitting on the live entry. The reader
    (:func:`sanitize_restored_queue`) lifts it back into the sidecar.

    An entry whose ``meta`` cannot be serialized keeps its content and loses
    only the metadata, because the prompt is the part that cannot be
    reconstructed.

    Pure: an over-cap queue is reported by the SAVE that writes the value, not
    from here (see :func:`count_durable_candidates`). This runs on every flush
    tick through ``queue_persist_pending`` and inside a retried snapshot, so a
    warning here would repeat for as long as the queue stayed over the cap.
    """
    out: list[dict[str, Any]] = []
    budget = MAX_DURABLE_QUEUE_BYTES
    for item in queue:
        if not _is_durable_queue_entry(item):
            continue
        if len(out) >= MAX_DURABLE_QUEUE_ENTRIES:
            continue
        entry: dict[str, Any] = {}
        for key in _DURABLE_QUEUE_KEYS:
            if key not in item:
                continue
            value = item[key]
            if key == "meta":
                try:
                    value = json.loads(json.dumps(value))
                except (TypeError, ValueError):
                    continue
                if not isinstance(value, dict):
                    continue
            entry[key] = value
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            continue
        proof = proofs.get(entry_id) if proofs else None
        if isinstance(proof, str) and proof:
            entry[ORIGIN_PROOF_KEY] = proof
        cost = len(json.dumps(entry))
        if cost > budget:
            continue
        budget -= cost
        out.append(entry)
    return out


def count_durable_candidates(queue: list[dict[str, Any]]) -> int:
    """How many entries in *queue* are user prompts a restart could hand back.

    Paired with ``len(durable_queue_entries(queue))``, the difference is exactly
    how many accepted prompts the bounds refuse to persist. The send that
    accepted them is NOT rejected for it — refusing a queued send would take the
    user's words away at the one moment they cannot be re-typed from the
    transcript — so the count is what makes the shortfall visible instead of
    silent.

    Both halves must come from ONE read of the queue: see
    :func:`durable_queue_view`.
    """
    return sum(1 for item in queue if _is_durable_queue_entry(item))


def warn_if_not_durable(
    queue: list[dict[str, Any]],
    entry_id: str,
    slot_key: str,
    proofs: dict[str, str] | None = None,
) -> bool:
    """Report an accepted prompt the durable write will not keep. True when kept.

    The bounds refuse to persist an entry past the count cap or the byte budget,
    and the send is still ACCEPTED: refusing a queued send takes the user's words
    away at the one moment they cannot be re-typed from the transcript. That
    leaves the accept as the only place the difference can be told, and it has to
    be told somewhere — an accepted prompt that a restart drops is the exact
    silence this whole change exists to end, so it must not move from the prompt
    to its acknowledgment and stop there.

    So it is told at WARNING in the gateway log, naming the slot, the entry, how
    many candidates the queue holds and how many the write keeps. Both the
    verdict and the reason are read off the writer's OWN output rather than
    re-derived: an entry absent from a full set was refused by the count cap, and
    absent from a short one by the byte budget. The two ceilings interact — a
    large prompt can be refused at position 3 — so anything that re-implemented
    them here would answer differently from the writer for exactly the entries
    this exists to report.

    Not a receipt field. A caller-visible ``durable`` boolean on the enqueue
    acknowledgments has no reader, so it is not shipped; the on-screen queue-card
    marker belongs with its consumer.
    """
    if not entry_id:
        return False
    snapshot = list(queue)
    kept = durable_queue_entries(snapshot, proofs)
    if any(entry.get("id") == entry_id for entry in kept):
        return True
    candidates = count_durable_candidates(snapshot)
    reason = (
        "queue holds the {} durable entries it may".format(MAX_DURABLE_QUEUE_ENTRIES)
        if len(kept) >= MAX_DURABLE_QUEUE_ENTRIES
        else "entry does not fit the {}-byte durable budget".format(MAX_DURABLE_QUEUE_BYTES)
    )
    logger.warning(
        "Slot %s queued prompt %s is accepted but NOT persisted (%s); "
        "%d candidate(s) queued, %d carried, so a restart before it runs drops it",
        slot_key,
        entry_id,
        reason,
        candidates,
        len(kept),
    )
    return False


def durable_queue_view(
    queue: list[dict[str, Any]], proofs: dict[str, str] | None = None
) -> tuple[list[dict[str, Any]], int]:
    """The durable entries of *queue* and its candidate count, from ONE read.

    The shortfall the save reports is ``count - len(entries)``, and that
    subtraction is only true of a single observation. Taking the two halves from
    separate reads of a LIVE queue makes an ordinary prompt that merely arrived
    between them look like one the bounds refused: the operator is then told a
    prompt exceeded the durable bounds when it is simply owed to the next save.
    A list copy is what makes the pair one observation. *proofs* is the owner's
    origin-proof sidecar, joined onto the records exactly as
    :func:`durable_queue_entries` does, so the save's snapshot compares equal to
    the writer's value.
    """
    snapshot = list(queue)
    return durable_queue_entries(snapshot, proofs), count_durable_candidates(snapshot)


def queue_persist_signature(entries: list[dict[str, Any]]) -> str:
    """A stable identity for one durable queue value.

    The periodic flush compares this against what it last wrote, which is what
    makes durability independent of the call site: a queue mutated in place —
    a reorder, a plan-approval filter, a force-stop clear — drifts from the
    persisted signature and is picked up on the next pass, without every such
    site having to remember to mark the slot dirty.
    """
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8", "replace")
    ).hexdigest()


#: The signature of an empty queue, and the value a fresh slot starts at. A
#: slot with nothing queued owes no write: starting it here rather than at ``""``
#: is what keeps the drift check from making every newborn slot save itself once
#: — a save rewrites the transcript, and an unnecessary one invalidates every
#: cache keyed on the file's mtime (the session-intent summary among them). A
#: queue that WAS persisted and has since emptied still drifts, because its
#: stored signature is that of the non-empty value.
EMPTY_QUEUE_SIGNATURE = queue_persist_signature([])


def sanitize_restored_queue(raw: object) -> list[dict[str, Any]]:
    """Validate a persisted ``queued_prompts`` value back into queue entries.

    On-disk metadata is a trust boundary (the file can be edited or corrupted
    outside the gateway), so every field is re-checked instead of trusted, and
    anything the durable writer never emits is dropped rather than carried: a
    restored entry must not arrive wearing a ``kind``, a ``payload``, or a
    callback key, because each of those changes what the drain DOES with it.

    Provenance is dropped for the same reason and matters most:
    ``_directive_user_origin``, ``_directive_channel_origin`` and the entry's
    ``meta`` turn actor are never restored, so a restored entry is non-directive
    and actor-less by construction. The drain
    reduces the consumed entries' flags into the authenticated-human authority a
    directive is admitted under, and this line is an ordinary writable file — a
    carried flag would be authority granted to whoever edited it. The writer does
    not emit them either (:data:`_DURABLE_QUEUE_KEYS`); dropping them here is the
    reader's half of the same rule, so a hand-added flag buys nothing.

    This reader is PURE and is the fail-closed half of a two-step restore: it
    strips every provenance key, and :func:`restore_queue_provenance` -- given
    the owner, whose key the proof is bound to -- puts back exactly the two the
    gateway's own proof (:data:`ORIGIN_PROOF_KEY`, the record's own field)
    verifies over: the admission snapshot and the channel address. A caller that
    stops after this step restores every entry as unproven prose with no
    admission and no address, the same fail-closed reading.

    ``meta``'s admission-time containment snapshot is stripped here because the
    asymmetry is sharp. The drain's re-check
    (``newly_held_constraints``) treats a constraint the entry recorded as
    already-held as "not a change", so an ABSENT snapshot fails closed against
    every currently-held constraint while a FORGED all-True one reports nothing
    newly held and fails OPEN — a hand-written entry would drain into a linked or
    mirrored slot and republish to an audience its admission never contemplated.
    Stripping the key restores the fail-closed baseline: the entry is re-checked
    against the constraints that hold NOW, the only set this process can vouch
    for -- unless the second step proves the snapshot is the one this gateway
    recorded. The rest of ``meta`` rides along, because it carries the sender's
    own plumbing (``sendId``, attachments) that decides nothing about audience.

    Capped at :data:`MAX_DURABLE_QUEUE_ENTRIES` entries and
    :data:`MAX_DURABLE_QUEUE_BYTES` of serialized content, the SAME two ceilings
    the persist path admits, costed against the SAME key projection — because the
    writer's bounds only bound what this gateway wrote, and the line it reads back
    can have been edited or corrupted to carry more. Whatever is admitted here is
    retained in the live slot, re-projected to every websocket client and
    re-serialized by every later save, so an unbounded read would let one
    hand-edited line cost the process memory and work forever, not just once at
    parse time. An entry that does not fit is dropped whole: a truncated prompt
    handed back as the user's own words is worse than one reported as not
    carried.

    A third ceiling, :data:`MAX_DURABLE_QUEUE_SCAN`, bounds how much of the raw
    value is INSPECTED at all: the caller that applies a restored session runs on
    the event loop, so walking every entry of an arbitrarily long list — even to
    reject it — is startup work that blocks chat and the heartbeat.
    """
    if not isinstance(raw, list):
        return []
    # Local import: session_control reaches this module through state, so taking
    # the key at module level would close an import cycle.
    from kiro_crew.dashboard.channel_busy import CHANNEL_ORIGIN_META_KEY
    from kiro_crew.dashboard.chat_delivery import TURN_ACTOR_META_KEY
    from kiro_crew.dashboard.session_control import (
        QUEUED_CONTAINMENT_META_KEY,
        SEND_ORIGIN_META_KEY,
    )

    entries: list[dict[str, Any]] = []
    budget = MAX_DURABLE_QUEUE_BYTES
    skipped = 0
    # Bound the READ, not only the retention: see MAX_DURABLE_QUEUE_SCAN. The
    # unscanned tail is counted as skipped rather than dropped quietly, so the
    # warning below states the real number of prompts not handed back.
    scanned = raw[:MAX_DURABLE_QUEUE_SCAN]
    skipped += len(raw) - len(scanned)
    for index, item in enumerate(scanned):
        if len(entries) >= MAX_DURABLE_QUEUE_ENTRIES:
            # Stop, do not keep walking: the remaining items cannot be admitted,
            # and counting them is all that is left to do.
            skipped += len(scanned) - index
            break
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content:
            continue
        entry_id = item.get("id")
        entry: dict[str, Any] = {
            # A missing or invalid id gets a fresh one so the entry stays
            # addressable: every queue mutation the user can reach (promote,
            # edit, delete) is keyed by id.
            "id": entry_id if isinstance(entry_id, str) and entry_id else uuid.uuid4().hex[:12],
            "content": content,
            # Restored entries are plain user prompts by construction; the
            # empty kind is what keeps them out of the system-injection paths.
            "kind": "",
            # PROCESS-LOCAL, and never round-trips: the writer admits only
            # ``_DURABLE_QUEUE_KEYS`` (id, content, meta), so this key cannot be
            # set by editing the line -- which is the whole point of having it.
            # It records that this entry's provenance was established by a
            # previous process and cannot be vouched for by this one, which is
            # what :data:`RESTORED_QUEUE_KEY` is read for.
            RESTORED_QUEUE_KEY: True,
        }
        meta = item.get("meta")
        if isinstance(meta, dict):
            # The admission-time containment snapshot is stripped for the same
            # reason the provenance flags are, and the asymmetry it would
            # otherwise leave is sharper: the drain's re-check treats a
            # constraint the entry recorded as already-held at admission as "not
            # a change", so an ABSENT snapshot fails closed against every
            # currently-held constraint while a FORGED all-True one reports
            # nothing newly held and fails OPEN. A hand-written entry could then
            # drain into a linked or mirrored slot and republish to an audience
            # its admission never contemplated. Dropping the key restores the
            # fail-closed baseline: a restored entry is re-checked against the
            # constraints that hold NOW, which is the only set this process can
            # vouch for. The cost is narrow — an unlinked, unmirrored slot has no
            # boolean constraint held, so the ordinary restore is unchanged.
            # The TURN ACTOR goes with it, and for the plainer version of the
            # same argument. It names WHO authored the entry -- an app, a cron, a
            # sub-agent -- and the drain turns that into what the turn is allowed
            # to do: `model.route` admits only an actor of `user`, so an entry
            # whose stamp a file editor removed drains as the person's and gets an
            # owner-scoped model decision spent on an app's prompt. Restored, the
            # stamp is worth exactly what the file is worth, so it is dropped and
            # the drain re-derives what it can from the entry's `kind`, which the
            # writer never emits either. A restored app entry therefore carries no
            # actor at all -- the same fail-closed baseline the flags above get.
            #
            # The SENDING SLOT goes with them, and it is the sharpest of the
            # three because the value is not merely read, it names a WRITE
            # TARGET: the drain resolves the recipient of its drop notice from
            # this key alone and appends the entry's own text there
            # (`session_control.notify_send_origin_dropped`). Carried back off
            # the line verbatim, an edited stamp turns a file write into a
            # transcript row in a session the editor does not own, with the
            # entry's content as its body. Nothing in the entry can attest to
            # who sent it, so the key is worth exactly what the file is worth
            # and is dropped. The cost is one notice: a relay that outlives a
            # restart and is then dropped reports to nobody, while the RELAY
            # itself still survives -- which is what putting the stamp in
            # ``meta`` rather than a consumption callback buys, since a
            # callback-carrying entry is not persisted at all.
            #
            # The CHANNEL CONVERSATION stamp (``channel_busy``) goes for the
            # same reason and names a write target of the same kind: the drain
            # resolves the recipient of its channel drop notice from this key
            # alone and sends to whatever allow-listed conversation it names,
            # from a slot that need never have been bound to it. Nothing IN THE
            # ENTRY can attest to the binding, so the key is worth what the file
            # is worth here. Unlike the sender stamp, though, the gateway signed
            # this one -- with the containment snapshot -- when it accepted the
            # entry, so :func:`restore_queue_provenance` puts both back when
            # that proof verifies: a hand-off comes back bound to the
            # conversation that sent it, and an edited or hand-written stamp
            # comes back as nothing.
            entry["meta"] = {
                k: v
                for k, v in meta.items()
                if k
                not in (
                    QUEUED_CONTAINMENT_META_KEY,
                    TURN_ACTOR_META_KEY,
                    SEND_ORIGIN_META_KEY,
                    CHANNEL_ORIGIN_META_KEY,
                )
            }
        # The proof is the record's own field, never the entry's. It is not
        # verified here -- verification needs the owner's key, and this reader
        # is pure -- and it is not carried: :func:`restore_queue_provenance`
        # reads it off the same line again for the entries admitted here.
        raw_proof = item.get(ORIGIN_PROOF_KEY)
        proof = raw_proof if isinstance(raw_proof, str) and raw_proof else None
        try:
            # Costed against the same key projection the WRITER emits
            # (:data:`_DURABLE_QUEUE_KEYS` plus the proof it joins on; no
            # ``kind``), not against the entry handed back. Charging the reader
            # for a key the writer never emitted makes the reader's budget the
            # smaller of the two, and a queue persisted just under the ceiling
            # would then drop its tail on the way back in — losing a prompt that
            # WAS durably written, which is the one outcome this whole value
            # exists to prevent.
            record = {k: v for k, v in entry.items() if k in _DURABLE_QUEUE_KEYS}
            if proof is not None:
                record[ORIGIN_PROOF_KEY] = proof
            cost = len(json.dumps(record))
        except (TypeError, ValueError):
            # ``meta`` came off an untrusted line: a value json cannot re-emit
            # would also break every later save of this slot.
            continue
        if cost > budget:
            skipped += 1
            continue
        budget -= cost
        entries.append(entry)
    if skipped:
        logger.warning(
            "%d persisted queued prompt(s) exceed the durable queue bounds and "
            "are not restored; %d handed back",
            skipped,
            len(entries),
        )
    return entries


def restore_queue_provenance(owner: Any, entries: list[dict[str, Any]], raw: object) -> None:
    """The second half of the restore: hand back the provenance the gateway can prove.

    *entries* is what :func:`sanitize_restored_queue` admitted from *raw*, every
    provenance key stripped. For each, the durable record it came from is read
    again by id and its :data:`ORIGIN_PROOF_KEY` verified against the entry as
    restored plus the record's own channel address and admission snapshot, under
    *owner*'s key. A proof that verifies puts exactly those two mappings back on
    the entry's ``meta`` and the proof into the owner's sidecar, BEFORE the drain
    re-validates anything: the entry then drains under the containment it was
    accepted with, and a channel hand-off is released when its binding is gone
    and its conversation told, exactly as if the process had never restarted.
    An entry with no proof, or one that does not verify -- content, address or
    snapshot rewritten, proof transplanted, another slot -- keeps the stripped
    reading and fails closed against the constraints that hold now, with the
    dashboard notice only. The turn actor and the sender stamp are not signed
    and stay stripped.

    Bounded by the same scan ceiling as the reader. Never raises: a record that
    cannot be found, read or verified simply proves nothing.
    """
    if not entries or not isinstance(raw, list):
        return
    from kiro_crew.dashboard.channel_busy import CHANNEL_ORIGIN_META_KEY
    from kiro_crew.dashboard.session_control import QUEUED_CONTAINMENT_META_KEY

    records: dict[str, dict[str, Any]] = {}
    for item in raw[:MAX_DURABLE_QUEUE_SCAN]:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        # First record per id: a line carrying two records under one id is
        # edited, and the proof is over the content, so at most one can verify.
        if isinstance(item_id, str) and item_id and item_id not in records:
            records[item_id] = item
    proofs = _origin_proofs_of(owner)
    slot_key = str(getattr(owner, "key", "") or "")
    for entry in entries:
        record = records.get(entry["id"])
        if record is None:
            continue
        tag = record.get(ORIGIN_PROOF_KEY)
        address, admission = _provenance_parts(record.get("meta"))
        if not isinstance(tag, str) or not queue_origin_token.queue_provenance_matches(
            slot_key, entry["id"], entry.get("content"), address, admission, tag
        ):
            continue
        proofs[entry["id"]] = tag
        if address is None and admission is None:
            continue
        meta = entry.get("meta")
        if not isinstance(meta, dict):
            meta = entry["meta"] = {}
        if admission is not None:
            meta[QUEUED_CONTAINMENT_META_KEY] = admission
        if address is not None:
            meta[CHANNEL_ORIGIN_META_KEY] = address


def _delivery_key(content: str) -> str:
    """Return a compact identity that survives queue-entry ID replacement."""
    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:32]


#: Meta keys a queued send's attachment lists ride under, each with the marker
#: word its ``[<marker> N] path`` tokens use. ``files`` is the image-free list
#: ``[attached_file N]`` markers index into, ``dirs`` the folder list
#: ``[attached_dir N]`` markers index into. Defined here, on the queue entry's
#: own module, because every dashboard module that reads them sits downstream.
ATTACHMENT_META_KEYS: tuple[str, ...] = ("files", "dirs")
_ATTACHMENT_MARKERS: dict[str, str] = {"files": "attached_file", "dirs": "attached_dir"}


def _marker_spans(content: str, marker: str, index: int, path: str) -> list[tuple[int, int]]:
    """Every span of the exact ``[<marker> <index>] <path>`` token in *content*.

    The path must end at a whitespace or the end of the text: a bare substring
    test would keep ``/tmp/report.pdf`` alive through ``/tmp/report.pdf.bak``,
    and a bare replace would rewrite the ``[attached_file 2] /tmp/b`` prefix of
    ``[attached_file 2] /tmp/bak`` -- one is a removed attachment drawing a card
    again, the other is a caption silently altered.
    """
    token = f"[{marker} {index}] {path}"
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        at = content.find(token, start)
        if at < 0:
            return spans
        end = at + len(token)
        if end == len(content) or content[end].isspace():
            spans.append((at, end))
        start = at + 1


def _renumber_marker(content: str, marker: str, old: int, new: int, path: str) -> str:
    """Rewrite each exact ``[<marker> <old>] <path>`` token to index *new*."""
    replacement = f"[{marker} {new}] {path}"
    for at, end in reversed(_marker_spans(content, marker, old, path)):
        content = content[:at] + replacement + content[end:]
    return content


def prune_attachment_meta(meta: Any, content: str, previous: str) -> str:
    """Reconcile an entry's attachment lists with an edited *content*.

    An edit to a queued message can remove a ``[attached_file N] path`` marker;
    the agent then receives text without that path and never gets the file, so
    a list still naming it would make the drained row show a card for an
    attachment that was never delivered. Each list is filtered in place to the
    entries the edit did not remove (order kept; a list left empty is removed),
    and the surviving markers in the text are renumbered to the filtered list's
    positions. The renumbering is what keeps a spaced path lossless: the
    renderer reads ``files[N-1]`` for marker ``N`` and, when the two disagree,
    falls back to a whitespace-bounded capture of the marker text -- which
    would hand back ``/tmp/My`` for ``/tmp/My Report.pdf``.

    "Removed by the edit" means the exact numbered marker was in *previous*
    (the entry's text before this edit) and is not in *content*. An entry the
    previous text never named is out of the edit's reach and is kept as-is: a
    send can carry a list entry with no marker (a caller that stamps
    ``meta.files`` on markerless text; a path the list redacted while the text
    kept it verbatim, so the two spellings differ), and the edit did not take
    that attachment away from the agent -- dropping it would make the drained
    row lose an attachment the user never touched. Returns the content to
    store; it equals *content* whenever nothing was pruned.
    """
    if not isinstance(meta, dict):
        return content
    for key in ATTACHMENT_META_KEYS:
        raw = meta.get(key)
        if not isinstance(raw, list):
            continue
        marker = _ATTACHMENT_MARKERS[key]
        indexed = [(i + 1, p) for i, p in enumerate(raw) if isinstance(p, str) and p]
        kept = [
            (old, p)
            for old, p in indexed
            if _marker_spans(content, marker, old, p) or not _marker_spans(previous, marker, old, p)
        ]
        if len(kept) == len(indexed):
            continue
        for new, (old, p) in enumerate(kept, start=1):
            if new != old:
                content = _renumber_marker(content, marker, old, new, p)
        if kept:
            meta[key] = [p for _, p in kept]
        else:
            meta.pop(key, None)
    return content


class SlotQueueRepository:
    """Mutate the current facade-owned queue and delivery ledger.

    Every operation receives its owner explicitly.  Replay and cleanup paths
    replace ``_queue`` and ``_subagent_delivery_pending`` wholesale, so keeping
    either container on this repository would split the slot into two states.
    """

    def __init__(
        self,
        *,
        id_provider: Callable[[], str] | None = None,
        timestamp_provider: Callable[[], str] | None = None,
        delivery_key: Callable[[str], str] = _delivery_key,
        max_pending_deliveries: Callable[[], int] | None = None,
    ) -> None:
        self._id_provider = id_provider or (lambda: uuid.uuid4().hex[:12])
        self._timestamp_provider = timestamp_provider or (
            lambda: datetime.now(timezone.utc).isoformat()
        )
        self._delivery_key = delivery_key
        self._max_pending_deliveries = max_pending_deliveries or (
            lambda: MAX_PENDING_SUBAGENT_DELIVERIES
        )

    def queue_append(
        self,
        owner: Any,
        content: str,
        kind: str = "",
        meta: dict | None = None,
        *,
        directive_user_origin: bool = False,
        directive_channel_origin: bool = False,
    ) -> str:
        """Append an entry and return its process-local queue ID."""
        queue_id = self._id_provider()
        item: dict[str, Any] = {
            "id": queue_id,
            "content": content,
            "kind": kind,
        }
        # Append deliberately retains the producer's metadata object: enqueue
        # sites can finish populating structured facts after constructing it.
        if meta:
            item["meta"] = meta
        if directive_user_origin:
            item["_directive_user_origin"] = True
        if directive_channel_origin:
            item["_directive_channel_origin"] = True
        _stamp_origin(owner, item, directive_channel_origin=directive_channel_origin)
        owner._queue.append(item)
        owner._note_enqueue()
        return queue_id

    def note_enqueue(self, owner: Any) -> None:
        """Record queue activity beside, rather than inside, an entry."""
        # Queue dicts are compared wholesale on the wire and in persistence
        # tests; placing the clock there would make their shape time-dependent.
        owner._last_enqueue_ts = self._timestamp_provider()

    def queue_insert(
        self,
        owner: Any,
        index: int,
        content: str,
        kind: str = "",
        payload: str = "",
        meta: dict | None = None,
        on_consumed: Callable[[bool], None] | None = None,
        on_irreversibly_consumed: Callable[[], Awaitable[None] | None] | None = None,
        directive_user_origin: bool = False,
        directive_channel_origin: bool = False,
    ) -> str:
        """Insert one entry while preserving retry callbacks and provenance."""
        queue_id = self._id_provider()
        item: dict[str, Any] = {
            "id": queue_id,
            "content": content,
            "kind": kind,
            "payload": payload,
        }
        # Insert is the recovery path: its process-local retry entry owns a
        # snapshot, so later producer mutation must not rewrite queued facts.
        if meta:
            item["meta"] = dict(meta)
        if on_consumed is not None:
            item["_on_consumed"] = on_consumed
        if on_irreversibly_consumed is not None:
            item["_on_irreversibly_consumed"] = on_irreversibly_consumed
        if directive_user_origin:
            item["_directive_user_origin"] = True
        if directive_channel_origin:
            item["_directive_channel_origin"] = True
        _stamp_origin(owner, item, directive_channel_origin=directive_channel_origin)
        owner._queue.insert(index, item)
        owner._note_enqueue()
        return queue_id

    def queue_pop(self, owner: Any, index: int = 0) -> dict[str, Any]:
        """Remove and return the exact entry at *index*."""
        return owner._queue.pop(index)

    def note_pending_subagent_delivery(
        self,
        owner: Any,
        content: str,
        agent_ids: list[str],
    ) -> None:
        """Remember which agents a queued completion still owes delivery."""
        if not content or not agent_ids:
            return
        key = self._delivery_key(content)
        owed = owner._subagent_delivery_pending.setdefault(key, [])
        owed.extend(agent_id for agent_id in agent_ids if agent_id not in owed)
        # Only the consuming row may settle an entry.  A turn tail can dequeue
        # its successor before the current settlement callback runs, so sweeping
        # merely because content left the queue would lose the successor's debt.
        while len(owner._subagent_delivery_pending) > self._max_pending_deliveries():
            owner._subagent_delivery_pending.pop(next(iter(owner._subagent_delivery_pending)))

    def owes_subagent_delivery(self, owner: Any, contents: list[str]) -> bool:
        """Return whether any named completion has unsettled delivery debt."""
        return any(
            self._delivery_key(content) in owner._subagent_delivery_pending for content in contents
        )

    def take_pending_subagent_deliveries(self, owner: Any, contents: list[str]) -> list[str]:
        """Claim delivery marks in consumed-row order and forget only those rows."""
        claimed: list[str] = []
        for content in contents:
            claimed.extend(owner._subagent_delivery_pending.pop(self._delivery_key(content), []))
        return claimed

    def queue_remove_by_id(self, owner: Any, queue_id: str) -> str | None:
        """Remove the matching entry and return its content."""
        for index, item in enumerate(owner._queue):
            if item["id"] == queue_id:
                del owner._queue[index]
                return item["content"]
        return None

    def queue_edit_by_id(
        self,
        owner: Any,
        queue_id: str,
        content: str,
        *,
        directive_user_origin: bool = False,
        directive_channel_origin: bool = False,
    ) -> bool:
        """Edit a user-owned entry without changing its identity or position."""
        for item in owner._queue:
            if item["id"] != queue_id:
                continue
            # Retry callbacks settle the exact automatic payload that failed;
            # moving them to replacement text would acknowledge the wrong work.
            if "_on_consumed" in item or "_on_irreversibly_consumed" in item:
                return False
            # The lists index the OLD text's markers; drop only what this edit
            # removed (named before, unnamed now) and renumber the survivors
            # (prune_attachment_meta). An entry the old text never named is
            # not the edit's to drop.
            previous = item.get("content")
            item["content"] = prune_attachment_meta(
                item.get("meta"), content, previous if isinstance(previous, str) else ""
            )
            if directive_user_origin:
                item["_directive_user_origin"] = True
            else:
                item.pop("_directive_user_origin", None)
            if directive_channel_origin:
                item["_directive_channel_origin"] = True
            else:
                item.pop("_directive_channel_origin", None)
            _stamp_origin(owner, item, directive_channel_origin=directive_channel_origin)
            return True
        return False

    def queue_promote_by_id(self, owner: Any, queue_id: str) -> bool:
        """Move the exact matching entry to the front without rebuilding it."""
        for index, item in enumerate(owner._queue):
            if item["id"] == queue_id:
                owner._queue.insert(0, owner._queue.pop(index))
                return True
        return False
