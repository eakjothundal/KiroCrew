"""The queue's provenance proof: minted by the gateway, verified at the restore and the drain.

A queued prompt persisted to the session's metadata line is an ordinary writable
file. Whatever provenance the line carries is worth what the file is worth, so a
RESTORED entry has none by default: no command authority (it drains as channel
text), no admission snapshot (it is re-checked against every constraint that
holds now) and no channel address (a released binding neither drops nor reports
it) -- unless the gateway can prove it accepted the entry with exactly that
provenance. This module is that proof: an HMAC-SHA256 over the slot key, the
queue id, the content, the channel conversation the entry came from (none for
dashboard text) and the containment snapshot recorded at admission, under a key
derived (domain-separated) from the fenced ``token_signing.key`` secret that
also signs dashboard tokens. Nothing an editor can write to the file produces a
valid one: rewritten words, a rewritten or hand-added address or snapshot, a
proof moved between entries or slots, or a hand-written tag all verify as
nothing.

The proof lives BESIDE the queue, never on the entry (see
``slot_queue_repository``): the live entry dict is exactly what the caller
enqueued, the board never sees the tag, and the durable record carries it as its
own field for the restore path to verify and read back.

This module's path carries ``token`` on purpose: the argv floor's credential-mint
rule denies an inline program that imports a product module named for the token
mint, which is the convention ``test_argv_floor_inline_and_brace_scope`` derives
from the tree -- every function that can produce a credential from the signing
key must sit behind a module path or a name that rule reads. Consumers import
this MODULE and call through it rather than re-exporting the producer names
under a path the gate does not cover.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

#: Domain separation for the proof key: the signing secret also signs dashboard
#: access and refresh tokens, so the queue proof is computed under a key DERIVED
#: from it for this purpose alone. Versioned with the signed record's shape.
_ORIGIN_PROOF_DOMAIN = b"kirocrew.dashboard.queue-provenance-proof.v2"
_ORIGIN_PROOF_KEY: bytes | None = None


def _origin_proof_key() -> bytes:
    """The derived proof key, computed once per process from the fenced secret.

    ``token_secret._get_secret`` never raises: a home it cannot write falls back
    to a per-process random secret, under which proofs still verify within the
    process and simply fail after a restart -- the fail-closed direction, since an
    unverifiable restored entry is prose with no admission and no address.
    """
    global _ORIGIN_PROOF_KEY
    if _ORIGIN_PROOF_KEY is None:
        # Local import: the secret must stay lazy (importing this module must
        # never create the key file), and the dashboard package imports the queue
        # repository early.
        from kiro_crew.dashboard.token_secret import _get_secret

        _ORIGIN_PROOF_KEY = hmac.new(_get_secret(), _ORIGIN_PROOF_DOMAIN, hashlib.sha256).digest()
    return _ORIGIN_PROOF_KEY


def _canonical(part: dict[str, Any] | None) -> str:
    """One signed mapping as text: sorted keys, no whitespace, ``""`` for none.

    The same bytes on both sides of a JSON round trip, which is what the durable
    record puts the address and the snapshot through: the writer signs the live
    dicts and the reader verifies the ones ``json.loads`` handed back. An empty
    string for an absent part is distinct from ``"{}"`` for an empty one, so
    "no address" and "an address with nothing in it" are different records.
    Raises ``TypeError``/``ValueError`` for a mapping json cannot emit, which the
    callers read as "no proof".
    """
    if part is None:
        return ""
    return json.dumps(part, sort_keys=True, separators=(",", ":"))


def queue_provenance_proof(
    slot_key: str,
    queue_id: str,
    content: str,
    *,
    channel_address: dict[str, Any] | None,
    admission: dict[str, Any] | None,
) -> str:
    """The proof the gateway stamps for an entry of *slot_key* it accepted as
    *content* from *channel_address* (``None`` for dashboard text) under the
    admission-time containment *admission* (``None`` when the producer stamped
    none).

    Length-prefixed fields, so no choice of key, id, content, address or snapshot
    can be read as another split of the same bytes.
    """
    parts = (slot_key, queue_id, content, _canonical(channel_address), _canonical(admission))
    message = "".join(f"{len(part)}:{part}" for part in parts).encode("utf-8", "surrogatepass")
    return hmac.new(_origin_proof_key(), message, hashlib.sha256).hexdigest()


def dashboard_origin_proof(
    slot_key: str, queue_id: str, content: str, *, admission: dict[str, Any] | None = None
) -> str:
    """The proof of a DASHBOARD-origin entry: :func:`queue_provenance_proof` with
    no channel address. The address-less proof is what grants a restored entry the
    composer's command word."""
    return queue_provenance_proof(
        slot_key, queue_id, content, channel_address=None, admission=admission
    )


def queue_provenance_matches(
    slot_key: str,
    queue_id: str,
    content: object,
    channel_address: object,
    admission: object,
    tag: object,
) -> bool:
    """Whether *tag* is exactly the proof over the five other arguments.

    False for a missing or non-string tag or content, an address or snapshot that
    is neither a mapping nor absent, a mapping json cannot re-emit, and for any
    mismatch -- rewritten content, a rewritten or hand-added address or snapshot,
    a transplanted proof, another slot. Constant-time comparison; never raises.
    The untyped parameters come off an entry or a line nothing here trusts.
    """
    if not isinstance(tag, str) or not isinstance(queue_id, str) or not isinstance(content, str):
        return False
    if channel_address is not None and not isinstance(channel_address, dict):
        return False
    if admission is not None and not isinstance(admission, dict):
        return False
    try:
        expected = queue_provenance_proof(
            slot_key, queue_id, content, channel_address=channel_address, admission=admission
        )
    except (TypeError, ValueError):
        # A mapping json cannot canonicalise carries no proof the writer could
        # have minted either.
        return False
    return hmac.compare_digest(expected, tag)
