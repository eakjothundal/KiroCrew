"""The transcript DERIVATION seam, and the fence that keeps every consumer on it.

A restricted (incognito / temporary) transcript is kept on disk for the user to
reopen, and nothing may be DERIVED from it: no summary, no export or transfer
bundle, no suggestions prompt, no MCP history read, no consolidation, no skill.
Every one of those readers takes its rows from disk, and the file's own
``memory_mode`` line is the privacy contract -- a ratchet any writer may tighten
between a reader's check and its read (a same-key hand-over landing a closed
restricted tab's rows; a second gateway on the same data home). Guarding each
consumer separately produced a new finding per consumer. The fix is one seam:
``ConversationLog.derive_messages`` / ``derive_messages_chained`` /
``derive_recent`` (and ``snapshot_for_consolidation(withhold_restricted=True)``)
validate the line and read the rows under ONE ``_locked`` hold and raise
``TranscriptWithheld`` instead of yielding rows a restricted or unreadable line
governs.

The plain reads stay for transcript PLUMBING (resume, save, rewind, fork,
mirror, the History browser, migrations, injections), which must see a
restricted transcript. The fence below enumerates every plain-read reference in
the source tree: a new consumer written against a plain read fails this test and
must either move to the seam or name itself here as plumbing, with the reviewer
seeing that choice in the diff.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from kiro_crew import history as history_mod
from kiro_crew.history import ConversationLog, TranscriptWithheld

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# Every attribute name that reads transcript ROWS from disk without the seam.
PLAIN_READS = frozenset(
    {
        "read_messages",
        "read_messages_chained",
        "read_messages_chained_full",
        "recent",
        "recent_chained",
        "_read_messages",
        "_read_messages_locked",
        "_recent_via_tail",
        "sliding_window",
        "read_file_change_messages",
        "get_unconsolidated",
    }
)

# (module path under src/, plain read) pairs that are TRANSCRIPT PLUMBING --
# they resume, save, rewind, fork, mirror, render, migrate or inject the
# transcript itself and must see a restricted one. Anything that hands rows to a
# model, a peer, a downloadable file or a memory store does NOT belong here: it
# goes through the derivation seam. Keep this sorted; a stale entry fails too.
PLUMBING: frozenset[tuple[str, str]] = frozenset(
    {
        # An app reading its own slot's conversation for the user.
        ("kiro_crew/apps/builtins/spec_builder/backend/runtime.py", "read_messages"),
        # Channel file migration: moves rows between stems, derives nothing.
        ("kiro_crew/channel_transcript_migration.py", "read_messages"),
        # SEL audit log `recent`, not a transcript.
        ("kiro_crew/cli_commands.py", "recent"),
        # The live session's OWN rows into its OWN prompt / compaction.
        ("kiro_crew/context.py", "read_messages"),
        ("kiro_crew/context.py", "read_messages_chained"),
        ("kiro_crew/context.py", "recent"),
        ("kiro_crew/dashboard/channel_slots.py", "read_messages"),
        ("kiro_crew/dashboard/chat_backfill.py", "read_messages_chained"),
        ("kiro_crew/dashboard/chat_backfill.py", "recent"),
        ("kiro_crew/dashboard/chat_fork.py", "read_messages_chained"),
        ("kiro_crew/dashboard/chat_fork.py", "read_messages_chained_full"),
        ("kiro_crew/dashboard/chat_handlers.py", "read_messages_chained"),
        ("kiro_crew/dashboard/chat_handlers.py", "read_messages_chained_full"),
        # ``selection.recent`` -- a mirror selection field, not a transcript read.
        ("kiro_crew/dashboard/chat_mirror.py", "recent"),
        ("kiro_crew/dashboard/chat_persistence.py", "read_messages_chained"),
        ("kiro_crew/dashboard/chat_rewind.py", "read_messages_chained"),
        ("kiro_crew/dashboard/chat_runner.py", "recent"),
        ("kiro_crew/dashboard/chat_slack.py", "recent"),
        ("kiro_crew/dashboard/chat_threads.py", "read_messages_chained"),
        ("kiro_crew/dashboard/cron_inject.py", "read_messages"),
        # Rendering the transcript's own artifacts / rows to its user.
        ("kiro_crew/dashboard/handlers/artifacts.py", "read_messages"),
        # SEL audit log `recent`, not a transcript.
        ("kiro_crew/dashboard/handlers/core.py", "recent"),
        ("kiro_crew/dashboard/handlers/cron.py", "read_messages"),
        ("kiro_crew/dashboard/handlers/session_control.py", "read_messages"),
        # The History browser showing the user their own transcript. Its
        # list_sessions(summarize=true) leg goes through the seam (derive_recent).
        ("kiro_crew/dashboard/handlers/sessions.py", "read_messages"),
        ("kiro_crew/decisions/points/compaction_keep.py", "read_messages"),
        # Thread diagnostics probe `recent`, not a transcript.
        ("kiro_crew/diag/threads.py", "recent"),
        ("kiro_crew/discord/session_resume.py", "recent"),
        # SEL `recent`, not a transcript.
        ("kiro_crew/feature_videos.py", "recent"),
        # The log and its read projection ARE the plain reads.
        ("kiro_crew/history.py", "_read_messages"),
        ("kiro_crew/history.py", "_read_messages_locked"),
        ("kiro_crew/history.py", "_recent_via_tail"),
        ("kiro_crew/history.py", "read_file_change_messages"),
        ("kiro_crew/history.py", "read_messages"),
        ("kiro_crew/history.py", "read_messages_chained"),
        ("kiro_crew/history.py", "read_messages_chained_full"),
        ("kiro_crew/history.py", "recent"),
        ("kiro_crew/history.py", "recent_chained"),
        ("kiro_crew/history.py", "sliding_window"),
        # Scheduling counts only (`len(...)`); the rows it consolidates come from
        # the gated snapshot and skill detection goes through derive_messages.
        ("kiro_crew/history_consolidation.py", "_read_messages"),
        ("kiro_crew/history_projection.py", "_read_messages"),
        ("kiro_crew/history_projection.py", "_read_messages_locked"),
        ("kiro_crew/history_projection.py", "_recent_via_tail"),
        ("kiro_crew/history_projection.py", "read_messages_chained"),
        ("kiro_crew/slack/gateway.py", "read_messages"),
        ("kiro_crew/teams/session_resume.py", "recent"),
    }
)


def _plain_read_sites() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in PLAIN_READS:
                found.add((rel, node.attr))
    return found


class TestTheFence:
    def test_every_plain_read_in_the_tree_is_named_as_plumbing(self):
        """A consumer written against a plain read must declare itself.

        If this fails on a site you just added: hands the rows to a model, a
        peer, a downloadable file or a memory store? Then use
        ``derive_messages`` / ``derive_messages_chained`` / ``derive_recent``
        (or ``snapshot_for_consolidation(withhold_restricted=True)``) and catch
        ``TranscriptWithheld``. Resuming, saving, rendering or migrating the
        transcript itself? Add the ``(module, read)`` pair to ``PLUMBING`` with a
        one-line reason.
        """
        found = _plain_read_sites()
        unnamed = sorted(found - PLUMBING)
        assert not unnamed, f"plain transcript reads not named as plumbing: {unnamed}"

    def test_the_plumbing_list_carries_no_stale_entry(self):
        stale = sorted(PLUMBING - _plain_read_sites())
        assert not stale, f"PLUMBING names reads that no longer exist: {stale}"

    def test_the_known_derivers_do_not_use_plain_reads(self):
        """The consumers this seam exists for are not merely fenced -- they are off it."""
        derivers = {
            "kiro_crew/dashboard/chat_summary.py",
            "kiro_crew/dashboard/session_transfer.py",
            "kiro_crew/dashboard/session_export.py",
            "kiro_crew/suggestions.py",
            "kiro_crew/mcp_tools/sessions.py",
        }
        offenders = sorted(site for site in _plain_read_sites() if site[0] in derivers)
        assert not offenders, offenders


KEY = "dashboard:seam"


def _log(tmp_path) -> ConversationLog:
    log = ConversationLog(base_dir=tmp_path / "sessions")
    log.init()
    with history_mod.allow_on_loop_persist():
        log.append(KEY, "user", "PRIVATE-1")
        log.append(KEY, "assistant", "PRIVATE-2")
    return log


class TestTheSeam:
    def test_a_persistent_line_yields_the_same_rows_as_the_plain_read(self, tmp_path):
        log = _log(tmp_path)
        assert log.derive_messages(KEY) == log.read_messages(KEY)
        assert log.derive_messages_chained(KEY) == log.read_messages_chained(KEY)
        assert log.derive_recent(KEY, 1) == log.recent(KEY, 1)
        assert [m["content"] for m in log.derive_recent(KEY, 5, roles={"user"})] == ["PRIVATE-1"]

    def test_an_absent_file_is_not_a_refusal(self, tmp_path):
        log = ConversationLog(base_dir=tmp_path / "sessions")
        log.init()
        assert log.derive_messages("dashboard:nothing-here") == []
        assert log.derive_recent("dashboard:nothing-here") == []

    @pytest.mark.parametrize("mode", ["incognito", "temporary", "Incognito"])
    def test_a_restricted_line_withholds_every_shape(self, tmp_path, mode):
        log = _log(tmp_path)
        with history_mod.allow_on_loop_persist():
            log.update_metadata(KEY, {"memory_mode": mode})
        for read in (
            lambda: log.derive_messages(KEY),
            lambda: log.derive_messages_chained(KEY),
            lambda: log.derive_recent(KEY, 5),
            lambda: log.snapshot_for_consolidation(KEY, withhold_restricted=True),
        ):
            with pytest.raises(TranscriptWithheld):
                read()
        # The plain reads still serve the transcript's own plumbing.
        assert [m["content"] for m in log.read_messages(KEY)] == ["PRIVATE-1", "PRIVATE-2"]

    def test_an_unreadable_line_withholds(self, tmp_path, monkeypatch):
        log = _log(tmp_path)
        monkeypatch.setattr(type(log), "_read_metadata_status", lambda self, key: ({}, False))
        with pytest.raises(TranscriptWithheld):
            log.derive_messages(KEY)

    def test_the_line_is_validated_under_the_same_lock_as_the_rows(self, tmp_path, monkeypatch):
        """A tightening that lands while the seam holds the lock cannot slip a row out.

        The writer that tightens a line (the hand-over save) takes ``_locked`` too,
        so under the seam's hold it waits; a tightening that lands first is seen by
        the seam's own check. Modelled here by tightening from INSIDE the lock,
        after the check: the rows the seam returns are still the ones the check
        vouched for, because both happened in one hold -- and the next derivation
        is refused.
        """
        log = _log(tmp_path)
        real_read = type(log).read_messages
        state = {"tightened": False}

        def _read_then_tighten(self, key):
            rows = real_read(self, key)
            if not state["tightened"]:
                state["tightened"] = True
                with history_mod.allow_on_loop_persist():
                    self.update_metadata(key, {"memory_mode": "incognito"})
            return rows

        monkeypatch.setattr(type(log), "read_messages", _read_then_tighten)
        first = log.derive_messages(KEY)
        assert [m["content"] for m in first] == ["PRIVATE-1", "PRIVATE-2"]
        with pytest.raises(TranscriptWithheld):
            log.derive_messages(KEY)


class TestTheSeamCoversTheWholeChain:
    """A chained read concatenates every transcript sharing the tab id.

    The contract that governs the result is the strictest line among ALL of them:
    a legacy tab whose earlier file was tightened to ``temporary`` must not ride
    its rows out under a persistent sibling's line. Every chained transcript is
    locked and validated before a row is read.
    """

    @staticmethod
    def _chain(tmp_path) -> ConversationLog:
        log = ConversationLog(base_dir=tmp_path / "sessions")
        log.init()
        with history_mod.allow_on_loop_persist():
            log.append("dashboard:chat-0", "user", "OLDER-PRIVATE", tab_id="tab-legacy")
            log.append("dashboard:chat-1", "user", "NEWER-PUBLIC", tab_id="tab-legacy")
        assert log.chained_keys("dashboard:chat-1") == ["dashboard:chat-0", "dashboard:chat-1"]
        return log

    def test_a_persistent_chain_reads_every_file(self, tmp_path):
        log = self._chain(tmp_path)
        rows = log.derive_messages_chained("dashboard:chat-1")
        assert [m["content"] for m in rows] == ["OLDER-PRIVATE", "NEWER-PUBLIC"]
        assert rows == log.read_messages_chained("dashboard:chat-1")

    @pytest.mark.parametrize("mode", ["incognito", "temporary", "Temporary"])
    def test_a_restricted_sibling_withholds_the_whole_chain(self, tmp_path, mode):
        log = self._chain(tmp_path)
        with history_mod.allow_on_loop_persist():
            log.update_metadata("dashboard:chat-0", {"memory_mode": mode})
        # The requested key's own line is still persistent...
        assert "memory_mode" not in log.get_metadata("dashboard:chat-1")
        # ...and the seam still refuses, because the sibling's line governs its rows.
        with pytest.raises(TranscriptWithheld):
            log.derive_messages_chained("dashboard:chat-1")
        # Plumbing still sees the chain.
        assert len(log.read_messages_chained("dashboard:chat-1")) == 2

    def test_an_unreadable_sibling_line_withholds_the_whole_chain(self, tmp_path, monkeypatch):
        log = self._chain(tmp_path)
        real_status = type(log)._read_metadata_status

        def _status(self, key):
            if key == "dashboard:chat-0":
                return {}, False
            return real_status(self, key)

        monkeypatch.setattr(type(log), "_read_metadata_status", _status)
        with pytest.raises(TranscriptWithheld):
            log.derive_messages_chained("dashboard:chat-1")

    def test_a_chain_that_grows_while_being_locked_is_refused(self, tmp_path, monkeypatch):
        """A file joining the chain between the resolve and the hold is unlocked and
        unvalidated, so the read is refused rather than served."""
        log = self._chain(tmp_path)
        real_chained = type(log).chained_keys
        calls = {"n": 0}

        def _grow_on_second_resolve(self, key):
            keys = real_chained(self, key)
            calls["n"] += 1
            if calls["n"] == 2:
                return keys + ["dashboard:chat-late"]
            return keys

        monkeypatch.setattr(type(log), "chained_keys", _grow_on_second_resolve)
        with pytest.raises(TranscriptWithheld):
            log.derive_messages_chained("dashboard:chat-1")
