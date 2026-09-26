"""Agent spec hooks on KAS: auto-approved calls, ``confirm: true`` hooks, and
subagent / task-runner turns.

* A PreToolUse hook runs on Crew's permission path, which an auto-approved call
  never takes, so the KAS projection withholds auto-approval for what such a hook
  covers.
* A ``confirm: true`` spec hook cannot run from Crew, and the user is told once
  per session.
* A subagent or task-runner turn gates each permission request on the spec
  hooks of ITS OWN agent, only when ITS OWN backend never receives them. kiro-cli runs the field itself, so a
  kiro-cli turn gets none from Crew and nothing fires twice.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kiro_crew.config.paths as paths_mod
import kiro_crew.hooks as hooks_mod
from kiro_crew.acp import kas_agents, kas_permissions
from kiro_crew.agent_sdk import spec_hooks
from kiro_crew.agent_sdk.backends import ACP_BACKEND_KAS, ACP_BACKEND_KIRO
from kiro_crew.agent_sdk.capabilities import capabilities_for
from kiro_crew.dashboard import chat_runner
from kiro_crew.hooks import (
    HOOK_EVENT_POST_TOOL_USE,
    HOOK_EVENT_PRE_TOOL_USE,
    ScriptHook,
    ScriptHookResult,
    ScriptHookStore,
)

_ALL = set(kas_permissions.AUTO_APPROVABLE_CAPABILITIES)
_PRE = {"preToolUse": [{"matcher": "web_fetch", "command": "guard.sh"}]}
_CONFIRM_DOCS = [
    {"name": "on", "trigger": "PreToolUse", "action": {"type": "command", "command": "a"}},
    {
        "name": "ask",
        "trigger": "PreToolUse",
        "confirm": True,
        "action": {"type": "command", "command": "b"},
    },
    {
        "name": "ask too",
        "trigger": "Stop",
        "confirm": True,
        "action": {"type": "command", "command": "c"},
    },
]


@pytest.fixture(autouse=True)
def _fresh_cache():
    spec_hooks._cache.clear()
    yield
    spec_hooks._cache.clear()


@pytest.fixture
def agents_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "agents"
    d.mkdir()
    monkeypatch.setattr(paths_mod, "kiro_agents_dir", lambda: d)
    return d


def _write_spec(agents_dir: Path, name: str, **fields) -> None:
    spec = {"name": name, "prompt": "p", **fields}
    (agents_dir / f"{name}.json").write_text(json.dumps(spec), encoding="utf-8")


def _provider(backend: str, cwd: str = "") -> SimpleNamespace:
    return SimpleNamespace(capabilities=capabilities_for(backend), cwd=cwd)


def _rules(policy) -> list[tuple[str, str]]:
    return [(r["capability"], r["effect"]) for r in (policy or {}).get("rules", [])]


# ── item 1: an auto-approved call is withheld for a PreToolUse hook ──────────


@pytest.mark.parametrize(
    ("matchers", "covered"),
    [
        ((), set()),
        (("",), _ALL),
        (("*",), _ALL),
        (("web_fetch",), {"web_fetch"}),
        (("WEB_FETCH",), {"web_fetch"}),
        (("execute_bash",), set()),
        (("fs_write", "grep"), set()),
        (("mcp__github__create_issue",), {"mcp"}),
        (("@github/create_issue",), {"mcp"}),
        (("WebFetch",), _ALL),
        (("web_*",), {"web_fetch", "web_search", "mcp"}),
        (("execute_*",), {"mcp"}),
        (("invoke_sub_agent", "disclose_context"), {"subagent", "skill"}),
    ],
)
def test_the_capabilities_a_pre_tool_matcher_covers(matchers, covered):
    assert kas_permissions.hook_gated_capabilities(matchers) == covered


def test_an_ask_outranks_the_allow_a_hook_covers():
    policy = {"rules": [{"capability": "web_fetch", "effect": "allow"}]}
    out = kas_permissions.withhold_hook_gated_auto_approval(policy, ("web_fetch",))
    assert _rules(out) == [("web_fetch", "allow"), ("web_fetch", "ask")]
    # The input is left as it was.
    assert _rules(policy) == [("web_fetch", "allow")]


def test_no_hook_or_no_policy_changes_nothing():
    policy = {"rules": [{"capability": "web_fetch", "effect": "allow"}]}
    assert kas_permissions.withhold_hook_gated_auto_approval(policy, ()) is policy
    assert kas_permissions.withhold_hook_gated_auto_approval(policy, ("execute_bash",)) is policy
    assert kas_permissions.withhold_hook_gated_auto_approval(None, ("*",)) is None


def test_the_projection_withholds_what_a_pre_tool_hook_covers():
    spec = {"allowedTools": ["web_fetch", "web_search", "@srv"]}
    plain = kas_agents.to_client_custom_agent("a1", spec, "p")
    gated = kas_agents.to_client_custom_agent(
        "a1", spec, "p", pre_tool_hook_matchers=("web_fetch",)
    )
    assert ("web_fetch", "ask") not in _rules(plain["permissions"])
    assert ("web_fetch", "ask") in _rules(gated["permissions"])
    assert ("web_search", "ask") not in _rules(gated["permissions"])
    assert ("mcp", "ask") not in _rules(gated["permissions"])


def test_the_built_batch_reads_the_specs_own_pre_tool_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(kas_agents, "get_global_hook_store", lambda: None)
    spec = {"prompt": "p", "allowedTools": ["web_fetch"], "hooks": _PRE}
    (entry,) = kas_agents.build_kas_custom_agents(tmp_path, "a1", spec)
    assert ("web_fetch", "ask") in _rules(entry["permissions"])


def test_the_projection_lists_spec_and_hooks_page_pre_tool_matchers(tmp_path, monkeypatch):
    store = ScriptHookStore(config_dir=tmp_path)
    store._hooks = {
        "p1": ScriptHook(id="p1", event=HOOK_EVENT_PRE_TOOL_USE, matcher="web_*", command="x"),
        "p2": ScriptHook(id="p2", event=HOOK_EVENT_PRE_TOOL_USE, command="x", enabled=False),
        "p3": ScriptHook(id="p3", event=HOOK_EVENT_PRE_TOOL_USE, skills=["s"]),
        "p4": ScriptHook(id="p4", event=HOOK_EVENT_POST_TOOL_USE, command="x"),
    }
    monkeypatch.setattr(kas_agents, "get_global_hook_store", lambda: store)
    matchers = kas_agents.pre_tool_hook_matchers("a1", {"hooks": _PRE})
    assert sorted(matchers) == ["web_*", "web_fetch"]


def test_the_projection_gates_everything_when_it_cannot_list_hooks(monkeypatch):
    def boom():
        raise RuntimeError("store broken")

    monkeypatch.setattr(kas_agents, "get_global_hook_store", boom)
    assert kas_agents.pre_tool_hook_matchers("a1", {}) == ("*",)


# ── item 2: a confirm: true spec hook is named to the user ───────────────────


@pytest.fixture
def notices(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(
        chat_runner, "append_and_surface", lambda state, slot, role, text, cls: seen.append(text)
    )
    return seen


def _prepare(provider, agent, *, is_new):
    return asyncio.run(
        chat_runner._prepare_spec_hooks(
            SimpleNamespace(), SimpleNamespace(), provider, agent, is_new=is_new
        )
    )


def test_confirm_hooks_get_one_notice_per_session(agents_dir, notices):
    _write_spec(agents_dir, "a1", hooks=_CONFIRM_DOCS)
    hooks, unreadable, _cwd = _prepare(_provider(ACP_BACKEND_KAS), "a1", is_new=True)
    assert [h.command for h in hooks] == ["a"]
    assert unreadable is False
    assert len(notices) == 1
    assert "2 hooks ask to be confirmed" in notices[0]
    notices.clear()
    _prepare(_provider(ACP_BACKEND_KAS), "a1", is_new=False)
    assert notices == []


def test_no_confirm_notice_without_confirm_hooks_or_on_kiro_cli(agents_dir, notices):
    _write_spec(agents_dir, "plain", hooks=_PRE)
    _write_spec(agents_dir, "a1", hooks=_CONFIRM_DOCS)
    _prepare(_provider(ACP_BACKEND_KAS), "plain", is_new=True)
    _prepare(_provider(ACP_BACKEND_KIRO), "a1", is_new=True)
    assert notices == []


def test_one_confirm_hook_reads_in_the_singular():
    assert "1 hook asks to be" in chat_runner._spec_confirm_hooks_notice("a1", 1)


# ── item 5: subagent and task-runner turns gate on their own spec hooks ──────


def test_a_kiro_cli_turn_gets_no_spec_hooks_and_reads_no_spec(monkeypatch):
    def must_not_read(_agent):
        raise AssertionError("a kiro-cli turn must not read the spec")

    monkeypatch.setattr(spec_hooks, "crew_fired_spec_hooks", must_not_read)
    out = asyncio.run(spec_hooks.turn_spec_hooks(_provider(ACP_BACKEND_KIRO, "/w"), "a1"))
    assert out == ([], None, False)


def test_a_kas_turn_gets_its_own_agents_spec_hooks(agents_dir):
    _write_spec(agents_dir, "worker", hooks=_PRE)
    hooks, cwd, unreadable = asyncio.run(
        spec_hooks.turn_spec_hooks(_provider(ACP_BACKEND_KAS, "/w"), "worker")
    )
    assert [(h.event, h.matcher, h.command) for h in hooks] == [
        (HOOK_EVENT_PRE_TOOL_USE, "web_fetch", "guard.sh")
    ]
    assert (cwd, unreadable) == ("/w", False)


def test_a_kas_turn_with_an_unreadable_spec_says_so(agents_dir):
    (agents_dir / "worker.json").write_text("{not json", encoding="utf-8")
    out = asyncio.run(spec_hooks.turn_spec_hooks(_provider(ACP_BACKEND_KAS, "/w"), "worker"))
    assert out == ([], "/w", True)


def _spec_hook(matcher: str = "Read*") -> ScriptHook:
    return ScriptHook(
        id="spec:a1:PreToolUse:0", event=HOOK_EVENT_PRE_TOOL_USE, matcher=matcher, command="g"
    )


@pytest.mark.parametrize(("exit_code", "blocked"), [(0, False), (2, True), (1, True), (-1, True)])
def test_the_spec_gate_blocks_on_deny_and_on_no_verdict(tmp_path, monkeypatch, exit_code, blocked):
    ran: list = []

    async def fake_run(hook, context="", hook_event=None, cwd=None):
        ran.append((hook.command, hook_event.get("tool_name"), cwd))
        return ScriptHookResult(
            hook_id=hook.id, hook_name=hook.name, event=hook.event, exit_code=exit_code
        )

    monkeypatch.setattr(hooks_mod, "run_script_hook", fake_run)
    store = ScriptHookStore(config_dir=tmp_path)
    store._hooks = {"page": ScriptHook(id="page", event=HOOK_EVENT_PRE_TOOL_USE, command="p")}
    reason = asyncio.run(
        hooks_mod.spec_pre_tool_block(store, [_spec_hook()], "/w", "Running: ReadFile")
    )
    # Only the spec's hook runs here; the Hooks page's keeps its own fire site.
    assert ran == [("g", "ReadFile", "/w")]
    assert (reason is not None) is blocked


def test_the_spec_gate_passes_with_no_hooks_and_blocks_with_no_store():
    assert asyncio.run(hooks_mod.spec_pre_tool_block(None, [], None, "x")) is None
    assert asyncio.run(hooks_mod.spec_pre_tool_block(None, [_spec_hook()], None, "x")) is not None


def _deny_result(*_a, **kwargs):
    return [
        ScriptHookResult(
            hook_id="spec", hook_name="spec", event=HOOK_EVENT_PRE_TOOL_USE, exit_code=2
        )
    ]


def _subagent_run(backend: str, monkeypatch, *, unreadable: bool = False):
    """One subagent run on *backend* whose agent spec carries a denying PreToolUse hook."""
    from kiro_crew.execution_context import execution_for_store
    from kiro_crew.providers.base import (
        EVENT_PERMISSION_REQUEST,
        EVENT_TOOL_CALL,
        EVENT_TOOL_RESULT,
        LLMEvent,
    )
    from kiro_crew.subagent import SubagentInfo, SubagentManager

    def read(agent):
        if unreadable:
            raise OSError("unreadable")
        return [_spec_hook("*")], [], 0

    monkeypatch.setattr(spec_hooks, "crew_fired_spec_hooks", read)

    async def _stream(*_a, **_kw):
        yield LLMEvent(kind=EVENT_PERMISSION_REQUEST, title="Running: ReadFile", request_id="r-1")
        yield LLMEvent(kind=EVENT_TOOL_CALL, title="Running: ReadFile", tool_call_id="t-1")
        yield LLMEvent(kind=EVENT_TOOL_RESULT, tool_call_id="t-1", tool_output="ok")

    provider = AsyncMock()
    provider.capabilities = capabilities_for(backend)
    provider.cwd = "/w"
    provider.context_usage_pct = lambda: 0.0
    provider.stream = MagicMock(side_effect=lambda *a, **kw: _stream())
    sessions = MagicMock()
    sessions.get_pid = MagicMock(return_value=None)
    sessions.get_or_create = AsyncMock(return_value=(provider, True, False))
    sessions.reset = AsyncMock()
    sessions.get_agent = MagicMock(return_value="")
    sessions.get_agent_selection = MagicMock(return_value=("template", ""))
    ctx = MagicMock()
    ctx.build_message = MagicMock(return_value=("built_message", None))
    ctx.hooks.auto_approve_subagent_spawn = True
    manager = SubagentManager(sessions=sessions, ctx_builder=ctx)
    manager.hook_store = MagicMock()
    manager.hook_store.fire = AsyncMock(side_effect=_deny_result)
    info = SubagentInfo(
        execution_context=execution_for_store("", template_id="worker"),
        id="t01",
        task="test",
        parent_session_key="dashboard:parent",
    )
    manager._log_spawned(info)
    with patch("kiro_crew.subagent.Stats"), patch("kiro_crew.subagent.sel"):
        asyncio.run(manager._run_inner(info, "subagent:t01"))
    return manager.hook_store.fire, provider


def _gate_calls(fire) -> list:
    return [c for c in fire.await_args_list if c.kwargs.get("stored_hooks") is False]


@pytest.mark.usefixtures("healthy_host_memory")
def test_a_kas_subagent_spec_hook_denies_the_permission_request(monkeypatch):
    fire, provider = _subagent_run(ACP_BACKEND_KAS, monkeypatch)
    (gate,) = _gate_calls(fire)
    assert gate.args[0] == HOOK_EVENT_PRE_TOOL_USE
    assert [h.id for h in gate.kwargs["extra_hooks"]] == ["spec:a1:PreToolUse:0"]
    assert gate.kwargs["extra_hooks_cwd"] == "/w"
    provider.reject_tool.assert_awaited_once_with("r-1")
    provider.approve_tool.assert_not_awaited()
    # The informational tool-call fire does not run the spec's hooks a second time,
    # and PostToolUse still gets them.
    pre_info = [
        c
        for c in fire.await_args_list
        if c.args[0] == HOOK_EVENT_PRE_TOOL_USE and "stored_hooks" not in c.kwargs
    ]
    assert pre_info and all("extra_hooks" not in c.kwargs for c in pre_info)
    (post,) = [c for c in fire.await_args_list if c.args[0] == HOOK_EVENT_POST_TOOL_USE]
    assert [h.id for h in post.kwargs["extra_hooks"]] == ["spec:a1:PreToolUse:0"]


@pytest.mark.usefixtures("healthy_host_memory")
def test_a_kas_subagent_with_an_unreadable_spec_denies_the_request(monkeypatch):
    fire, provider = _subagent_run(ACP_BACKEND_KAS, monkeypatch, unreadable=True)
    assert _gate_calls(fire) == []
    provider.reject_tool.assert_awaited_once_with("r-1")


@pytest.mark.usefixtures("healthy_host_memory")
def test_a_kiro_cli_subagent_fires_no_spec_hooks_from_crew(monkeypatch):
    fire, provider = _subagent_run(ACP_BACKEND_KIRO, monkeypatch)
    assert _gate_calls(fire) == []
    for call in fire.await_args_list:
        assert list(call.kwargs.get("extra_hooks", ())) == []


def _task_step(backend: str, monkeypatch) -> tuple[list, list]:
    """One task-runner step on *backend* whose agent spec carries a denying hook."""
    from kiro_crew import task_executor
    from kiro_crew.acp.types import STOP_REASON_END_TURN, TurnUsage
    from kiro_crew.providers.base import (
        EVENT_COMPLETE,
        EVENT_PERMISSION_REQUEST,
        EVENT_TOOL_CALL,
        LLMEvent,
    )
    from kiro_crew.task_models import Project, Task

    monkeypatch.setattr(
        spec_hooks, "crew_fired_spec_hooks", lambda agent: ([_spec_hook("*")], [], 0)
    )
    store = MagicMock()
    store.fire = AsyncMock(side_effect=_deny_result)
    monkeypatch.setattr(task_executor, "get_global_hook_store", lambda: store)
    informational: list = []

    async def spy(*args, **kwargs):
        informational.append(kwargs)

    monkeypatch.setattr(task_executor, "fire_tool_hooks", spy)
    monkeypatch.setattr(task_executor, "check_context", AsyncMock())
    monkeypatch.setattr(
        task_executor, "build_task_prompt", AsyncMock(side_effect=lambda run, t, *a, **k: "go")
    )
    fake_config = MagicMock()
    fake_config.load.return_value = SimpleNamespace(agent=SimpleNamespace(provider="acp"))
    monkeypatch.setattr(task_executor, "KiroCrewConfig", fake_config)
    rejected: list = []

    class _Client:
        capabilities = capabilities_for(backend)
        cwd = "/w"

        def context_usage_pct(self):
            return 0.0

        async def reject_tool(self, request_id):
            rejected.append(request_id)

        async def approve_tool(self, request_id):
            raise AssertionError("a denied call must not be approved")

        async def stream(self, prompt):
            yield LLMEvent(
                kind=EVENT_PERMISSION_REQUEST, title="Running: ReadFile", request_id="r-1"
            )
            yield LLMEvent(kind=EVENT_TOOL_CALL, title="Running: ReadFile")
            yield LLMEvent(
                kind=EVENT_COMPLETE,
                stop_reason=STOP_REASON_END_TURN,
                usage=TurnUsage(duration_ms=0),
            )

    run = Project(spec_path="spec.md", spec_content="body")
    run.task_id = "task-spec-hooks"
    task = Task(index=1, title="t", description="d")
    run.tasks = [task]
    run.branch_name = ""
    run.work_dir = ""
    sessions = MagicMock()
    sessions.open_task_session = AsyncMock(return_value=(_Client(), True, False))
    sessions.reset = AsyncMock()
    sessions.record_failure = AsyncMock()
    asyncio.run(
        task_executor.execute_task(
            run, task, sessions, None, "worker", None, False, None, "", AsyncMock(), "tk"
        )
    )
    return rejected, _gate_calls(store.fire), informational


def test_a_kas_task_step_spec_hook_denies_the_permission_request(monkeypatch):
    rejected, gate, informational = _task_step(ACP_BACKEND_KAS, monkeypatch)
    assert rejected == ["r-1"]
    assert [h.id for h in gate[0].kwargs["extra_hooks"]] == ["spec:a1:PreToolUse:0"]
    assert len(informational) == 1 and "extra_hooks" not in informational[0]


def test_a_kiro_cli_task_step_fires_no_spec_hooks_from_crew(monkeypatch):
    rejected, gate, informational = _task_step(ACP_BACKEND_KIRO, monkeypatch)
    assert gate == []
    assert all("extra_hooks" not in kwargs for kwargs in informational)
