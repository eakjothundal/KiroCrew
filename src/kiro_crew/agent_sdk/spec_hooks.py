"""Run an agent spec's own ``hooks`` from Crew's turn loop, for a harness that cannot.

kiro-cli reads the spec off disk and runs its ``hooks`` itself. KAS takes the agent
over the wire as ``_meta.kiro.customAgents``, whose schema has no slot for them
(``acp.kas_agents.UNSUPPORTED_SPEC_KEYS``), so on KAS nothing runs them. The backends
in ``ACP_BACKENDS_CREW_FIRES_SPEC_HOOKS`` get them from Crew instead: the turn loop
already fires the five Crew hook events for every backend, and this module turns the
spec's field into :class:`kiro_crew.hooks.ScriptHook` records that ride the same
``ScriptHookStore.fire`` call. So a spec hook meets exactly the gates a Hooks-page
hook meets -- ``capabilities.script_hooks`` and its SEL audit, the sandboxed spawn,
the timeout, the PreToolUse fail-closed rule -- and none that a Hooks-page hook does
not.

Membership is the whole double-fire guard. A non-member's harness runs the field
itself, so the turn loop asks :func:`spec_script_hooks` only when the session's
capabilities say Crew owns the job.

Both spec shapes are read:

* the object form kiro-cli uses and Crew materializes, ``{event: [{command,
  matcher?, timeout_ms?}]}`` on the five camelCase event names;
* the array of KAS hook documents, validated by
  :func:`kiro_crew.agent.normalize_spec_hooks`. ``enabled: false`` is skipped, and
  so is ``confirm: true``, because no prompt can be shown from here; an ``agent``
  action and a trigger with no Crew event are skipped too.

The result is cached by the field's content, so a spec that stays the same costs
one conversion and logs its warnings once, not once per turn.

The turn loop is not the only place a tool runs: a subagent run and a task-runner
step fire the same hook store for their own tool calls. :func:`turn_spec_hooks`
answers for those, keyed on the SUBAGENT's agent and its own provider, so a kiro-cli
subagent under a KAS parent still gets nothing from Crew.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from typing import Any

from kiro_crew.agent_sdk.capabilities import capabilities_of
from kiro_crew.hooks import (
    HOOK_EVENT_AGENT_SPAWN,
    HOOK_EVENT_POST_TOOL_USE,
    HOOK_EVENT_PRE_TOOL_USE,
    HOOK_EVENT_STOP,
    HOOK_EVENT_USER_PROMPT_SUBMIT,
    ScriptHook,
    _normalize_hook_timeout,
)

logger = logging.getLogger(__name__)

#: kiro-cli's object-form event names, onto the events the hook store fires.
_OBJECT_EVENT_TO_HOOK_EVENT = {
    "agentSpawn": HOOK_EVENT_AGENT_SPAWN,
    "userPromptSubmit": HOOK_EVENT_USER_PROMPT_SUBMIT,
    "preToolUse": HOOK_EVENT_PRE_TOOL_USE,
    "postToolUse": HOOK_EVENT_POST_TOOL_USE,
    "stop": HOOK_EVENT_STOP,
}

#: The events whose matcher names a tool. kiro-cli reads a matcher on these only,
#: so a matcher on any other event is dropped rather than applied to the context.
_TOOL_EVENTS = frozenset({HOOK_EVENT_PRE_TOOL_USE, HOOK_EVENT_POST_TOOL_USE})

#: Bound on the hooks taken from one spec, the same total the materialized spec is
#: capped at, so a hand-written spec cannot make every turn spawn without limit.
_MAX_SPEC_HOOKS = 20

#: Bound on a retained command, matching the document validator's payload cap.
_MAX_COMMAND_LEN = 4096

#: Bound on the conversion cache. A spec edit adds an entry, so the cache is
#: cleared rather than grown once it reaches this.
_CACHE_MAX = 64

#: Per spec content: the hooks that run, and how many ``confirm: true`` documents
#: were skipped (the session-start notice names the count).
_cache: dict[tuple[str, str], tuple[tuple[ScriptHook, ...], int]] = {}


def _diagnostic(value: object) -> str:
    """A spec-supplied value for a log line: escaped, then redacted."""
    # circular import: agent imports hooks, which this module imports at load time.
    from kiro_crew.agent import _hook_diagnostic

    return _hook_diagnostic(value)


def _matcher_ok(matcher: object) -> bool:
    """The object form's matcher rules: a string, length-capped, safe characters."""
    # circular import: agent imports hooks, which this module imports at load time.
    from kiro_crew.agent import _hook_matcher_ok

    return _hook_matcher_ok(matcher)


def _reject(agent_id: str, event: object, value: object, reason: str) -> None:
    """Warn about, and SEL-audit, a spec hook that will not run.

    The same audit the kiro-cli merge writes for a hook it leaves out
    (``agent._sel_hook_rejected``, which redacts before it truncates), because what
    runs differs from what was authored either way. The conversion is cached by
    content, so an unchanged spec audits once, not once per turn.
    """
    # circular import: agent imports hooks, which this module imports at load time.
    from kiro_crew.agent import _sel_hook_rejected

    logger.warning(
        "agent %r: spec hook on %s does not run on this backend: %s",
        agent_id,
        _diagnostic(event),
        reason,
    )
    _sel_hook_rejected(str(event), str(value), f"spec hook for {agent_id}: {reason}")


def _hook(agent_id: str, event: str, index: int, entry: dict, timeout: int) -> ScriptHook | None:
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip() or len(command) > _MAX_COMMAND_LEN:
        _reject(agent_id, event, command, "no usable command")
        return None
    matcher = entry.get("matcher") if event in _TOOL_EVENTS else None
    if matcher is not None and not _matcher_ok(matcher):
        # Dropped whole, as the materialized spec's merge drops it: running the
        # hook with no matcher would widen it to every tool.
        _reject(agent_id, event, command, "invalid matcher")
        return None
    return ScriptHook(
        id=f"spec:{agent_id}:{event}:{index}",
        name=f"{agent_id} spec hook ({event} #{index + 1})",
        event=event,
        matcher=matcher if isinstance(matcher, str) else "",
        command=command,
        timeout=timeout,
    )


def _from_object_form(agent_id: str, hooks: dict) -> list[ScriptHook]:
    out: list[ScriptHook] = []
    for key, entries in hooks.items():
        if len(out) > _MAX_SPEC_HOOKS:
            break
        event = _OBJECT_EVENT_TO_HOOK_EVENT.get(key) if isinstance(key, str) else None
        if event is None or not isinstance(entries, list):
            _reject(agent_id, key, entries, "not a hook event list")
            continue
        for index, entry in enumerate(entries):
            if len(out) > _MAX_SPEC_HOOKS:
                break
            if not isinstance(entry, dict):
                _reject(agent_id, event, entry, "entry is not an object")
                continue
            timeout_ms = entry.get("timeout_ms")
            timeout = (
                _normalize_hook_timeout(math.ceil(timeout_ms / 1000))
                if isinstance(timeout_ms, (int, float)) and not isinstance(timeout_ms, bool)
                else _normalize_hook_timeout(None)
            )
            hook = _hook(agent_id, event, index, entry, timeout)
            if hook is not None:
                out.append(hook)
    return out


def _from_documents(agent_id: str, hooks: list, unconfirmable: list[int]) -> list[ScriptHook]:
    # circular import: agent imports hooks, which this module imports at load time.
    from kiro_crew.agent import _event_for_hook_trigger, normalize_spec_hooks

    out: list[ScriptHook] = []
    for index, doc in enumerate(normalize_spec_hooks(hooks)):
        if len(out) > _MAX_SPEC_HOOKS:
            break
        trigger = doc.get("trigger")
        raw_action = doc.get("action")
        action: dict = raw_action if isinstance(raw_action, dict) else {}
        command = action.get("command")
        if doc.get("enabled") is False:
            _reject(agent_id, trigger, command, "hook is disabled")
            continue
        if doc.get("confirm") is True:
            unconfirmable.append(index)
            _reject(
                agent_id,
                trigger,
                command,
                "hook asks to be confirmed, which Crew cannot prompt for",
            )
            continue
        event = _OBJECT_EVENT_TO_HOOK_EVENT.get(_event_for_hook_trigger(trigger) or "")
        if event is None or action.get("type") != "command":
            _reject(agent_id, trigger, command, "no Crew hook event can run this trigger or action")
            continue
        # The document's timeout is in seconds, clamped like a Hooks-page hook's.
        hook = _hook(
            agent_id,
            event,
            index,
            {"command": command, "matcher": doc.get("matcher")},
            _normalize_hook_timeout(doc.get("timeout")),
        )
        if hook is not None:
            out.append(hook)
    return out


def spec_script_hooks(agent_id: str, spec: dict[str, Any]) -> list[ScriptHook]:
    """The spec's own ``hooks`` as script hooks the hook store can fire.

    Empty when the spec carries none. Either shape is read (see the module
    docstring); anything else is warned about, audited once, and yields nothing.
    """
    return list(_convert(agent_id, spec)[0])


def _convert(agent_id: str, spec: dict[str, Any]) -> tuple[tuple[ScriptHook, ...], int]:
    value = spec.get("hooks")
    if not value:
        return (), 0
    try:
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
    except (TypeError, ValueError):
        digest = ""
    key = (agent_id, digest)
    if digest and key in _cache:
        return _cache[key]
    unconfirmable: list[int] = []
    if isinstance(value, dict):
        hooks = _from_object_form(agent_id, value)
    elif isinstance(value, list):
        hooks = _from_documents(agent_id, value, unconfirmable)
    else:
        _reject(agent_id, "hooks", value, "hooks is neither an object nor an array")
        hooks = []
    if len(hooks) > _MAX_SPEC_HOOKS:
        _reject(
            agent_id,
            "hooks",
            len(hooks),
            f"more than {_MAX_SPEC_HOOKS} spec hooks, ignoring the remainder",
        )
        hooks = hooks[:_MAX_SPEC_HOOKS]
    result = (tuple(hooks), len(unconfirmable))
    if digest:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = result
    return result


def crew_fired_spec_hooks(agent_id: str) -> tuple[list[ScriptHook], list[str], int]:
    """The active agent spec's hooks as script hooks, the keys nothing carries, and
    how many ``confirm: true`` hooks are skipped.

    Reads the spec the KAS projection reads, through the same reader
    (:func:`kiro_crew.acp.kas_agents.load_agent_spec`). The second and third values
    are for the session-start notice: the spec keys a KAS session runs without, and
    the hooks that wait for a confirmation Crew cannot ask for. Raises when the spec
    cannot be read; the caller fails PreToolUse closed on that.
    """
    # circular import: the ACP layer imports the config loader, which sits below
    # this module; resolved at call time like the other driver seams here.
    from kiro_crew.acp.kas_agents import load_agent_spec, spec_keys_without_carrier
    from kiro_crew.config.paths import kiro_agents_dir

    spec = load_agent_spec(kiro_agents_dir(), agent_id)
    hooks, unconfirmable = _convert(agent_id, spec)
    return list(hooks), spec_keys_without_carrier(spec), unconfirmable


async def turn_spec_hooks(
    provider: object, agent_id: str
) -> tuple[list[ScriptHook], str | None, bool]:
    """The spec hooks a subagent or task-runner turn fires, where they run, and
    whether the spec could not be read.

    *provider* and *agent_id* are the turn's OWN: a subagent runs its own agent on
    its own backend, so a kiro-cli subagent under a KAS parent gets
    ``([], None, False)`` here, as its harness already runs the field. The second
    value is the session's workspace, the ``extra_hooks_cwd`` the hooks run in
    (``None`` for the gateway's own). When the third is true the caller blocks
    every permission request, as the chat turn loop does: a deny hook that was
    never loaded gave no verdict.
    """
    if not agent_id or not capabilities_of(provider).crew_fires_spec_hooks:
        return [], None, False
    cwd = getattr(provider, "cwd", "")
    work_dir = cwd if isinstance(cwd, str) and cwd else None
    try:
        hooks, _lost, _unconfirmable = await asyncio.to_thread(crew_fired_spec_hooks, agent_id)
    except Exception:  # noqa: BLE001 - the caller fails permission requests closed
        logger.warning(
            "agent spec hooks for %r could not be read; tool calls are blocked",
            agent_id,
            exc_info=True,
        )
        return [], work_dir, True
    return hooks, work_dir, False
