"""Remote Crew chaining — a crew reached by riding another crew's hop.

Covers the three things the arrangement adds, each of which is a decision that
has to be made SERVER-SIDE because the request can originate inside an embedded
pane whose code the hub does not control:

* the data model (``via_instance_id`` / ``via_remote_port``), including that a
  registry file written before chaining existed still loads;
* the forward: a chained crew dials its PARENT's host and the parent's loopback
  port, never its own coordinates, which this gateway has no route to;
* the two guards — the depth cap, checked before anything is dialled, and the
  cycle guard, which can only be checked once the hop is open because only the
  far end can say which gateway it is.

The token path is covered from both ends: the hub asks the parent to mint
(``_mint_through_parent``), and the parent mints for the hub without disturbing
its own stored credential (``mint_embed_token``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kiro_crew.instances.registry import (
    MAX_CHAINED_PER_PARENT,
    MAX_VIA_HOPS,
    Instance,
    InstancesRegistry,
    InvalidInstanceError,
    ancestor_ids,
    descendant_ids,
)

# ── helpers ──────────────────────────────────────────────────────────────

# Stands in for "this crew's id on its PARENT" in fixtures that do not care what
# it is. The one test that does care asserts on its own genuinely divergent pair,
# because ids that coincide on both sides hide a mint aimed at the wrong one.
VIA_ID = "peer-id"


class _FakeTunnel:
    """Minimal stand-in for ``_SshTunnel``, recording what it was asked to forward."""

    made: list[_FakeTunnel] = []

    def __init__(
        self,
        iid,
        ssh_host,
        lp,
        rp,
        *,
        connect_timeout_secs=0,
        compression=True,
        probe_failure_threshold=0,
        on_exit=None,
        transport="ssh",
        ssm_target="",
        aws_profile="",
        aws_region="",
    ):
        from kiro_crew.instances.ssh_tunnel_manager import TunnelState, TunnelStatus

        self.iid = iid
        self.ssh_host = ssh_host
        self.local_port = lp
        self.remote_port = rp
        self.transport = transport
        self.pid = None
        self.stopped = False
        self.start_result = True
        self._S = TunnelState
        self.status = TunnelStatus(instance_id=iid, local_port=lp, remote_port=rp)
        _FakeTunnel.made.append(self)

    async def start(self):
        self.status.state = self._S.CONNECTED if self.start_result else self._S.ERROR
        if not self.start_result:
            self.status.error = "boom"
        return self.start_result

    async def stop(self):
        self.stopped = True
        self.status.state = self._S.STOPPED


def _free_ports(monkeypatch):
    """Make both loopback port probes answer "free", so these stay hermetic."""
    from kiro_crew.instances import port_allocator
    from kiro_crew.instances import ssh_tunnel_manager as stm

    monkeypatch.setattr(port_allocator, "_is_port_free", lambda *_a, **_k: True)
    monkeypatch.setattr(stm, "_is_port_free", lambda *_a, **_k: True)


def _mgr(tmp_path, monkeypatch, *, mint=None):
    """A manager over a fresh registry, with a fake forwarder and a fake mint."""
    _free_ports(monkeypatch)
    from kiro_crew.instances.ssh_tunnel_manager import SshTunnelManager

    _FakeTunnel.made = []
    reg = InstancesRegistry(path=tmp_path / "instances.json")

    async def ok_mint(
        host,
        *,
        remote_bin="",
        ttl="20h",
        remote_port=None,
        embed_parent_port=None,
        timeout_secs=None,
    ):
        return f"SSH_TOKEN_FOR_{host}_epp{embed_parent_port}"

    return reg, SshTunnelManager(
        reg,
        base_port=53700,
        mint_token=mint or ok_mint,
        tunnel_factory=_FakeTunnel,
        parent_port=4242,
    )


class _State:
    owner_id = "owner"

    def __init__(self, registry, manager=None):
        self.instances_registry = registry
        self.instances_manager = manager


class _FakeReq:
    def __init__(self, state, *, match=None, body=None, query=None, user="owner"):
        self.app = {"state": state}
        self.headers = {}
        self.match_info = match or {}
        self.query = query or {}
        self._body = body
        self._attrs = {"app": ""}
        if user is not None:
            self._attrs["user"] = user

    def get(self, key, default=None):
        return self._attrs.get(key, default)

    def __contains__(self, key):
        return key in self._attrs

    def __getitem__(self, key):
        return self._attrs[key]

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


def _enable(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"instances": {"enabled": True}}))
    from kiro_crew.config import loader

    loader._invalidate_config_cache()


def _resp_body(resp):
    return json.loads(resp.body.decode())


def _relay_mint(mgr, *, token="CHAINED_TOKEN", port=None, seen=None):
    """A stand-in for the chained mint that also reports the hop port.

    The real method reads `port` out of the parent's authenticated reply and
    records it, and the connect path refuses to dial without it -- the row's own
    copy arrived over a pane's postMessage and is not allowed to choose which
    loopback service on the parent gets forwarded. A double that returns only a
    token therefore describes a DIFFERENT contract, not a simpler one.

    Defaults to the row's port, which is what a parent in agreement with the row
    would answer; pass *port* to make it disagree.
    """

    async def fake(inst, _params):
        if seen is not None:
            seen.append(inst.id)
        mgr._chained_hop_port[inst.id] = inst.via_remote_port if port is None else port
        return token

    return fake


# ── the data model ───────────────────────────────────────────────────────


class TestChainFields:
    def test_the_pair_travels_together(self):
        """Either half alone reads as "top level" to every consumer while the
        record plainly means something else, so the half-pair is refused rather
        than silently reinterpreted."""
        with pytest.raises(InvalidInstanceError, match="via_remote_port"):
            Instance(id="c", name="C", ssh_host="c-host", via_instance_id="b").validate()
        with pytest.raises(InvalidInstanceError, match="only a chained instance"):
            Instance(id="c", name="C", ssh_host="c-host", via_remote_port=5476).validate()
        with pytest.raises(InvalidInstanceError, match="only a chained instance"):
            Instance(id="c", name="C", ssh_host="c-host", via_remote_id=VIA_ID).validate()
        # A chained record also needs the id its PARENT knows it by: without that
        # the parent cannot be asked to mint, so the row could never connect.
        with pytest.raises(InvalidInstanceError, match="needs via_remote_id"):
            Instance(
                id="c", name="C", ssh_host="c-host", via_instance_id="b", via_remote_port=5476
            ).validate()
        # All three set is the chained record, and it validates.
        Instance(
            id="c",
            name="C",
            ssh_host="c-host",
            via_instance_id="b",
            via_remote_port=5476,
            via_remote_id=VIA_ID,
        ).validate()
        # None set is every record written before chaining existed.
        Instance(id="b", name="B", ssh_host="b-host").validate()

    def test_a_record_cannot_be_reached_through_itself(self):
        with pytest.raises(InvalidInstanceError, match="through itself"):
            Instance(
                id="c",
                name="C",
                ssh_host="c-host",
                via_instance_id="c",
                via_remote_port=5476,
                via_remote_id=VIA_ID,
            ).validate()

    def test_a_malformed_hop_port_is_refused(self):
        for bad in (0, -1, 70000, True):
            with pytest.raises(InvalidInstanceError):
                Instance(
                    id="c",
                    name="C",
                    ssh_host="c-host",
                    via_instance_id="b",
                    via_remote_port=bad,  # type: ignore[arg-type]
                    via_remote_id=VIA_ID,
                ).validate()

    def test_a_parents_side_id_outside_the_grammar_is_refused(self):
        """It ends up in a request path on the parent, so its shape is checked at
        the boundary rather than trusted because it arrived on a record."""
        with pytest.raises(InvalidInstanceError, match="via_remote_id"):
            Instance(
                id="c",
                name="C",
                ssh_host="c-host",
                via_instance_id="b",
                via_remote_port=5476,
                via_remote_id="victim/disconnect?x=",
            ).validate()

    def test_a_registry_file_written_before_chaining_loads_unchanged(self, tmp_path):
        """The feature must not make an existing instances.json unreadable."""
        path = tmp_path / "instances.json"
        path.write_text(
            json.dumps(
                {
                    "instances": [
                        {"id": "b", "name": "B", "ssh_host": "b-host", "remote_port": 5476}
                    ],
                    "last_active_id": "b",
                }
            )
        )
        loaded = InstancesRegistry(path=path).get("b")
        assert loaded is not None
        assert loaded.via_instance_id == ""
        assert loaded.via_remote_port == 0
        # Including the parent-side id, which a file written before chaining
        # cannot carry: the loader defaults it rather than refusing the record.
        assert loaded.via_remote_id == ""
        # And it is still writable: an update must not fail on a field the
        # caller never touched.
        InstancesRegistry(path=path).update("b", was_connected=True)

    def test_the_pair_round_trips_through_the_stored_shape(self, tmp_path):
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        reloaded = InstancesRegistry(path=tmp_path / "instances.json").get("c")
        assert reloaded is not None
        assert (reloaded.via_instance_id, reloaded.via_remote_port) == ("b", 53999)


class TestChainWalks:
    def _chain(self) -> list[Instance]:
        return [
            Instance(id="b", name="B", ssh_host="b-host"),
            Instance(
                id="c",
                name="C",
                ssh_host="c-host",
                via_instance_id="b",
                via_remote_port=1,
                via_remote_id=VIA_ID,
            ),
            Instance(
                id="d",
                name="D",
                ssh_host="d-host",
                via_instance_id="c",
                via_remote_port=2,
                via_remote_id=VIA_ID,
            ),
        ]

    def test_ancestors_are_nearest_first(self):
        assert ancestor_ids(self._chain(), "d") == ["c", "b"]
        assert ancestor_ids(self._chain(), "b") == []

    def test_an_absent_parent_ends_the_walk(self):
        rows = [
            Instance(
                id="c",
                name="C",
                ssh_host="c-host",
                via_instance_id="gone",
                via_remote_port=1,
                via_remote_id=VIA_ID,
            )
        ]
        assert ancestor_ids(rows, "c") == ["gone"]

    def test_a_looped_registry_terminates(self):
        """A hand edit can write a loop. Both walks must end, because the guards
        that refuse the loop are the CALLERS of these functions.

        The exact chain matters, not just its length: the walk stops AT the id
        that repeats, so the caller sees the loop closing. Without the visit
        guard the bounded range still terminates, but it pads the chain with the
        same two ids over and over and the closure is unreadable."""
        rows = [
            Instance(
                id="a",
                name="A",
                ssh_host="a-host",
                via_instance_id="b",
                via_remote_port=1,
                via_remote_id=VIA_ID,
            ),
            Instance(
                id="b",
                name="B",
                ssh_host="b-host",
                via_instance_id="a",
                via_remote_port=2,
                via_remote_id=VIA_ID,
            ),
        ]
        assert ancestor_ids(rows, "a") == ["b", "a"]
        assert ancestor_ids(rows, "b") == ["a", "b"]
        assert descendant_ids(rows, "a") == ["b"]

    def test_descendants_list_parents_before_their_children(self):
        assert descendant_ids(self._chain(), "b") == ["c", "d"]
        assert descendant_ids(self._chain(), "d") == []


# ── the forward ──────────────────────────────────────────────────────────


class TestChainedForward:
    def test_it_dials_the_parents_host_and_the_hop_port(self, tmp_path, monkeypatch):
        """The crew's own ssh_host/remote_port describe it on ITS machine. This
        gateway has no route to them; the whole point of the chain is that the
        parent does."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", remote_port=5476, instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            remote_port=5476,
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )

        async def no_cycle(_inst, _port):
            return ""

        monkeypatch.setattr(mgr, "_chain_cycle_reason", no_cycle)

        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        st = asyncio.run(mgr.connect("c"))

        assert st.state.value == "connected"
        built = [t for t in _FakeTunnel.made if t.iid == "c"]
        assert len(built) == 1
        assert built[0].ssh_host == "b-host", "dialled the crew instead of its parent"
        assert built[0].remote_port == 53999, "forwarded to the crew's own port, not the hop"

    def test_the_hop_port_comes_from_the_parent_not_from_the_row(self, tmp_path, monkeypatch):
        """The row's copy of the hop port arrived over a pane's postMessage, and it
        is what `ssh -L` aims at on the parent's machine. Unchecked, a forged notice
        chooses which loopback-only service there this gateway forwards and then
        renders as a crew. The parent's own mint reply names the port its forward
        listens on, over the credential we already hold for it."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", remote_port=5476, instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            remote_port=5476,
            instance_id="c",
            via_instance_id="b",
            # What a forged notice put on the row: the parent's own loopback
            # postgres, say, rather than the port its forward to C listens on.
            via_remote_port=5432,
            via_remote_id=VIA_ID,
        )

        async def no_cycle(_inst, _port):
            return ""

        monkeypatch.setattr(mgr, "_chain_cycle_reason", no_cycle)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr, port=53999))
        st = asyncio.run(mgr.connect("c"))

        assert st.state.value == "connected"
        built = [t for t in _FakeTunnel.made if t.iid == "c"]
        assert len(built) == 1
        assert built[0].remote_port == 53999, (
            f"dialled {built[0].remote_port}, the value on the row, so a forged notice "
            f"still picks the service"
        )
        assert 5432 not in [t.remote_port for t in built], "dialled the pane's port at all"
        # Persisted, so a reconnect and the rebuild path dial the same port.
        assert reg.get("c").via_remote_port == 53999

    def test_a_parent_that_names_no_hop_port_is_refused_before_dialling(
        self, tmp_path, monkeypatch
    ):
        """Falling back to the row's value here is the whole hazard, so there is no
        fallback. Every build that answers this endpoint at all reports the port."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=5432,
            via_remote_id=VIA_ID,
        )

        async def silent_mint(_inst, _params):
            return "CHAINED_TOKEN"  # reports no port

        monkeypatch.setattr(mgr, "_mint_through_parent", silent_mint)
        before = len(_FakeTunnel.made)
        st = asyncio.run(mgr.connect("c"))

        assert st.state.value == "error"
        assert "port" in (st.error or ""), st.error
        assert len(_FakeTunnel.made) == before, "opened a forward before the parent named the port"

    def test_a_missing_parent_is_an_error_status_naming_it(self, tmp_path, monkeypatch):
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="gone",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        st = asyncio.run(mgr.connect("c"))
        assert st.state.value == "error"
        assert "gone" in st.error
        assert not _FakeTunnel.made, "opened a forward with no hop to ride"

    def test_an_ssm_parent_is_refused(self, tmp_path, monkeypatch):
        """ssm as the parent hop is a follow-up: its forwarder takes no second
        local forward from this gateway."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(
            name="B",
            connection_method="ssm",
            ssm_target="i-0123456789abcdef0",
            instance_id="b",
        )
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        st = asyncio.run(mgr.connect("c"))
        assert st.state.value == "error"
        assert "ssh hop" in st.error
        assert not _FakeTunnel.made

    def test_the_orphan_reclaim_argv_uses_the_hop_port(self, tmp_path, monkeypatch):
        """The reclaim compares the argv it EXPECTS against the leaked child's
        real one. Built from the crew's own port it could never match, and a
        mismatch reads as "not our child" — the forwarder would keep the port."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            remote_port=5476,
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        params = mgr._resolve_chained_transport(inst, reg.get("b"))
        assert params.forward_remote_port(inst.remote_port) == 53999
        assert params.ssh_host == "b-host"

    def test_a_chained_crew_cannot_be_restarted_from_here(self, tmp_path, monkeypatch):
        """`kirocrew restart` would be dispatched at the PARENT's shell, which
        restarts the wrong machine."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        out = asyncio.run(mgr.restart_remote("c"))
        assert out["ok"] is False
        assert "cannot run commands" in out["message"]


# ── the token ────────────────────────────────────────────────────────────


class TestChainedToken:
    def test_a_chained_mint_goes_through_the_parent_not_over_ssh(self, tmp_path, monkeypatch):
        """This gateway holds no key for a chained crew, so an ssh mint aimed at
        it cannot work — and aimed at the parent it would mint the PARENT's
        token, which the crew's own CSP would reject as a frame ancestor."""
        ssh_mints: list[str] = []

        async def recording_mint(host, **_kw):
            ssh_mints.append(host)
            return "SSH_TOKEN"

        reg, mgr = _mgr(tmp_path, monkeypatch, mint=recording_mint)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        relayed: list[str] = []

        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr, seen=relayed))
        params = mgr._resolve_chained_transport(inst, reg.get("b"))
        token = asyncio.run(mgr._mint_for(inst, params))

        assert token == "CHAINED_TOKEN"
        assert relayed == ["c"]
        assert ssh_mints == [], "dispatched an ssh mint for a crew it has no key for"

    def test_the_parent_mints_with_the_hubs_port_and_keeps_its_own_token(
        self, tmp_path, monkeypatch
    ):
        """The token the parent hands back is scoped to the HUB's page. Storing it
        would replace the parent's own credential for that crew and break the
        parent's pane for it."""
        seen_ports: list[int | None] = []

        async def recording_mint(host, *, embed_parent_port=None, **_kw):
            seen_ports.append(embed_parent_port)
            return f"TOKEN_epp{embed_parent_port}"

        reg, mgr = _mgr(tmp_path, monkeypatch, mint=recording_mint)
        reg.add(name="C", ssh_host="c-host", instance_id="c")
        asyncio.run(mgr.connect("c"))
        own = mgr.get_token("c")
        assert own == "TOKEN_epp4242", "our own mint did not carry our parent port"

        ok, payload = asyncio.run(mgr.mint_embed_token("c", 9191))
        assert ok is True
        assert payload["token"] == "TOKEN_epp9191"
        assert payload["port"] == mgr.status("c").local_port, "named a hop it did not verify"
        assert seen_ports == [4242, 9191]
        assert mgr.get_token("c") == own, "overwrote our own credential for the crew"

    def test_the_parent_refuses_to_mint_for_a_crew_it_reaches_through_a_hop(
        self, tmp_path, monkeypatch
    ):
        """The depth cap seen from the far end, and the ONLY place it can be seen:
        the asking hub counts hops in its own registry and cannot know that ours
        adds another one."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        ok, payload = asyncio.run(mgr.mint_embed_token("c", 9191))
        assert ok is False
        assert payload["code"] == "chain_too_deep"
        assert payload["status"] == 400

    def test_the_parent_refuses_to_mint_for_a_crew_that_is_not_connected(
        self, tmp_path, monkeypatch
    ):
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="C", ssh_host="c-host", instance_id="c")
        ok, payload = asyncio.run(mgr.mint_embed_token("c", 9191))
        assert ok is False
        assert payload["code"] == "instance_not_connected"

    def test_a_hop_freed_during_the_mint_is_refused_not_handed_to_its_new_owner(
        self, tmp_path, monkeypatch
    ):
        """The token and the port are one claim. `status()` hands out the tunnel's
        LIVE status object and a teardown pops that tunnel without zeroing the port
        on it, while the allocator gives a just-freed port to the next connect
        first -- so a crew disconnected inside the mint's round trip, plus any crew
        connected before it returns, would pair C's token with D's forward and the
        asking hub would forward to whatever now answers there.
        """
        holder: list = []

        async def mint_that_loses_the_hop(host, *, embed_parent_port=None, **_kw):
            mgr_ = holder[0]
            # Only the EMBED mint, which runs with no manager lock held. The
            # connect-time mint (embed_parent_port 4242) runs inside `connect`'s
            # own critical section, so re-entering the manager there would park
            # this test on the lock rather than exercise anything.
            if host == "c-host" and embed_parent_port == 9191:
                await mgr_.disconnect("c")
                await mgr_.connect("d")
            return f"TOKEN_FOR_{host}"

        reg, mgr = _mgr(tmp_path, monkeypatch, mint=mint_that_loses_the_hop)
        holder.append(mgr)
        reg.add(name="C", ssh_host="c-host", instance_id="c")
        reg.add(name="D", ssh_host="d-host", instance_id="d")
        assert asyncio.run(mgr.connect("c")).state.value == "connected"
        hop = mgr.status("c").local_port
        assert hop > 0

        ok, payload = asyncio.run(mgr.mint_embed_token("c", 9191))

        assert ok is False, "answered with a token paired to a hop it no longer owns"
        assert payload["code"] == "instance_hop_changed"
        assert payload["status"] == 409
        # The port a stale read would have named belongs to the other crew now,
        # which is what makes this a credential crossing rather than a dead port.
        assert mgr.status("d").local_port == hop

    def test_a_reconnect_during_the_mint_is_refused_although_the_crew_is_live_again(
        self, tmp_path, monkeypatch
    ):
        """Membership cannot see this: the crew is back in `_tunnels` and CONNECTED
        by the time the mint returns. Only the generation stamp says the forward the
        token was minted against is not the forward we would name.
        """
        holder: list = []

        async def mint_that_reconnects(host, *, embed_parent_port=None, **_kw):
            mgr_ = holder[0]
            if host == "c-host" and embed_parent_port == 9191:
                await mgr_.disconnect("c")
                await mgr_.connect("c")
            return f"TOKEN_FOR_{host}"

        reg, mgr = _mgr(tmp_path, monkeypatch, mint=mint_that_reconnects)
        holder.append(mgr)
        reg.add(name="C", ssh_host="c-host", instance_id="c")
        assert asyncio.run(mgr.connect("c")).state.value == "connected"

        ok, payload = asyncio.run(mgr.mint_embed_token("c", 9191))

        assert ok is False, "a reconnect satisfied membership and the mint was answered"
        assert payload["code"] == "instance_hop_changed"
        assert mgr.status("c").state.value == "connected", "the crew really is live again"

    def test_a_forward_that_died_during_the_mint_is_refused(self, tmp_path, monkeypatch):
        """The third reading, and the one a teardown never reaches: a probe marks the
        live status ERROR without popping the tunnel or moving the generation, so
        membership and the stamp both still agree. The hop is dead all the same, and
        answering would hand the asking hub a token plus a port nothing listens on.
        """
        holder: list = []

        async def mint_that_loses_the_forward(host, *, embed_parent_port=None, **_kw):
            mgr_ = holder[0]
            if host == "c-host" and embed_parent_port == 9191:
                from kiro_crew.instances.ssh_tunnel_manager import TunnelState

                mgr_._tunnels["c"].status.state = TunnelState.ERROR
            return f"TOKEN_FOR_{host}"

        reg, mgr = _mgr(tmp_path, monkeypatch, mint=mint_that_loses_the_forward)
        holder.append(mgr)
        reg.add(name="C", ssh_host="c-host", instance_id="c")
        assert asyncio.run(mgr.connect("c")).state.value == "connected"
        epoch_before = mgr._tunnel_epoch.get("c", 0)

        ok, payload = asyncio.run(mgr.mint_embed_token("c", 9191))

        assert ok is False, "answered with a token for a forward that had died"
        assert payload["code"] == "instance_hop_changed"
        assert "c" in mgr._tunnels, "the tunnel was never popped, so membership still agrees"
        assert mgr._tunnel_epoch.get("c", 0) == epoch_before, "the generation never moved"

    def test_an_unknown_crew_is_a_404(self, tmp_path, monkeypatch):
        _reg, mgr = _mgr(tmp_path, monkeypatch)
        ok, payload = asyncio.run(mgr.mint_embed_token("nope", 9191))
        assert ok is False
        assert payload["status"] == 404

    # ── the cycle guard ──────────────────────────────────────────────────────

    def test_the_mint_asks_the_parent_by_the_parents_own_id_not_ours(self, tmp_path, monkeypatch):
        """The two ids genuinely DIFFER here, which is the case a harness using
        matching names cannot see: our row is `c` (derived from the name) while the
        parent knows the crew as `c-2`. The parent looks it up by its own id, so
        asking with ours makes it answer 404 for a crew it holds.

        Also the TTL: the parent issues the token under ITS record for the crew, so
        a refresh scheduled from our row's 20h default would run after a shorter
        token has already expired.
        """
        from kiro_crew.instances import ssh_tunnel_manager as stm

        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id="c-2",
        )
        assert inst.id == "c" and inst.via_remote_id == "c-2", "the ids must differ here"

        _FakeMintSession.posted = []
        monkeypatch.setattr(stm.aiohttp, "ClientSession", _FakeMintSession)
        monkeypatch.setattr(
            mgr, "_peer_target", lambda _pid, path: (f"http://127.0.0.1:1{path}", "kc")
        )
        monkeypatch.setattr(mgr, "_peer_cookie_header", lambda _pid, _name: {})

        params = mgr._resolve_chained_transport(inst, reg.get("b"))
        token = asyncio.run(mgr._mint_through_parent(inst, params))

        assert token == "CHILD_TOKEN"
        assert len(_FakeMintSession.posted) == 1
        assert "/api/instances/c-2/embed-token" in _FakeMintSession.posted[0]
        assert "/api/instances/c/embed-token" not in _FakeMintSession.posted[0]
        assert mgr._chained_ttl["c"] == "2h", "kept our own TTL over the one the parent issued"
        assert mgr._minted_ttl(inst) == "2h"

    def test_the_real_mint_records_the_hop_port_the_parent_reports(self, tmp_path, monkeypatch):
        """Read at THIS level because every connect test doubles the mint, so the
        reply-parsing itself is only reachable here -- a mutation removing it stays
        green against those tests."""
        from kiro_crew.instances import ssh_tunnel_manager as stm

        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=5432,
            via_remote_id="c-2",
        )

        _FakeMintSession.posted = []
        _FakeMintSession.reply = {"token": "CHILD_TOKEN", "port": 4242, "ttl": "2h"}
        monkeypatch.setattr(stm.aiohttp, "ClientSession", _FakeMintSession)
        monkeypatch.setattr(
            mgr, "_peer_target", lambda _pid, path: (f"http://127.0.0.1:1{path}", "kc")
        )
        monkeypatch.setattr(mgr, "_peer_cookie_header", lambda _pid, _name: {})

        params = mgr._resolve_chained_transport(inst, reg.get("b"))
        asyncio.run(mgr._mint_through_parent(inst, params))
        assert mgr._chained_hop_port["c"] == 4242, "took the row's port over the parent's"

    @pytest.mark.parametrize("bad", [None, 0, 65536, -1, True, "53999", 1.5])
    def test_an_unusable_reported_port_is_dropped_not_adopted(self, tmp_path, monkeypatch, bad):
        """Dropped, so the caller refuses the dial. Adopting a malformed value would
        put it straight into an `ssh -L` target, and `True` is in the list because
        `isinstance(True, int)` is True and it would otherwise read as port 1."""
        from kiro_crew.instances import ssh_tunnel_manager as stm

        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id="c-2",
        )

        reply = {"token": "CHILD_TOKEN", "ttl": "2h"}
        if bad is not None:
            reply["port"] = bad
        _FakeMintSession.posted = []
        _FakeMintSession.reply = reply
        monkeypatch.setattr(stm.aiohttp, "ClientSession", _FakeMintSession)
        monkeypatch.setattr(
            mgr, "_peer_target", lambda _pid, path: (f"http://127.0.0.1:1{path}", "kc")
        )
        monkeypatch.setattr(mgr, "_peer_cookie_header", lambda _pid, _name: {})

        params = mgr._resolve_chained_transport(inst, reg.get("b"))
        try:
            asyncio.run(mgr._mint_through_parent(inst, params))
        finally:
            _FakeMintSession.reply = {"token": "CHILD_TOKEN", "port": 4242, "ttl": "2h"}
        assert "c" not in mgr._chained_hop_port, f"adopted {bad!r} as a hop port"

    @pytest.mark.parametrize("bad", ["not-a-ttl", "", None, 7200, True, "20"])
    def test_a_mint_reply_that_cannot_state_the_lifetime_is_refused(
        self, tmp_path, monkeypatch, bad
    ):
        """The parent issued this token, so it is the only party that knows when it
        dies. Our row's TTL is a separate number with its own default and nothing
        holds it below the parent's, so falling back to it can schedule the refresh
        AFTER the token is already dead. Refused instead, the same as a reply we
        cannot read -- every gateway that answers this endpoint reports its TTL.
        """
        from kiro_crew.instances import ssh_tunnel_manager as stm
        from kiro_crew.instances.ssh_tunnel_manager import TokenMintError

        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            ttl="9h",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id="c-2",
        )

        reply = {"token": "T", "port": 1}
        if bad is not None:
            reply["ttl"] = bad
        _FakeMintSession.posted = []
        _FakeMintSession.reply = reply
        try:
            monkeypatch.setattr(stm.aiohttp, "ClientSession", _FakeMintSession)
            monkeypatch.setattr(
                mgr, "_peer_target", lambda _pid, path: (f"http://127.0.0.1:1{path}", "kc")
            )
            monkeypatch.setattr(mgr, "_peer_cookie_header", lambda _pid, _name: {})
            params = mgr._resolve_chained_transport(inst, reg.get("b"))
            with pytest.raises(TokenMintError):
                asyncio.run(mgr._mint_through_parent(inst, params))
        finally:
            _FakeMintSession.reply = {"token": "CHILD_TOKEN", "port": 4242, "ttl": "2h"}

        assert "c" not in mgr._chained_ttl, f"recorded {bad!r} as a token lifetime"

    def test_the_stored_lifetime_is_the_shorter_of_the_two(self, tmp_path, monkeypatch):
        """Both directions, because the safe one is not symmetric. Shorter than the
        token's real life costs an early re-mint; longer schedules the refresh after
        it is dead. Taking the minimum makes that hold however the two are
        configured, rather than assuming the parent's is always the smaller.
        """
        reg, _mgr_unused = _mgr(tmp_path, monkeypatch)
        _reg2, mgr = _mgr(tmp_path / "second", monkeypatch)
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            ttl="9h",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id="c-2",
        )

        mgr._chained_ttl["c"] = "2h"
        assert mgr._minted_ttl(inst) == "2h", "kept our longer TTL over the parent's shorter one"
        mgr._chained_ttl["c"] = "40h"
        assert mgr._minted_ttl(inst) == "9h", "took a TTL longer than ours, scheduling late"

        top = reg.add(name="D", ssh_host="d-host", instance_id="d", ttl="5h")
        mgr._chained_ttl["d"] = "1h"
        assert mgr._minted_ttl(top) == "5h", "a top-level token is ours, not a parent's"

    def test_a_stored_id_cannot_inject_a_parent_control_plane_path(self, tmp_path, monkeypatch):
        """`Instance.from_dict` is deliberately tolerant, so a registry file written
        by hand or by an agent can carry any id string. Unchecked, an id like
        `victim/disconnect?x=` interpolates into a DIFFERENT authenticated route on
        the parent and spends our credential for it there."""
        from kiro_crew.instances.registry import Instance
        from kiro_crew.instances.ssh_tunnel_manager import TokenMintError

        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        valid = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )
        params = mgr._resolve_chained_transport(valid, reg.get("b"))

        hostile = Instance.from_dict(
            {
                "id": "c",
                "name": "C",
                "ssh_host": "c-host",
                "via_instance_id": "b",
                "via_remote_port": 53999,
                "via_remote_id": "victim/disconnect?x=",
            }
        )
        assert (
            hostile.via_remote_id == "victim/disconnect?x="
        ), "from_dict rejected it, so the risk is elsewhere"

        dialled: list[str] = []

        def record_target(*_a, **_k):
            dialled.append("built a request")
            return "http://127.0.0.1:1/x", "cookie"

        monkeypatch.setattr(mgr, "_peer_target", record_target)
        with pytest.raises(TokenMintError):
            asyncio.run(mgr._mint_through_parent(hostile, params))
        assert dialled == [], "built a parent request for an id outside the grammar"

    def test_a_parent_remint_completes_with_the_lock_already_held(self, tmp_path, monkeypatch):
        """The chained mint runs inside `connect`'s lock hold, so the re-mint it
        reaches for on a rejected parent credential has to work THERE. Bounded by
        `wait_for` so a re-entrant acquire fails the test instead of hanging it."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")

        async def scenario():
            await mgr.connect("b")
            async with mgr._lock:
                return await asyncio.wait_for(mgr._remint_parent_under_lock("b"), timeout=5)

        assert asyncio.run(scenario()) is True
        assert mgr.get_token("b"), "re-minted nothing for the parent"

    def test_the_rejected_credential_path_never_re_enters_the_manager_lock(self):
        """`asyncio.Lock` is not reentrant and the holder here is the same task, so
        a 401 reaching the lock-taking public refresh would hang the connect while
        it still holds the lock, wedging every later connect and disconnect."""
        import ast
        import inspect
        import textwrap

        from kiro_crew.instances.ssh_tunnel_manager import SshTunnelManager

        def calls_in(fn):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            return {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}

        relay_calls = calls_in(SshTunnelManager._mint_through_parent)
        assert "self.refresh_token" not in relay_calls, "re-enters the lock via the public refresh"
        assert "self._remint_parent_under_lock" in relay_calls

        helper = textwrap.dedent(inspect.getsource(SshTunnelManager._remint_parent_under_lock))
        acquired = [
            ast.unparse(item.context_expr)
            for node in ast.walk(ast.parse(helper))
            if isinstance(node, ast.AsyncWith)
            for item in node.items
        ]
        assert (
            "self._lock" not in acquired
        ), "the caller already holds it; taking it again deadlocks"


class TestCycleGuard:
    def _chained(self, tmp_path, monkeypatch):
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )

        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        return reg, mgr

    def test_a_hop_that_lands_back_on_us_is_refused_and_torn_down(self, tmp_path, monkeypatch):
        from kiro_crew.instances import ssh_tunnel_manager as stm

        reg, mgr = self._chained(tmp_path, monkeypatch)
        monkeypatch.setattr(stm, "gateway_id", lambda *_a, **_k: "OURS")

        async def far_end_is_us(_port):
            return "OURS"

        monkeypatch.setattr(mgr, "_peer_gateway_id", far_end_is_us)
        st = asyncio.run(mgr.connect("c"))

        assert st.state.value == "error"
        assert "loop" in st.error
        built = [t for t in _FakeTunnel.made if t.iid == "c"]
        assert built and built[0].stopped, "left the looping forward open"
        assert mgr.status("c") is None
        assert reg is not None

    def test_a_hop_that_lands_on_an_ancestor_is_refused(self, tmp_path, monkeypatch):
        from kiro_crew.instances import ssh_tunnel_manager as stm

        _reg, mgr = self._chained(tmp_path, monkeypatch)
        monkeypatch.setattr(stm, "gateway_id", lambda *_a, **_k: "OURS")
        # Bring the parent up so the guard has an ancestor tunnel to read.
        asyncio.run(mgr.connect("b"))
        parent_port = mgr.status("b").local_port

        async def ports(port):
            # The parent and the far end of the new hop are the same gateway.
            return "PARENT" if port in (parent_port, mgr.status("c").local_port) else "OTHER"

        monkeypatch.setattr(mgr, "_peer_gateway_id", ports)
        st = asyncio.run(mgr.connect("c"))

        assert st.state.value == "error"
        assert "'b'" in st.error and "loop" in st.error

    def test_a_crew_that_reports_no_id_is_allowed(self, tmp_path, monkeypatch):
        """Fail-open on a crew older than the field: the loop it cannot rule out
        is a nested pane, not an escape from a boundary, and the depth cap
        already bounds the arrangement."""
        _reg, mgr = self._chained(tmp_path, monkeypatch)

        async def silent(_port):
            return ""

        monkeypatch.setattr(mgr, "_peer_gateway_id", silent)
        st = asyncio.run(mgr.connect("c"))
        assert st.state.value == "connected"

    def test_a_top_level_crew_is_not_cycle_checked(self, tmp_path, monkeypatch):
        """A top-level crew pointing back at this gateway is what the product
        already allows. Refusing it here would break a working setup over an
        arrangement chaining does not introduce."""
        from kiro_crew.instances import ssh_tunnel_manager as stm

        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="Self", ssh_host="localhost", instance_id="self")
        monkeypatch.setattr(stm, "gateway_id", lambda *_a, **_k: "OURS")
        probed: list[int] = []

        async def far_end_is_us(port):
            probed.append(port)
            return "OURS"

        monkeypatch.setattr(mgr, "_peer_gateway_id", far_end_is_us)
        st = asyncio.run(mgr.connect("self"))
        assert st.state.value == "connected"
        assert probed == [], "probed a crew that rides no hop"


# ── the cascade ──────────────────────────────────────────────────────────


class _FakeMintReader:
    def __init__(self, raw: bytes):
        self._raw = raw

    async def read(self, _n: int) -> bytes:
        return self._raw


class _FakeMintResp:
    def __init__(self, status: int, payload: dict):
        self.status = status
        self.content = _FakeMintReader(json.dumps(payload).encode("utf-8"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakeMintSession:
    """Records the URL each chained mint is aimed at and answers with a canned reply.

    The URL is what the assertion is really about: the path carries the crew's id
    as the PARENT knows it, and a fake that forgets the URL cannot tell the right
    id from the wrong one.
    """

    posted: list[str] = []
    reply: dict = {"token": "CHILD_TOKEN", "port": 4242, "ttl": "2h"}

    def __init__(self, *_a, **_kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def post(self, url, **_kw):
        _FakeMintSession.posted.append(url)
        return _FakeMintResp(200, _FakeMintSession.reply)


class _FakeMgr:
    """Records every disconnect, and what the registry still held when it landed.

    The rows are what ``disconnect`` reads to find a crew's children, so a sweep
    that runs after they are gone can only reach the ids its caller captured
    beforehand. Recording the row count alongside each call is what lets a test
    assert that, rather than assert a call order that would also hold by luck.
    """

    def __init__(self, registry=None):
        self._registry = registry
        self.disconnected: list[str] = []
        self.rows_at_disconnect: list[int] = []

    async def disconnect(self, instance_id, **_kw):
        self.disconnected.append(instance_id)
        self.rows_at_disconnect.append(len(self._registry.list()) if self._registry else -1)
        return True

    def status(self, _instance_id):
        # Nothing is connected in these tests; the add handler renders an instance
        # view, and a view asks the manager what each row's tunnel is doing.
        return None


class TestChainCascade:
    def test_disconnecting_a_parent_closes_its_childrens_forwards(self, tmp_path, monkeypatch):
        """A child's forward targets a port on a machine this gateway can no
        longer reach once the parent's hop is gone: a pane that looks connected
        and answers nothing."""
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=53999,
            via_remote_id=VIA_ID,
        )

        async def no_cycle(_inst, _port):
            return ""

        monkeypatch.setattr(mgr, "_chain_cycle_reason", no_cycle)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        asyncio.run(mgr.connect("b"))
        asyncio.run(mgr.connect("c"))
        assert mgr.status("c") is not None

        asyncio.run(mgr.disconnect("b"))

        assert mgr.status("b") is None
        assert mgr.status("c") is None, "left a child riding a hop that is gone"
        # The child keeps its intent: the user turned off the PARENT, so
        # reconnecting the parent must be able to bring its crews back.
        assert reg.get("c").was_connected is True
        assert reg.get("b").was_connected is False

    def test_removing_a_parent_removes_the_rows_below_it(self, tmp_path, monkeypatch):
        """A row left behind describes a forward that can never be opened again."""
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )
        reg.add(
            name="D",
            ssh_host="d-host",
            instance_id="d",
            via_instance_id="c",
            via_remote_port=2,
            via_remote_id=VIA_ID,
        )
        reg.add(name="Other", ssh_host="other-host", instance_id="other")

        resp = asyncio.run(handlers.api_instances_remove(_FakeReq(_State(reg), match={"id": "b"})))
        assert resp.status == 200
        body = _resp_body(resp)
        assert body["removed"] == "b"
        assert sorted(body["removed_chained"]) == ["c", "d"]
        assert [i.id for i in reg.list()] == ["other"]

    def test_removing_an_id_with_no_row_deletes_nothing(self, tmp_path, monkeypatch):
        """A 404 has to be decided BEFORE anything is deleted. Reading it off the
        parent's own removal meant the descendants were already gone by the time
        the missing parent answered "not found" -- so a DELETE of an orphan's
        former parent id reported a failure having just deleted the crews under
        it, and dropped the list of which ones they were with it.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        # `c` and `d` still name `b` as their hop, but b's own row is gone -- the
        # shape an unregister leaves behind, since it removes one row and cascades
        # nothing.
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )
        reg.add(
            name="D",
            ssh_host="d-host",
            instance_id="d",
            via_instance_id="c",
            via_remote_port=2,
            via_remote_id=VIA_ID,
        )

        resp = asyncio.run(handlers.api_instances_remove(_FakeReq(_State(reg), match={"id": "b"})))
        assert resp.status == 404
        assert sorted(i.id for i in reg.list()) == ["c", "d"], "404'd after deleting rows"

    def test_a_child_that_will_not_stop_refuses_the_removal_with_nothing_deleted(
        self, tmp_path, monkeypatch
    ):
        """Rows deleted over a live forwarder strand it together with its minted
        token, and the row that goes takes the `forwarder_pid` reclaim hint with it,
        so nothing can find it again while the gateway runs. An EXCEPTION out of the
        teardown is unambiguous -- unlike a status reading, which cannot tell a
        failed stop from a reconnect that landed after a successful one -- so it is
        safe to refuse on, and every row stays.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id="c-on-b",
        )

        class _StuckChild:
            def __init__(self):
                self.calls: list[str] = []

            async def disconnect(self, instance_id):
                self.calls.append(instance_id)
                if instance_id == "c":
                    raise OSError("terminate refused")
                return True

        mgr = _StuckChild()
        resp = asyncio.run(
            handlers.api_instances_remove(_FakeReq(_State(reg, mgr), match={"id": "b"}))
        )
        assert resp.status == 409
        assert _resp_body(resp)["code"] == "remove_teardown_failed"
        assert sorted(i.id for i in reg.list()) == ["b", "c"], "deleted rows over a live forward"
        # Deepest first, and it stopped AT the failure rather than carrying on to
        # the parent -- the parent's forward is the child's route out.
        assert mgr.calls == ["c"]

    def test_a_forward_that_reappears_after_the_rows_go_is_reported_not_raised(
        self, tmp_path, monkeypatch
    ):
        """The second pass cannot refuse: the rows are already gone, so raising
        would report failure for a removal that happened. The crew it could not stop
        is named in the response instead of being hidden.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")

        class _RacingStuck:
            def __init__(self):
                self.calls = 0

            async def disconnect(self, _instance_id):
                self.calls += 1
                # Down cleanly before the rows go; a reconnect slips in and then
                # refuses to stop on the second pass.
                if self.calls > 1:
                    raise OSError("reappeared and will not stop")
                return True

        resp = asyncio.run(
            handlers.api_instances_remove(_FakeReq(_State(reg, _RacingStuck()), match={"id": "b"}))
        )
        assert resp.status == 200
        body = _resp_body(resp)
        assert body["removed"] == "b"
        assert body["stranded"] == ["b"], "hid a forward it could not stop"
        assert [i.id for i in reg.list()] == [], "kept the row it reported removed"

    def test_the_sweep_reaches_every_captured_crew_once_the_rows_are_gone(
        self, tmp_path, monkeypatch
    ):
        """A connect racing the removal can re-establish a child's forward. The
        sweep afterwards has to name that child itself: `disconnect` finds children
        by reading the registry, and the rows are gone by then, so sweeping only
        the crew that was asked for tears down nothing below it and leaves a
        forward holding a port with no row left to surface it."""
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )
        reg.add(
            name="D",
            ssh_host="d-host",
            instance_id="d",
            via_instance_id="c",
            via_remote_port=2,
            via_remote_id=VIA_ID,
        )

        mgr = _FakeMgr(registry=reg)
        resp = asyncio.run(
            handlers.api_instances_remove(_FakeReq(_State(reg, mgr), match={"id": "b"}))
        )
        assert resp.status == 200
        # Two passes, each deepest-first over the captured crews. The FIRST runs
        # before any row is deleted, so a stop that raises can still refuse with
        # nothing removed; the SECOND runs after, for a forward a racing connect
        # re-established between the teardown and the offloaded deletion.
        assert mgr.disconnected == [
            "d",
            "c",
            "b",
            "d",
            "c",
            "b",
        ], f"swept {mgr.disconnected}, so a racing connect below 'b' survives"
        # The first pass runs WITH the rows present -- that is what lets it refuse
        # before anything is deleted. The second runs with the registry empty, which
        # is exactly why its ids cannot come from it: only the captured list still
        # names 'c' and 'd'.
        assert mgr.rows_at_disconnect[:3] == [3, 3, 3], "tore down after deleting rows"
        assert mgr.rows_at_disconnect[3:] == [0, 0, 0], "swept while rows still existed"

    def test_a_chained_add_cannot_land_inside_a_parent_removal(self, tmp_path, monkeypatch):
        """Validating the parent and inserting the child are two awaits. A removal
        between them snapshots the subtree without the new row and then deletes the
        parent under it, leaving a crew whose only route is a hop that is gone."""
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")

        async def interleave():
            # The add reaches its parent check first; the removal then runs to
            # completion while the add is still between validation and insert --
            # which it can only do if the two are not one critical section.
            adding = asyncio.create_task(
                handlers.api_instances_add(
                    _FakeReq(
                        _State(reg),
                        body={
                            "name": "C",
                            "ssh_host": "c-host",
                            "via_instance_id": "b",
                            "via_remote_port": 53999,
                            "via_remote_id": VIA_ID,
                        },
                    )
                )
            )
            await asyncio.sleep(0)
            removing = asyncio.create_task(
                handlers.api_instances_remove(
                    _FakeReq(_State(reg, _FakeMgr(registry=reg)), match={"id": "b"})
                )
            )
            return await asyncio.gather(adding, removing)

        added, removed = asyncio.run(interleave())
        assert removed.status == 200
        orphans = [i.id for i in reg.list() if i.via_instance_id and not reg.get(i.via_instance_id)]
        assert (
            orphans == []
        ), f"{orphans} name a parent that is gone, so they can never be connected again"
        # Either order is correct. What must not happen is a row surviving its
        # parent: the add either lands before the snapshot and goes with it, or
        # lands after the parent is gone and is refused.
        assert added.status in (201, 400)

    def test_the_rows_go_leaves_first_so_no_row_outlives_its_parent(self, tmp_path, monkeypatch):
        """An interrupted sweep may leave a parent with fewer children, which still
        connects; a row whose parent is already gone can never be opened again."""
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )
        reg.add(
            name="D",
            ssh_host="d-host",
            instance_id="d",
            via_instance_id="c",
            via_remote_port=2,
            via_remote_id=VIA_ID,
        )

        order: list[str] = []
        real_remove = reg.remove

        def recording_remove(target):
            order.append(target)
            return real_remove(target)

        monkeypatch.setattr(reg, "remove", recording_remove)
        resp = asyncio.run(
            handlers.api_instances_remove(_FakeReq(_State(reg, _FakeMgr()), match={"id": "b"}))
        )
        assert resp.status == 200
        assert order == ["d", "c", "b"], f"removed in {order}, so a row can outlive its parent"


# ── the depth cap, at the add boundary ───────────────────────────────────


class TestDepthCap:
    def _reg(self, tmp_path):
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        return reg

    def test_a_second_hop_is_allowed(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = self._reg(tmp_path)
        body = {
            "name": "C",
            "ssh_host": "c-host",
            "id": "c",
            "via_instance_id": "b",
            "via_remote_id": VIA_ID,
            "via_remote_port": 53999,
        }
        resp = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert resp.status == 201
        assert reg.get("c").via_remote_port == 53999

    def test_a_third_hop_is_refused_before_anything_is_written(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = self._reg(tmp_path)
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )
        body = {
            "name": "D",
            "ssh_host": "d-host",
            "id": "d",
            "via_instance_id": "c",
            "via_remote_id": VIA_ID,
            "via_remote_port": 2,
        }
        resp = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert resp.status == 400
        assert _resp_body(resp)["code"] == "chain_too_deep"
        assert reg.get("d") is None, "wrote the record it refused"
        # Bound to the constant, not to the number 2: raising the cap moves this
        # test's own expectation with it, instead of leaving it to assert a limit
        # the product does not have.
        assert MAX_VIA_HOPS == 2, "the refusal above assumes a third hop is one too many"

    def test_an_unknown_parent_is_refused(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = self._reg(tmp_path)
        body = {
            "name": "C",
            "ssh_host": "c-host",
            "id": "c",
            "via_instance_id": "nope",
            "via_remote_id": VIA_ID,
            "via_remote_port": 1,
        }
        resp = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert resp.status == 400
        assert _resp_body(resp)["code"] == "chain_parent_unknown"

    def test_a_second_row_for_the_same_remote_crew_is_refused(self, tmp_path, monkeypatch):
        """One row per remote crew, settled HERE rather than by the announcing pane.
        The pane decides from a list it refreshes only after the add and the connect
        that add triggers, so a second announcement inside that window reads a list
        without the first row and asks for a duplicate -- which the registry would
        take under a suffixed id, giving one remote crew two rows, two forwards, two
        tabs and two charges against the width cap.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")

        body = {
            "name": "C",
            "ssh_host": "c-host",
            "id": "c",
            "via_instance_id": "b",
            "via_remote_id": "c-on-b",
            "via_remote_port": 53999,
        }
        first = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert first.status == 201

        again = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert again.status == 400
        assert _resp_body(again)["code"] == "chain_duplicate"
        assert len([i for i in reg.list() if i.via_remote_id == "c-on-b"]) == 1

    def test_the_uniqueness_guard_is_scoped_to_the_pair(self, tmp_path, monkeypatch):
        """It must not refuse a DIFFERENT crew behind the same parent, nor the same
        remote id behind a different parent -- two parents can each hold a crew whose
        id on that parent happens to read the same.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(name="E", ssh_host="e-host", instance_id="e")

        def add(name, parent, remote_id):
            return asyncio.run(
                handlers.api_instances_add(
                    _FakeReq(
                        _State(reg),
                        body={
                            "name": name,
                            "ssh_host": f"{name.lower()}-host",
                            "via_instance_id": parent,
                            "via_remote_id": remote_id,
                            "via_remote_port": 1,
                        },
                    )
                )
            )

        assert add("C", "b", "shared-id").status == 201
        assert add("D", "b", "other-id").status == 201, "refused a different crew on b"
        assert add("F", "e", "shared-id").status == 201, "refused the same id on another parent"
        assert add("G", "b", "shared-id").status == 400, "admitted the duplicate pair"

    def test_two_racing_notices_for_one_crew_leave_one_row(self, tmp_path, monkeypatch):
        """The check is only decisive INSIDE the lock. Outside it both notices read a
        list without the other's row, both pass, and both add -- the same bug in a new
        place. A retry from the announcing pane produces exactly this pair, so it is
        the ordinary case rather than an error path.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")

        body = {
            "name": "C",
            "ssh_host": "c-host",
            "via_instance_id": "b",
            "via_remote_id": "c-on-b",
            "via_remote_port": 53999,
        }

        async def race():
            return await asyncio.gather(
                handlers.api_instances_add(_FakeReq(_State(reg), body=dict(body))),
                handlers.api_instances_add(_FakeReq(_State(reg), body=dict(body))),
            )

        first, second = asyncio.run(race())
        assert sorted([first.status, second.status]) == [201, 400], "both notices were accepted"
        refused = [r for r in (first, second) if r.status == 400]
        assert [_resp_body(r)["code"] for r in refused] == ["chain_duplicate"]
        assert len([i for i in reg.list() if i.via_remote_id == "c-on-b"]) == 1
        # One row means one forward and one tab: both follow the row, and a second
        # row is the only way one remote crew acquires a second of either.
        assert len([i for i in reg.list() if i.via_instance_id == "b"]) == 1

    def test_the_uniqueness_check_sits_inside_the_mutation_lock(self):
        """Structural, because the behaviour test above cannot see WHY it passed: a
        check-then-add outside the lock is the same defect one step along, and no
        single-threaded assertion distinguishes the two placements.
        """
        import ast
        import inspect

        from kiro_crew.dashboard import handlers_instances as handlers

        tree = ast.parse(inspect.getsource(handlers.api_instances_add))
        fn = tree.body[0]

        def guarded_blocks(node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.AsyncWith) and any(
                    isinstance(item.context_expr, ast.Name)
                    and item.context_expr.id == "_CHAIN_MUTATION_LOCK"
                    for item in sub.items
                ):
                    yield sub

        blocks = list(guarded_blocks(fn))
        assert len(blocks) == 1, f"expected one guarded block, found {len(blocks)}"

        def refusal_calls(node):
            return [
                n
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_chain_refusal"
            ]

        inside = sum(len(refusal_calls(stmt)) for stmt in blocks[0].body)
        total = len(refusal_calls(fn))
        assert inside == 1, f"the refusal is called {inside} times inside the lock, expected 1"
        assert total == inside, f"{total - inside} call(s) sit outside the lock"

    def test_a_parent_already_at_the_width_cap_is_refused(self, tmp_path, monkeypatch):
        """Depth and width are independent bounds. Every crew here is two hops, which
        the depth cap allows without limit, so the population behind one parent needs
        its own cap -- these rows are created by that parent's pane announcing crews,
        not by anyone at this dashboard, and each one is a forward and a mint here.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")

        def _add(child_id: str, parent: str = "b"):
            body = {
                "name": child_id.upper(),
                "ssh_host": f"{child_id}-host",
                "id": child_id,
                "via_instance_id": parent,
                # Its own id on the parent: eight crews behind one hop are eight
                # DIFFERENT crews, so sharing one id here would exercise the
                # uniqueness guard instead of the cap this test is about.
                "via_remote_id": f"{child_id}-on-{parent}",
                "via_remote_port": 1,
            }
            return asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))

        for n in range(MAX_CHAINED_PER_PARENT):
            resp = _add(f"kid{n}")
            assert resp.status == 201, f"refused child {n}, below the cap"

        resp = _add("overflow")
        assert resp.status == 400
        assert _resp_body(resp)["code"] == "chain_parent_full"
        assert reg.get("overflow") is None, "wrote the record it refused"
        # Bound to the constant, so raising the cap moves this expectation with it.
        assert (
            sum(1 for i in reg.list() if i.via_instance_id == "b") == MAX_CHAINED_PER_PARENT
        ), "the refusal must land exactly at the cap, not before or after it"

    def test_the_width_cap_is_counted_per_parent(self, tmp_path, monkeypatch):
        """A full parent must not refuse crews riding a DIFFERENT one, or one parent's
        pane could stop every other parent from being used.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        reg.add(name="E", ssh_host="e-host", instance_id="e")
        for n in range(MAX_CHAINED_PER_PARENT):
            reg.add(
                name=f"KID{n}",
                ssh_host=f"kid{n}-host",
                instance_id=f"kid{n}",
                via_instance_id="b",
                via_remote_port=1,
                via_remote_id=f"kid{n}-on-b",
            )

        body = {
            "name": "F",
            "ssh_host": "f-host",
            "id": "f",
            "via_instance_id": "e",
            "via_remote_id": VIA_ID,
            "via_remote_port": 1,
        }
        resp = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert resp.status == 201, "a full parent blocked a crew riding another parent"

        top = {"name": "G", "ssh_host": "g-host", "id": "g"}
        resp = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=top)))
        assert resp.status == 201, "the chained cap refused a top-level crew"

    def test_the_published_total_is_the_token_s_own_not_the_row_s(self, tmp_path, monkeypatch):
        """`token_ttl_remaining` counts down from the STORED total, so a reader
        measuring how far a token has run must divide by that same number. For a
        chained crew the row's TTL is a different figure -- the parent issues the
        token -- and dividing by the row puts a short token permanently past the
        refresh threshold, re-minting on every poll and remounting the pane.
        """
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", instance_id="b")
        inst = reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            ttl="20h",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )

        # The parent issued an hour; our row says twenty. The stored total is the
        # shorter, so that is the number the status must publish.
        mgr._chained_ttl["c"] = "1h"
        mgr._store_token("c", "TOKEN", mgr._minted_ttl(inst))
        assert mgr.token_ttl_total("c") == 3600, "published our row's TTL, not the token's"

        remaining = mgr.token_ttl_remaining("c")
        assert remaining is not None and remaining <= 3600
        # No token, no total -- absent rather than zero, because a zero total
        # would read as a fully elapsed token and send readers to a refresh.
        assert mgr.token_ttl_total("nope") is None

    def test_the_status_dict_carries_the_total_beside_the_remaining(self, tmp_path, monkeypatch):
        """Both or neither: a consumer that gets a remaining without the total it
        counts down from has to invent a denominator, which is the defect.
        """
        from kiro_crew.dashboard import handlers_instances as handlers

        class _St:
            def to_dict(self):
                return {"instance_id": "c", "state": "connected"}

        class _Mgr:
            def __init__(self, total):
                self._total = total

            def status(self, _iid):
                return _St()

            def token_ttl_remaining(self, _iid):
                return 3500

            def token_ttl_total(self, _iid):
                return self._total

        reg = self._reg(tmp_path)
        d = handlers._status_for(_State(reg, manager=_Mgr(3600)), "c")
        assert d["token_ttl_remaining"] == 3500
        assert d["token_ttl_total"] == 3600

        d2 = handlers._status_for(_State(reg, manager=_Mgr(None)), "c")
        assert d2["token_ttl_remaining"] == 3500
        assert "token_ttl_total" not in d2

    def test_an_ssm_parent_is_refused(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        reg.add(
            name="B",
            connection_method="ssm",
            ssm_target="i-0123456789abcdef0",
            instance_id="b",
        )
        body = {
            "name": "C",
            "ssh_host": "c-host",
            "id": "c",
            "via_instance_id": "b",
            "via_remote_id": VIA_ID,
            "via_remote_port": 1,
        }
        resp = asyncio.run(handlers.api_instances_add(_FakeReq(_State(reg), body=body)))
        assert resp.status == 400
        assert _resp_body(resp)["code"] == "chain_parent_not_ssh"

    def test_a_patch_cannot_re_parent_a_crew(self, tmp_path, monkeypatch):
        """Re-parenting would move a crew onto a hop whose depth was never
        checked, so only the hop PORT is editable."""
        from kiro_crew.dashboard import handlers_instances as handlers

        assert "via_instance_id" not in handlers._PATCH_FIELD_TYPES
        assert handlers._PATCH_FIELD_TYPES["via_remote_port"] is int

        _enable(tmp_path, monkeypatch)
        reg = self._reg(tmp_path)
        reg.add(
            name="C",
            ssh_host="c-host",
            instance_id="c",
            via_instance_id="b",
            via_remote_port=1,
            via_remote_id=VIA_ID,
        )
        resp = asyncio.run(
            handlers.api_instances_update(
                _FakeReq(
                    _State(reg),
                    match={"id": "c"},
                    body={"via_instance_id": "other", "via_remote_port": 2},
                )
            )
        )
        assert resp.status == 200
        updated = reg.get("c")
        assert updated.via_instance_id == "b", "a PATCH re-parented the crew"
        assert updated.via_remote_port == 2


# ── the embed-token endpoint ─────────────────────────────────────────────


class TestEmbedTokenEndpoint:
    class _Mgr:
        def __init__(self, result):
            self.result = result
            self.calls: list[tuple[str, int]] = []

        async def mint_embed_token(self, iid, port):
            self.calls.append((iid, port))
            return self.result

    def test_it_hands_back_the_token_and_the_hop_port(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        mgr = self._Mgr((True, {"token": "TOK", "port": 53999}))
        resp = asyncio.run(
            handlers.api_instances_embed_token(
                _FakeReq(_State(reg, mgr), match={"id": "c"}, body={"embed_parent_port": 9191})
            )
        )
        assert resp.status == 200
        assert _resp_body(resp) == {"token": "TOK", "port": 53999}
        assert mgr.calls == [("c", 9191)]

    def test_a_malformed_port_is_refused_before_any_mint(self, tmp_path, monkeypatch):
        """`isinstance(True, int)` is True, so a bool would otherwise be accepted
        as port 1 and mint a token no page can use."""
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        for bad in (True, 0, -1, 70000, "9191", None):
            mgr = self._Mgr((True, {"token": "TOK", "port": 1}))
            resp = asyncio.run(
                handlers.api_instances_embed_token(
                    _FakeReq(_State(reg, mgr), match={"id": "c"}, body={"embed_parent_port": bad})
                )
            )
            assert resp.status == 400, f"accepted embed_parent_port={bad!r}"
            assert mgr.calls == []

    def test_a_refusal_carries_the_managers_status(self, tmp_path, monkeypatch):
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        mgr = self._Mgr((False, {"error": "too deep", "code": "chain_too_deep", "status": 400}))
        resp = asyncio.run(
            handlers.api_instances_embed_token(
                _FakeReq(_State(reg, mgr), match={"id": "c"}, body={"embed_parent_port": 9191})
            )
        )
        assert resp.status == 400
        body = _resp_body(resp)
        assert body["code"] == "chain_too_deep"
        assert "status" not in body, "leaked the transport hint into the payload"

    def test_a_non_owner_caller_is_refused(self, tmp_path, monkeypatch):
        """The route mints a credential for a machine, so it carries the same
        owner-only gate as every other route in this control plane."""
        from kiro_crew.dashboard import handlers_instances as handlers

        _enable(tmp_path, monkeypatch)
        reg = InstancesRegistry(path=tmp_path / "instances.json")
        mgr = self._Mgr((True, {"token": "TOK", "port": 1}))
        resp = asyncio.run(
            handlers.api_instances_embed_token(
                _FakeReq(
                    _State(reg, mgr),
                    match={"id": "c"},
                    body={"embed_parent_port": 9191},
                    user=None,
                )
            )
        )
        assert resp.status == 401
        assert mgr.calls == []

    def test_the_route_is_registered_ahead_of_the_catch_all_proxy(self):
        """A `{path:.*}` route registered first would swallow this one.

        Every verb is recorded, not just POST: the catch-all is an ``add_route("*",
        ...)``, so a POST-only stub would see no catch-all at all and the assertion
        would pass without checking anything.
        """
        from kiro_crew.dashboard.routes import connections

        paths: list[str] = []

        class _Router:
            """Records the path of every registration, whatever the verb.

            Generic rather than one method per verb: an enumerated stub goes stale
            the moment a route family uses a verb it does not list, and the failure
            then looks like a routing bug rather than a stale double.
            """

            def __getattr__(self, name):
                def record(*args, **_kw):
                    # add_route("*", path, handler) puts the path second; every
                    # add_<verb>(path, handler) puts it first.
                    for arg in args:
                        if isinstance(arg, str) and arg.startswith("/"):
                            paths.append(arg)
                            break
                    return None

                if name.startswith("add_"):
                    return record
                raise AttributeError(name)

        class _App(dict):
            router = _Router()

        connections.register(_App())  # type: ignore[arg-type]

        embed = next(i for i, p in enumerate(paths) if p.endswith("/embed-token"))
        catch_all = [i for i, p in enumerate(paths) if "{path:.*}" in p]
        assert catch_all, "no catch-all route recorded — the stub missed a verb"
        assert embed < min(catch_all)


# ── gateway identity ─────────────────────────────────────────────────────


class TestGatewayIdentity:
    def test_it_is_stable_and_persisted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config import loader

        loader._invalidate_config_cache()
        from kiro_crew import gateway_identity
        from kiro_crew.gateway_identity import GATEWAY_ID_FILE, gateway_id

        first = gateway_id()
        assert len(first) == 32
        # Drop the memo so the second call has to answer from the FILE. That is
        # what "persisted" claims, and a memo hit would assert nothing about disk.
        gateway_identity._CACHED_IDS.clear()
        assert gateway_id() == first, "minted a second id for one gateway"
        stored = (loader.config_dir() / GATEWAY_ID_FILE).read_text().strip()
        assert stored == first

    def test_a_repeat_read_does_not_touch_the_disk(self, tmp_path, monkeypatch):
        """`/api/health` is the most polled endpoint there is and the id cannot
        change, so only the first resolve may pay for the file."""
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config import loader

        loader._invalidate_config_cache()
        from kiro_crew import gateway_identity

        gateway_identity._CACHED_IDS.clear()
        first = gateway_identity.gateway_id()
        assert len(first) == 32

        reads: list[str] = []

        def counting_read(path):
            reads.append(str(path))
            return ""

        monkeypatch.setattr(gateway_identity, "_read_id", counting_read)
        assert gateway_identity.gateway_id() == first
        assert reads == [], "re-read the id file for a value that cannot change"

    def test_a_corrupt_file_is_replaced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config import loader

        loader._invalidate_config_cache()
        from kiro_crew.gateway_identity import GATEWAY_ID_FILE, gateway_id

        path = loader.config_dir()
        path.mkdir(parents=True, exist_ok=True)
        (path / GATEWAY_ID_FILE).write_text("not-a-uuid\n")
        fresh = gateway_id()
        assert len(fresh) == 32 and fresh != "not-a-uuid"

    def test_reading_without_create_does_not_mint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
        from kiro_crew.config import loader

        loader._invalidate_config_cache()
        from kiro_crew.gateway_identity import GATEWAY_ID_FILE, gateway_id

        assert gateway_id(create=False) == ""
        assert not (loader.config_dir() / GATEWAY_ID_FILE).exists()

    def test_two_homes_get_different_ids(self, tmp_path, monkeypatch):
        """Two gateways sharing a machine but not a data home are two gateways,
        which is precisely the distinction the cycle guard needs."""
        from kiro_crew.config import loader
        from kiro_crew.gateway_identity import gateway_id

        ids = []
        for name in ("home-a", "home-b"):
            monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / name))
            loader._invalidate_config_cache()
            ids.append(gateway_id())
        assert ids[0] != ids[1]


class TestChainedRepairPaths:
    """The three paths that share `connect`'s repairs and did not have them."""

    def _pair(self, tmp_path, monkeypatch, *, hop=53999):
        reg, mgr = _mgr(tmp_path, monkeypatch)
        reg.add(name="B", ssh_host="b-host", remote_port=5476, instance_id="b")
        reg.add(
            name="C",
            ssh_host="c-host",
            remote_port=5476,
            instance_id="c",
            via_instance_id="b",
            via_remote_port=hop,
            via_remote_id=VIA_ID,
        )

        async def no_cycle(_inst, _port):
            return ""

        monkeypatch.setattr(mgr, "_chain_cycle_reason", no_cycle)
        return reg, mgr

    def test_self_heal_mints_before_tier_one_so_a_succeeding_rebuild_dials_the_reported_hop(
        self, tmp_path, monkeypatch
    ):
        """A parent that came back on a different loopback port is the commonest
        reason a chained crew needs healing at all -- and `ssh -L` binds the LOCAL
        side whatever answers remotely, so a rebuild against the row's stale hop
        starts fine and `start()` reports success. Tier 1 therefore marks the crew
        recovered and returns without ever asking the parent where its forward
        listens, leaving it healthy on the board and unreachable in fact. The mint
        rides the parent's own hop, so it runs first.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        assert asyncio.run(mgr.connect("c")).state.value == "connected"

        # The parent reconnected and serves this crew on a different port.
        minted: list[str] = []
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr, port=54321, seen=minted))

        from kiro_crew.instances.ssh_tunnel_manager import TunnelState

        mgr._tunnels["c"].status.state = TunnelState.ERROR
        _FakeTunnel.made = []
        asyncio.run(mgr._recover("c"))

        rebuilt = [t for t in _FakeTunnel.made if t.iid == "c"]
        assert len(rebuilt) == 1, "tier 1's rebuild is the one that has to dial the new hop"
        assert rebuilt[0].remote_port == 54321, "tier 1 healed onto the stale hop port"
        assert minted == ["c"], "tier 1 ran without asking the parent for its port"
        # And the parent's answer is persisted, so a later reconnect agrees.
        assert reg.get("c").via_remote_port == 54321

    def test_self_heal_tier_two_rebuilds_without_minting_again(self, tmp_path, monkeypatch):
        """A chained crew's token is minted at the top of the recovery, so tier 2
        is the rebuild alone -- the same shape fargate's tier 2 has, for the
        opposite reason. Both of its rebuilds dial the port the parent named.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        assert asyncio.run(mgr.connect("c")).state.value == "connected"

        minted: list[str] = []
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr, port=54321, seen=minted))

        from kiro_crew.instances.ssh_tunnel_manager import TunnelState

        fails = {"n": 1}
        real_start = _FakeTunnel.start

        async def start_failing_once(self):
            if fails["n"] > 0:
                fails["n"] -= 1
                self.start_result = False
            return await real_start(self)

        monkeypatch.setattr(_FakeTunnel, "start", start_failing_once)

        mgr._tunnels["c"].status.state = TunnelState.ERROR
        _FakeTunnel.made = []
        asyncio.run(mgr._recover("c"))

        rebuilt = [t for t in _FakeTunnel.made if t.iid == "c"]
        assert len(rebuilt) == 2, "did not reach tier 2 (tier 1's rebuild succeeded)"
        assert [t.remote_port for t in rebuilt] == [54321, 54321]
        assert minted == ["c"], "tier 2 minted a second time for one recovery"
        assert reg.get("c").via_remote_port == 54321

    def test_self_heal_stands_down_when_the_parent_names_no_hop(self, tmp_path, monkeypatch):
        """Dialling the row's port when the parent names none is the exposure
        `connect` refuses: that copy arrived over a pane's postMessage, so it is
        not allowed to choose which loopback-only service on the parent this
        gateway forwards and renders as a crew.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        assert asyncio.run(mgr.connect("c")).state.value == "connected"

        async def mint_naming_no_port(inst, _params):
            mgr._chained_hop_port.pop(inst.id, None)
            return "CHAINED_TOKEN"

        monkeypatch.setattr(mgr, "_mint_through_parent", mint_naming_no_port)

        from kiro_crew.instances.ssh_tunnel_manager import TunnelState

        mgr._tunnels["c"].status.state = TunnelState.ERROR
        _FakeTunnel.made = []
        asyncio.run(mgr._recover("c"))

        assert [t for t in _FakeTunnel.made if t.iid == "c"] == [], "dialled an unnamed hop"
        assert reg.get("c").via_remote_port == 53999

    def test_editing_a_parent_tears_down_the_crews_riding_it(self, tmp_path, monkeypatch):
        """A child's forward was built from the parent's coordinates. Moving the
        parent to another machine while the child's `ssh -L ... <old host>` stays up
        leaves that child reporting CONNECTED and serving the machine the user just
        left, while its row names the new one.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        assert asyncio.run(mgr.connect("b")).state.value == "connected"
        assert asyncio.run(mgr.connect("c")).state.value == "connected"
        assert "c" in mgr._tunnels and "b" in mgr._tunnels

        asyncio.run(mgr.reconfigure("b", lambda: reg.update("b", ssh_host="b-moved")))

        assert "b" not in mgr._tunnels, "kept the edited crew's own forward"
        assert "c" not in mgr._tunnels, "left a chained crew forwarding to the old host"
        # The child keeps its intent: the user edited the PARENT, and forgetting the
        # child would mean reconnecting the parent does not bring its crews back.
        assert reg.get("c").was_connected is True

    def test_a_child_that_will_not_stop_aborts_the_parent_s_edit(self, tmp_path, monkeypatch):
        """Same reason the parent's own failed stop aborts it: persisting new
        coordinates while a forward built from the old ones is still live leaves the
        record describing one machine and the reachable tunnel serving another.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        asyncio.run(mgr.connect("b"))
        asyncio.run(mgr.connect("c"))

        async def wont_stop():
            raise RuntimeError("stop refused")

        mgr._tunnels["c"].stop = wont_stop

        with pytest.raises(Exception):
            asyncio.run(mgr.reconfigure("b", lambda: reg.update("b", ssh_host="b-moved")))

        assert reg.get("b").ssh_host == "b-host", "persisted the edit over a live child"

    def test_a_superseded_parent_hop_remint_is_discarded(self, tmp_path, monkeypatch):
        """Two of the three callers reach this without the manager lock, and the
        mint budget is tens of seconds, so the parent can be disconnected and
        reconnected inside it. Membership cannot see that -- the NEW tunnel
        satisfies it -- so the fresh credential would be overwritten by one minted
        against the generation before it. The tunnel generation is what sees it.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        assert asyncio.run(mgr.connect("b")).state.value == "connected"
        live = mgr.get_token("b")

        async def mint_then_replace(inst, _params):
            # Stand in for a reconnect landing while the mint is in flight.
            mgr._tunnel_epoch["b"] = mgr._tunnel_epoch.get("b", 0) + 1
            return "STALE_GENERATION_TOKEN"

        monkeypatch.setattr(mgr, "_mint_for", mint_then_replace)
        assert asyncio.run(mgr._remint_parent_under_lock("b")) is False
        assert mgr.get_token("b") == live, "overwrote a live credential with a stale one"

    def test_an_undisturbed_parent_hop_remint_still_stores(self, tmp_path, monkeypatch):
        """The guard must not be over-eager: with the generation unchanged the
        re-mint is the whole point of the 401 branch and must land.
        """
        reg, mgr = self._pair(tmp_path, monkeypatch)
        monkeypatch.setattr(mgr, "_mint_through_parent", _relay_mint(mgr))
        asyncio.run(mgr.connect("b"))
        first = mgr.get_token("b")

        async def quiet_mint(_inst, _params):
            return "FRESH_TOKEN"

        monkeypatch.setattr(mgr, "_mint_for", quiet_mint)
        assert asyncio.run(mgr._remint_parent_under_lock("b")) is True
        assert mgr.get_token("b") == "FRESH_TOKEN" != first
