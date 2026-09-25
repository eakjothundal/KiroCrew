"""The gateway's persistent identity: one id per data home, or none at all.

The id exists so a gateway can tell ITSELF from another gateway at the far end of
a forward -- the chain cycle guard's whole question. Two answers for one data home
is therefore the failure that matters more than any of the others here, which is
what most of these cases are about.
"""

from __future__ import annotations

import os
import stat

import pytest

from kiro_crew import gateway_identity


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each case starts with an empty process cache.

    The cache is keyed by path and lives for the life of the process, so without
    this a later case reads an earlier case's answer for its own tmp_path only by
    accident of ordering.
    """
    gateway_identity._CACHED_IDS.clear()
    yield
    gateway_identity._CACHED_IDS.clear()


def _home(tmp_path, monkeypatch):
    monkeypatch.setattr(gateway_identity, "config_dir", lambda: tmp_path)
    return tmp_path / gateway_identity.GATEWAY_ID_FILE


class TestOneIdPerDataHome:
    def test_it_mints_once_and_reads_the_same_id_back(self, tmp_path, monkeypatch):
        path = _home(tmp_path, monkeypatch)
        first = gateway_identity.gateway_id()
        assert gateway_identity._ID_RE.match(first)
        assert path.read_text(encoding="utf-8").strip() == first
        # A second reader in another process has no cache; it must adopt the
        # persisted id rather than mint its own.
        gateway_identity._CACHED_IDS.clear()
        assert gateway_identity.gateway_id() == first

    def test_a_second_repair_adopts_the_first_instead_of_minting_again(self, tmp_path, monkeypatch):
        """The regression. Two gateways repairing one corrupt file must converge on
        one id: each removing the other's replacement leaves two ids for a single
        data home, and the cycle guard would then be comparing two answers for one
        gateway."""
        path = _home(tmp_path, monkeypatch)
        path.write_text("not-a-gateway-id", encoding="utf-8")

        repaired = gateway_identity.gateway_id()
        assert gateway_identity._ID_RE.match(repaired)

        # The second process: same home, no cache, and the file is now VALID, so
        # the repair path must not run at all.
        gateway_identity._CACHED_IDS.clear()
        assert gateway_identity.gateway_id() == repaired
        assert path.read_text(encoding="utf-8").strip() == repaired

    def test_the_cached_value_is_the_one_on_disk(self, tmp_path, monkeypatch):
        """Caching what this process WROTE rather than what the file holds is how
        two processes end up disagreeing, so the cache is filled from a read."""
        path = _home(tmp_path, monkeypatch)
        minted = gateway_identity.gateway_id()
        assert (
            gateway_identity._CACHED_IDS[str(path)]
            == path.read_text(encoding="utf-8").strip()
            == minted
        )

    @pytest.mark.parametrize(
        "corrupt",
        ["", "   ", "nope", "0123456789abcdef", "Z" * 32, "0" * 33],
    )
    def test_every_malformed_shape_is_replaced(self, tmp_path, monkeypatch, corrupt):
        path = _home(tmp_path, monkeypatch)
        path.write_text(corrupt, encoding="utf-8")
        got = gateway_identity.gateway_id()
        assert gateway_identity._ID_RE.match(got), f"{corrupt!r} was published as an id"
        assert path.read_text(encoding="utf-8").strip() == got


class TestTheReadOnlyMode:
    def test_create_false_reports_absence_without_writing(self, tmp_path, monkeypatch):
        path = _home(tmp_path, monkeypatch)
        assert gateway_identity.gateway_id(create=False) == ""
        assert not path.exists(), "a read-only call minted a file"

    def test_create_false_still_reads_a_valid_id(self, tmp_path, monkeypatch):
        path = _home(tmp_path, monkeypatch)
        minted = gateway_identity.gateway_id()
        gateway_identity._CACHED_IDS.clear()
        assert gateway_identity.gateway_id(create=False) == minted
        assert path.exists()

    def test_create_false_does_not_repair_a_corrupt_file(self, tmp_path, monkeypatch):
        path = _home(tmp_path, monkeypatch)
        path.write_text("corrupt", encoding="utf-8")
        assert gateway_identity.gateway_id(create=False) == ""
        assert path.read_text(encoding="utf-8") == "corrupt", "a read-only call rewrote the file"


class TestItNeverRaises:
    def test_an_unwritable_home_yields_the_process_local_id(self, tmp_path, monkeypatch):
        """Refusing would turn an unwritable directory into a gateway that cannot
        connect a crew at all, and a process-local id still tells two LIVE
        gateways apart, which is the comparison the cycle guard makes."""

        def boom(*_a, **_kw):
            raise OSError("read-only file system")

        _home(tmp_path, monkeypatch)
        monkeypatch.setattr(gateway_identity.platform_compat, "open_lock_file", boom)
        assert gateway_identity.gateway_id() == gateway_identity._IN_MEMORY_ID

    def test_a_directory_where_the_id_belongs_yields_the_fallback(self, tmp_path, monkeypatch):
        path = _home(tmp_path, monkeypatch)
        path.mkdir()
        assert gateway_identity.gateway_id() == gateway_identity._IN_MEMORY_ID


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits; Windows ACLs are asserted elsewhere")
class TestItIsNotWorldReadable:
    def test_the_id_file_is_owner_only(self, tmp_path, monkeypatch):
        """It is an identity fingerprint, so another local account must not be able
        to read it out of the data home."""
        path = _home(tmp_path, monkeypatch)
        gateway_identity.gateway_id()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert not mode & stat.S_IRGRP, f"group-readable: {mode:o}"
        assert not mode & stat.S_IROTH, f"world-readable: {mode:o}"
