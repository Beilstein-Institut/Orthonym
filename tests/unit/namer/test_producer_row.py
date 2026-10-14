"""The ``producer`` key of the ``name_tiered`` row, and the life of the per-call record."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import producer_trail as pt

from tests.support.producer_pairs import LABELLED_ESTER

pytestmark = pytest.mark.unit


# Rows that other lanes own (metals: ZN; carbon-free species: DS8; lowercase r/s centres: RS) are
# not fixtures here: a lane that merges first could turn their decline into a name.
# The declines of the default tier come from the shipped OPSIN validity gate, which the suite
# switches off unless a test asks for it (tests/conftest.py, the ``opsin_gate`` marker).
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", "CC(=O)OCl"])
def test_every_row_has_the_key(smiles, tier):
    flags = {} if tier == "pin" else _emit_tier_flags(tier)
    row = Orthonym(style="pin", **flags).name_tiered(smiles)
    assert "producer" in row, row
    if row["tier"] == "abstain":
        assert row["producer"] is None, row       # no name ships (the seams add the converse for named rows)


@pytest.mark.opsin_gate
def test_the_decline_row_has_the_key():
    # the default tier declines a name it builds but does not certify: the NO_VERIFIED_PIN row
    row = Orthonym(style="pin").name_tiered(LABELLED_ESTER)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row
    assert "producer" in row and row["producer"] is None


def test_the_key_comes_after_the_existing_keys():
    row = Orthonym(style="pin").name_tiered("CCO")
    assert list(row)[-1] == "producer", list(row)


def test_name_compound_opens_no_trail():
    from orthonym import name_compound
    name_compound("CCO")
    assert pt._TRAIL.get() is None


def test_name_opens_no_trail():
    Orthonym(style="pin").name("CCO")
    assert pt._TRAIL.get() is None


def test_trail_is_closed_after_the_call():
    Orthonym(style="pin").name_tiered("CCO")
    assert pt._TRAIL.get() is None


def test_trail_is_closed_after_an_exception(monkeypatch):
    e = Orthonym(style="pin")
    monkeypatch.setattr(Orthonym, "name", lambda self, s, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        e.name_tiered("CCO")
    assert pt._TRAIL.get() is None


def test_trail_is_closed_after_an_invalid_input():
    with pytest.raises(ValueError):
        Orthonym(style="pin").name_tiered("not a smiles ((")
    assert pt._TRAIL.get() is None


def test_each_call_opens_its_own_record(monkeypatch):
    # a name that records a string for this call must not leak it into the next call's row
    seen = []

    def fake_name(self, smiles, **k):
        seen.append(pt._TRAIL.get())
        pt.note("inner:test", "ethanol")
        return "ethanol"

    monkeypatch.setattr(Orthonym, "name", fake_name)
    monkeypatch.setattr(Orthonym, "_tier_row_and_pin_form",
                        lambda self, smiles, name, prov: ({"name": name, "tier": "pin_verified"}, True))
    e = Orthonym(style="pin")
    first = e.name_tiered("CCO")
    assert first["producer"] == "inner:test", first
    assert seen[0] is not None and pt._TRAIL.get() is None
    monkeypatch.setattr(Orthonym, "name", lambda self, s, **k: "propane")
    second = e.name_tiered("CCC")
    assert second["producer"] is None, second      # "propane" was never noted in this call


def test_the_row_reads_the_name_that_ships(monkeypatch):
    # an entry for a string that does not ship (a rejected candidate) never reaches the row
    def fake_name(self, smiles, **k):
        pt.note("inner:rejected", "propanol")
        pt.note("inner:shipped", "ethanol")
        return "ethanol"

    monkeypatch.setattr(Orthonym, "name", fake_name)
    monkeypatch.setattr(Orthonym, "_tier_row_and_pin_form",
                        lambda self, smiles, name, prov: ({"name": name, "tier": "pin_verified"}, True))
    row = Orthonym(style="pin").name_tiered("CCO")
    assert row["producer"] == "inner:shipped", row


def test_an_abstain_row_has_a_null_producer_even_when_the_name_was_noted(monkeypatch):
    # decision 3: the default tier's NO_VERIFIED_PIN decline builds a name and withholds it
    def fake_name(self, smiles, **k):
        pt.note("inner:built", "ethanol")
        return "ethanol"

    monkeypatch.setattr(Orthonym, "name", fake_name)
    monkeypatch.setattr(Orthonym, "_tier_row_and_pin_form",
                        lambda self, smiles, name, prov: ({"name": name, "tier": "abstain"}, False))
    row = Orthonym(style="pin").name_tiered("CCO")
    assert row["tier"] == "abstain" and row["producer"] is None, row


def test_the_stashed_default_tier_row_gets_the_key(monkeypatch):
    # name derives the default tier's row and stashes it; name_tiered returns a copy of it
    stash_row = {"name": "ethanol", "tier": "pin_verified"}

    def fake_name(self, smiles, **k):
        pt.note("outer:retained_name", "ethanol")
        self._default_tier_row = ("ethanol", stash_row)
        return "ethanol"

    monkeypatch.setattr(Orthonym, "name", fake_name)
    row = Orthonym(style="pin").name_tiered("CCO")
    assert row["producer"] == "outer:retained_name", row
    assert "producer" not in stash_row                  # the stash is copied, never edited


def test_twin_runs_with_no_trail(monkeypatch):
    seen = []
    monkeypatch.setattr(Orthonym, "name",
                        lambda self, s, **k: (seen.append(pt._TRAIL.get()), "ethanol")[1])
    tok = pt.open_trail("CCO")
    try:
        engine = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
        assert engine._strict_pin_twin_name("CCO") == "ethanol"
        assert pt._TRAIL.get() is not None            # the call's own record is back
    finally:
        pt.close_trail(tok)
    assert seen == [None]


def test_twin_records_nothing_into_the_call(monkeypatch):
    def fake_name(self, s, **k):
        pt.note("inner:twin", "ethanol")
        return "ethanol"

    monkeypatch.setattr(Orthonym, "name", fake_name)
    tok = pt.open_trail("CCO")
    try:
        Orthonym(style="pin", **_emit_tier_flags("best-effort"))._strict_pin_twin_name("CCO")
        assert pt.lookup("ethanol") is None
    finally:
        pt.close_trail(tok)


def test_twin_restores_the_record_after_a_failure(monkeypatch):
    def boom(self, s, **k):
        raise RuntimeError("twin failure")

    monkeypatch.setattr(Orthonym, "name", boom)
    tok = pt.open_trail("CCO")
    try:
        engine = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
        assert engine._strict_pin_twin_name("CCO") is None     # a twin failure never breaks the row
        assert pt._TRAIL.get() is not None
    finally:
        pt.close_trail(tok)


def test_help_text_names_the_key():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "orthonym", "--help"], capture_output=True, text=True).stdout
    # the word "producers" is already in the help of another option: name the list of row keys
    assert "verified, spelling_failures and producer." in " ".join(out.split())
