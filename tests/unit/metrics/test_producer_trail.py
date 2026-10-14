"""The per-call record of which code built a name (``orthonym.metrics.producer_trail``)."""
import pytest
from rdkit import Chem

from orthonym.metrics import producer_trail as pt

pytestmark = pytest.mark.unit
ETHANOL = Chem.MolFromSmiles("CCO")
PROPANE = Chem.MolFromSmiles("CCC")


def test_nothing_recorded_without_a_trail():
    assert pt.noted("inner:x", "ethanol", ETHANOL) == "ethanol"
    assert pt.lookup("ethanol") is None


def test_noted_returns_the_same_object():
    class R:                      # an object with.name, like a GeneralEngineResult
        name = "ethanol"
    r = R()
    tok = pt.open_trail("CCO")
    try:
        assert pt.noted("engine:general", r, ETHANOL) is r
        assert pt.lookup("ethanol") == "engine:general"
    finally:
        pt.close_trail(tok)


def test_noted_returns_a_string_as_the_same_object():
    name = "".join(["etha", "nol"])
    tok = pt.open_trail("CCO")
    try:
        assert pt.noted("inner:a", name, ETHANOL) is name
        assert pt.carried(name, "(1R)-ethanol") == "(1R)-ethanol"
    finally:
        pt.close_trail(tok)


def test_a_named_tuple_result_is_read_by_its_name_field():
    from orthonym.assembly.name_tree import NamingResult   # a NamedTuple: name, tree, hint
    result = NamingResult("ethanol")
    tok = pt.open_trail("CCO")
    try:
        assert pt.noted("outer:retained_name", result, ETHANOL) is result
        assert pt.lookup("ethanol") == "outer:retained_name"
    finally:
        pt.close_trail(tok)


def test_a_value_with_no_string_name_is_not_recorded():
    tok = pt.open_trail("CCO")
    try:
        for value in (None, 7, ("ethanol", 1), ["ethanol"], object(), "", b"ethanol"):
            assert pt.noted("inner:a", value, ETHANOL) is value
        assert pt._TRAIL.get().by_name == {}
    finally:
        pt.close_trail(tok)


def test_first_producer_keeps_the_string():
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)
        pt.noted("outer:b", "ethanol", ETHANOL)
        assert pt.lookup("ethanol") == "inner:a"
    finally:
        pt.close_trail(tok)


def test_no_mol_means_the_input_itself():
    # The table lookups are of the input itself and pass no molecule: always recorded.
    tok = pt.open_trail("CCO")
    try:
        pt.noted("table:coordination", "ethanol")
        assert pt.lookup("ethanol") == "table:coordination"
    finally:
        pt.close_trail(tok)


def test_fragment_return_is_not_recorded():
    tok = pt.open_trail("CCO")            # 3 heavy atoms
    try:
        pt.noted("inner:a", "propane", PROPANE)     # 3 heavy atoms: recorded
        pt.noted("inner:b", "methyl", Chem.MolFromSmiles("C"))   # 1: dropped
        assert pt.lookup("propane") == "inner:a" and pt.lookup("methyl") is None
    finally:
        pt.close_trail(tok)


def test_hydrogens_do_not_count_as_heavy_atoms():
    # [H][H] has no heavy atoms: a trail opened on it takes a molecule with none.
    tok = pt.open_trail("[H][H]")
    try:
        assert pt._TRAIL.get().n_heavy == 0
        pt.noted("table:trivial", "molecular hydrogen", Chem.MolFromSmiles("[H][H]"))
        assert pt.lookup("molecular hydrogen") == "table:trivial"
    finally:
        pt.close_trail(tok)


def test_an_unreadable_input_records_no_whole_molecule_seam():
    tok = pt.open_trail("not a smiles")
    try:
        assert pt._TRAIL.get().n_heavy == -1
        pt.noted("inner:a", "ethanol", ETHANOL)         # with a molecule: dropped
        pt.noted("table:coordination", "ethanol")       # without one: the input itself
        assert pt.lookup("ethanol") == "table:coordination"
    finally:
        pt.close_trail(tok)


@pytest.mark.parametrize("name", ["unknown organic compound", "inorganic compound (not supported)",
                                  "11-unknownundec-9-enoate", ""])
def test_failure_names_are_not_recorded(name):
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", name, ETHANOL)
        pt.noted("table:trivial", name)
        assert pt.lookup(name) is None
    finally:
        pt.close_trail(tok)


def test_carried_passes_the_label_on():
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)
        assert pt.carried("ethanol", "(1R)-ethanol") == "(1R)-ethanol"
        assert pt.lookup("(1R)-ethanol") == "inner:a"
        assert pt.carried("ethanol", "unknown organic compound") == "unknown organic compound"
        assert pt.lookup("unknown organic compound") is None
    finally:
        pt.close_trail(tok)


def test_carried_from_an_unlabelled_string_labels_nothing():
    tok = pt.open_trail("CCO")
    try:
        assert pt.carried("ethanol", "(1R)-ethanol") == "(1R)-ethanol"
        assert pt.lookup("(1R)-ethanol") is None
        assert pt.carried("ethanol", "ethanol") == "ethanol"
    finally:
        pt.close_trail(tok)


def test_carried_never_replaces_a_label():
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)
        pt.noted("outer:b", "(1R)-ethanol", ETHANOL)
        pt.carried("ethanol", "(1R)-ethanol")
        assert pt.lookup("(1R)-ethanol") == "outer:b"
    finally:
        pt.close_trail(tok)


def test_carried_returns_the_argument_without_a_trail():
    name_out = "".join(["(1R)-etha", "nol"])
    assert pt.carried("ethanol", name_out) is name_out
    assert pt.carried(None, None) is None
    assert pt.carried(3, 4) == 4


def test_seam_never_raises(monkeypatch):
    tok = pt.open_trail("CCO")
    try:
        class Bad:
            def GetNumHeavyAtoms(self):
                raise RuntimeError("boom")
        assert pt.noted("inner:a", "ethanol", Bad()) == "ethanol"

        class BadName:
            @property
            def name(self):
                raise RuntimeError("boom")
        value = BadName()
        assert pt.noted("inner:a", value, ETHANOL) is value

        def boom(name):
            raise RuntimeError("boom")
        monkeypatch.setattr("orthonym.errors.is_failure_name", boom)
        assert pt.noted("inner:a", "ethanol", ETHANOL) == "ethanol"
        pt.carried("ethanol", "(1R)-ethanol")          # must not raise either
        assert pt.lookup("ethanol") is None
    finally:
        pt.close_trail(tok)


def test_a_broken_label_check_does_not_lose_a_recorded_label(monkeypatch):
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)

        def boom(name):
            raise RuntimeError("boom")
        monkeypatch.setattr("orthonym.errors.is_failure_name", boom)
        assert pt.carried("ethanol", "(1R)-ethanol") == "(1R)-ethanol"
        assert pt.lookup("ethanol") == "inner:a"
        assert pt.lookup("(1R)-ethanol") is None
    finally:
        pt.close_trail(tok)


def test_lookup_of_a_non_string_is_none():
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)
        assert pt.lookup(None) is None and pt.lookup(7) is None and pt.lookup(("ethanol",)) is None
    finally:
        pt.close_trail(tok)


def test_suspended_records_nothing_and_restores():
    tok = pt.open_trail("CCO")
    try:
        with pt.suspended():
            pt.noted("inner:a", "ethanol", ETHANOL)
        assert pt.lookup("ethanol") is None
        pt.noted("inner:a", "ethanol", ETHANOL)
        assert pt.lookup("ethanol") == "inner:a"
    finally:
        pt.close_trail(tok)


def test_suspended_restores_the_record_after_an_exception():
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)
        with pytest.raises(RuntimeError):
            with pt.suspended():
                assert pt._TRAIL.get() is None
                raise RuntimeError("x")
        assert pt.lookup("ethanol") == "inner:a"
    finally:
        pt.close_trail(tok)


def test_close_restores_the_outer_trail():
    outer = pt.open_trail("CCO")
    inner = pt.open_trail("CCO")
    pt.noted("inner:a", "ethanol", ETHANOL)
    pt.close_trail(inner)
    assert pt.lookup("ethanol") is None
    pt.close_trail(outer)


def test_close_twice_leaves_no_record():
    tok = pt.open_trail("CCO")
    pt.close_trail(tok)
    pt.close_trail(tok)                      # a used token: no raise, no record
    assert pt._TRAIL.get() is None


def test_each_trail_is_its_own_record():
    first = pt.open_trail("CCO")
    pt.noted("inner:a", "ethanol", ETHANOL)
    pt.close_trail(first)
    second = pt.open_trail("CCO")
    try:
        assert pt.lookup("ethanol") is None
    finally:
        pt.close_trail(second)


def test_the_variable_is_idle_outside_a_call():
    assert pt._TRAIL.get() is None


def test_the_trail_does_not_change_what_get_provenance_returns():
    from orthonym.metrics.provenance import get_provenance
    before = dict(get_provenance())
    tok = pt.open_trail("CCO")
    try:
        pt.noted("inner:a", "ethanol", ETHANOL)
        assert dict(get_provenance()) == before
        assert "producer" not in get_provenance()
    finally:
        pt.close_trail(tok)
