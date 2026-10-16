"""A promotion re-run name keeps its certification when no breadth piece was kept in the re-run.

``Orthonym._name_with_pin_promotion`` names a molecule a second time with the best-effort ring-
substituent producers admitted (``rules.pin_vocabulary.promote_at_pin_tier``) and used to label every
name it kept ``pin_unverified``. The label is now decided by what the re-run kept: ``promote_at_pin_tier``
records each breadth name it keeps (``record_breadth_promotion``) and ``record_pin_promotion_rerun`` sets
the whole-call flag only if one was recorded. A name every piece of which a strict producer built
(``assembly.strict_ring_prefix``) is then the strict path's own name.
"""
import pytest

from orthonym.assembly import nested_memo
from orthonym.assembly.memo import pin_promotion_var
from orthonym.metrics import provenance as pv
from orthonym.rules.pin_vocabulary import promote_at_pin_tier

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_provenance():
    pv.clear_provenance()
    yield
    pv.clear_provenance()


@pytest.fixture
def in_rerun():
    tok = pin_promotion_var.set(True)
    yield
    pin_promotion_var.reset(tok)


def test_rerun_without_a_breadth_piece_is_not_flagged():
    pv.record_pin_promotion_rerun()
    assert pv.get_provenance()["pin_promotion_rerun"] is False


def test_rerun_with_a_breadth_piece_is_flagged():
    pv.record_breadth_promotion("4-methylcyclohex-3-en-1-yl")
    pv.record_pin_promotion_rerun()
    assert pv.get_provenance()["pin_promotion_rerun"] is True


def test_a_breadth_piece_the_name_does_not_contain_does_not_flag_it():
    # a route the re-run left: the shipped name is built from strict pieces only; flagging it would make
    # the label depend on the order of the input atoms
    pv.record_breadth_promotion("(2R)-2-(hydroxymethyl)azetidin-1-yl")
    pv._GATE_OUTCOME_NAME.set("[(2R)-1-(pyrimidin-2-yl)azetidin-2-yl]methanol")
    pv.record_pin_promotion_rerun()
    assert pv.get_provenance()["pin_promotion_rerun"] is False


def test_a_breadth_piece_the_name_contains_flags_it():
    pv.record_breadth_promotion("2-(thiophen-2-yl)-1,3-thiazol-4-yl")
    pv._GATE_OUTCOME_NAME.set("1-[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine")
    pv.record_pin_promotion_rerun()
    assert pv.get_provenance()["pin_promotion_rerun"] is True


def test_promote_at_pin_tier_records_the_name_it_keeps(in_rerun):
    assert promote_at_pin_tier(lambda: "cyclohex-3-en-1-yl") == "cyclohex-3-en-1-yl"
    assert pv.get_provenance()["breadth_promoted_names"] == ("cyclohex-3-en-1-yl",)


def test_promote_at_pin_tier_records_nothing_for_a_rejected_name(in_rerun):
    # a multi-word name is rejected and the provenance rolled back
    assert promote_at_pin_tier(lambda: "two words") is None
    assert promote_at_pin_tier(lambda: None) is None
    assert pv.get_provenance()["breadth_promoted_names"] == ()


def test_promote_at_pin_tier_outside_the_rerun_records_nothing():
    assert promote_at_pin_tier(lambda: "cyclohex-3-en-1-yl") is None
    assert pv.get_provenance()["breadth_promoted_names"] == ()


def test_the_record_survives_a_rollback_like_the_other_name_scoped_records():
    # a memo entry computed under a kept breadth piece outlives the rollback of an enclosing
    # promotion: a later hit must still meet the record (restore_provenance docstring)
    snap = pv.get_provenance()
    pv.record_breadth_promotion("cyclohex-3-en-1-yl")
    pv.restore_provenance(snap)
    assert pv.get_provenance()["breadth_promoted_names"] == ("cyclohex-3-en-1-yl",)


def test_clear_provenance_starts_the_next_molecule_clean():
    pv.record_breadth_promotion("cyclohex-3-en-1-yl")
    pv.clear_provenance()
    assert pv.get_provenance()["breadth_promoted_names"] == ()


def test_the_replay_memo_carries_the_record():
    # nested_memo._VARS keys must equal get_provenance keys; the record is an accumulator
    assert set(nested_memo._VARS) == set(pv.get_provenance())
    assert "breadth_promoted_names" in nested_memo._ACCUMULATORS


def test_the_record_is_not_read_by_the_non_pin_label():
    # name_carries_non_pin_part must not read it: it would drop the re-run name itself
    pv.record_breadth_promotion("cyclohex-3-en-1-yl")
    assert not pv.name_carries_non_pin_part(pv.get_provenance(), "3-(cyclohex-3-en-1-yl)butanal")


def test_an_isolated_body_does_not_drop_a_breadth_piece():
    # ``assembly.memo`` replays no provenance: a piece built inside ``isolated_provenance`` (the chain
    # seniority check, ``perception.chains``) and served from the memo later must still be recorded
    with pv.isolated_provenance():
        pv.record_breadth_promotion("cyclohex-3-en-1-yl")
    assert pv.get_provenance()["breadth_promoted_names"] == ("cyclohex-3-en-1-yl",)
