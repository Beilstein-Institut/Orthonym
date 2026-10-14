"""Labels of the default-tier names that cite lowercase r/s,.

OPSIN 2.9.0 reads no lowercase ``r``/``s``, so the stereo-layer carve-out of the validity gate
ships such a name when the descriptor check accepts every descriptor, with the constitution as
the only part OPSIN confirmed (``gate_outcome`` ``self_consistency_constitution_only``, row
``verified`` ``opsin_constitution``). The row is a name in the form of a preferred name that is
not certified as the PIN: ``tier`` ``pin_unverified``, ``is_pin`` false. The submitted paper's
rule for the default tier states it: a strict-path name whose stereodescriptors OPSIN cannot
read is "verified by the constitution and the descriptor check" (``_default_tier_emits``).
``guide/tiers.md`` shows the label for ``(4as,8as)-decahydronaphthalene``.

A name that cites R/S only is read back in full by OPSIN and is ``pin_verified``; the last test
shows that the three values below are those of the lowercase descriptors and not of every
stereo name.
"""
import pytest

from tests.support.default_tier import default_tier_row

pytestmark = pytest.mark.opsin_gate

TT = "C[C@H]1CC[C@H](P[C@H]2CC[C@H](C)CC2)CC1"
TTT = "C[C@H]1CC[C@H](P([C@H]2CC[C@H](C)CC2)[C@H]2CC[C@H](C)CC2)CC1"
TTC = "C[C@H]1CC[C@H](P([C@H]2CC[C@H](C)CC2)[C@H]2CC[C@@H](C)CC2)CC1"


@pytest.mark.parametrize("smiles,name", [
    ("O[C@H]1CC[C@H](O)CC1", "(1r,4r)-cyclohexane-1,4-diol"),                          # the Blue Book
    ("O[C@H]1CC[C@@H](O)CC1", "(1s,4s)-cyclohexane-1,4-diol"),                         #:48111
    ("N[C@H]1CC[C@H](C(=O)O)CC1", "(1r,4r)-4-aminocyclohexane-1-carboxylic acid"),
    ("C[C@H]1CC[C@H](NS(=O)(=O)c2ccccc2)CC1",
     "N-[(1r,4r)-4-methylcyclohexyl]benzenesulfonamide"),                              # a prefix of a host
    (TT, "bis[(1r,4r)-4-methylcyclohexyl]phosphane"),                                  #:48117
    (TTT, "tris[(1r,4r)-4-methylcyclohexyl]phosphane"),
    (TTC, "bis[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane"),     #:48121
    ("C[C@H](S)[C@@H](S)[C@@H](C)S", "(2R,3r,4S)-pentane-2,3,4-trithiol"),             #:45783, (R, r, S)
    ("C1CC[C@@H]2CCCC[C@@H]2C1", "(4as,8as)-decahydronaphthalene"),                   # guide/tiers.md
    ("C1CC[C@H]2CCC[C@@H]12", "(3as,6as)-octahydropentalene"),
    ("O[C@@H]1C[C@@H](O)C1", "(1r,3r)-cyclobutane-1,3-diol"),
    ("C[C@@H]1CC[C@H](CC1)c1ccccc1", "[(1r,4r)-4-methylcyclohexyl]benzene"),
])
def test_default_tier_r_s_name_is_pin_unverified(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_unverified", row
    assert row["is_pin"] is False, row
    assert row["gate_outcome"] == "self_consistency_constitution_only", row


def test_a_name_that_cites_r_s_only_is_pin_verified():
    # OPSIN reads (1S,2S) and checks the full key: the label of the lowercase names is not the
    # label of every stereo name (``_default_tier_emits`` takes the pin_verified row first)
    row = default_tier_row("O[C@H]1CCCC[C@@H]1O")
    assert row["name"] == "(1S,2S)-cyclohexane-1,2-diol", row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["gate_outcome"] == "self_consistency_verified", row
