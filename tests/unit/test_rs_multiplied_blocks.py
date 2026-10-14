"""The descriptor check of the stereo-layer carve-out covers every copy of a multiplied
block, and refuses a name that cites a block for fewer copies than the input holds.

 "Achiral cyclic compounds" (the Blue Book): the configuration at the
pseudoasymmetric centres of 1,4-disubstituted cyclohexanes is specified with r and s;
``bis[(1r,4r)-4-methylcyclohexyl]phosphane (PIN)`` (:48117) cites one block for two rings
and ``bis[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane (PIN)`` (:48121)
cites it for two of three. "NAMING OF STEREOISOMERS" (:44643): "When they relate to
substituent groups, they are cited at the front of the corresponding prefix."
"""
import pytest

from orthonym.namer import _stereo_descriptors_verified
from orthonym.rules.stereochemistry import strip_stereo_blocks
from tests.support.default_tier import default_tier_row

pytestmark = pytest.mark.opsin_gate

TT = "C[C@H]1CC[C@H](P[C@H]2CC[C@H](C)CC2)CC1"
TC = "C[C@H]1CC[C@H](P[C@H]2CC[C@@H](C)CC2)CC1"
TTT = "C[C@H]1CC[C@H](P([C@H]2CC[C@H](C)CC2)[C@H]2CC[C@H](C)CC2)CC1"
TTC = "C[C@H]1CC[C@H](P([C@H]2CC[C@H](C)CC2)[C@H]2CC[C@@H](C)CC2)CC1"
TCC = "C[C@H]1CC[C@H](P([C@H]2CC[C@@H](C)CC2)[C@H]2CC[C@@H](C)CC2)CC1"
CCC = "C[C@H]1CC[C@@H](P([C@H]2CC[C@@H](C)CC2)[C@H]2CC[C@@H](C)CC2)CC1"
# 1,3,5-trimethylcyclohexane: the cis,trans isomer has three tagged ring centres and one CIP
# label (r, on the odd centre); the all-cis isomer is labelled s on all three.
MTM_CT = "C[C@H]1C[C@H](C)C[C@@H](C)C1"
MTM_CT2 = "C[C@H]1C[C@H](C)C[C@H](C)C1"       # the same compound, written from the other end
MTM_CCC = "C[C@H]1C[C@@H](C)C[C@@H](C)C1"
# a double bond cited for both copies of an ester group
DI_ZZ = "CCCCCC/C=C\\CCCCCCCC(=O)OCC(CO)OC(=O)CCCCCCC/C=C\\CCCCCC"
DI_ZE = "CCCCCC/C=C\\CCCCCCCC(=O)OCC(CO)OC(=O)CCCCCCC/C=C/CCCCCC"

TABLE = [
    ("bis[(1r,4r)-4-methylcyclohexyl]phosphane", TT, True),                                #:48117
    ("(4-methylcyclohexyl)[(1r,4r)-4-methylcyclohexyl]phosphane", TT, False),              # omits ring 2
    ("bis[(1r,4r)-4-methylcyclohexyl]phosphane", TC, False),
    ("bis(4-methylcyclohexyl)[(1r,4r)-4-methylcyclohexyl]phosphane", TTT, False),
    ("tris[(1r,4r)-4-methylcyclohexyl]phosphane", TTT, True),
    ("bis[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane", TTC, True),   #:48121
    ("bis[(1r,4r)-4-methylcyclohexyl](4-methylcyclohexyl)phosphane", TTC, False),
    # the senses of the copies decide: two trans rings cited for a trans/cis/cis input
    ("bis[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane", TCC, False),
    ("[(1r,4r)-4-methylcyclohexyl]bis[(1s,4s)-4-methylcyclohexyl]phosphane", TCC, True),
    ("tris[(1s,4s)-4-methylcyclohexyl]phosphane", CCC, True),
    ("tris[(1r,4r)-4-methylcyclohexyl]phosphane", CCC, False),
    ("bis[(1s,4s)-4-methylcyclohexyl](4-methylcyclohexyl)phosphane", CCC, False),
    # a tagged centre that no CIP label names is not a unit a descriptor can cite
    ("(1r)-1,3,5-trimethylcyclohexane", MTM_CT, True),
    ("(1r)-1,3,5-trimethylcyclohexane", MTM_CT2, True),
    ("(1s,3s,5s)-1,3,5-trimethylcyclohexane", MTM_CCC, True),
    ("(1s,3s)-1,3,5-trimethylcyclohexane", MTM_CCC, False),                                # omits a labelled centre
    ("3-hydroxypropane-1,2-diyl di[(9Z)-hexadec-9-enoate]", DI_ZZ, True),
    ("3-hydroxypropane-1,2-diyl di[(9Z)-hexadec-9-enoate]", DI_ZE, False),                 # one copy is E
]


@pytest.mark.parametrize("name,smiles,expected", TABLE)
def test_verifier_table(name, smiles, expected):
    assert _stereo_descriptors_verified(name, smiles, strip_stereo_blocks(name)[0]) is expected


@pytest.mark.parametrize("smiles,name", [
    (TTC, "bis[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane"),   #:48121
    (TTT, "tris[(1r,4r)-4-methylcyclohexyl]phosphane"),
    (TT, "bis[(1r,4r)-4-methylcyclohexyl]phosphane"),                                #:48117, unchanged
    (TC, "[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane"),       # unchanged
    (TCC, "[(1r,4r)-4-methylcyclohexyl]bis[(1s,4s)-4-methylcyclohexyl]phosphane"),
    (CCC, "tris[(1s,4s)-4-methylcyclohexyl]phosphane"),
    (MTM_CT, "(1r)-1,3,5-trimethylcyclohexane"),
    (MTM_CT2, "(1r)-1,3,5-trimethylcyclohexane"),
    (MTM_CCC, "(1s,3s,5s)-1,3,5-trimethylcyclohexane"),
])
def test_default_tier_names(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row


@pytest.mark.parametrize("name", [
    "(1r,4r)-cyclohexane-1,4-diol",                                                   # a leading block
    "N-[(1r,4r)-4-methylcyclohexyl]benzamide",                                        # a nested block
    "bis[(1r,4r)-4-methylcyclohexyl]phosphane",                                       # a multiplied block
    "bis[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane",           # two blocks
    "(2R)-N-[(1r,4r)-4-methylcyclohexyl]-3-[(1s,3s)-3-hydroxycyclobutyl]propanamide",  # leading and two nested
    "1-{[(1r,4r)-4-methylcyclohexyl]methyl}piperidine",                               # two enclosing marks
    "rel-(1R,2S)-cyclohexane-1,2-diol",                                               # a word and a block
    "hexane",                                                                         # no block
])
def test_locate_stereo_blocks_matches_strip(name):
    from orthonym.rules.stereochemistry import locate_stereo_blocks
    stripped, blocks = strip_stereo_blocks(name)
    located = locate_stereo_blocks(name)
    assert [text for _s, _e, text in located] == blocks
    assert all(name[s:e] == text + "-" for s, e, text in located)
    rest = list(name)
    for s, e, _text in sorted(located, reverse=True):
        del rest[s:e]
    assert "".join(rest).strip() == stripped


def test_block_units_bind_the_cited_atoms():
    from orthonym.namer import _stereo_block_units
    diol = "(1r,4r)-cyclohexane-1,4-diol"
    records = _stereo_block_units(diol, "O[C@H]1CC[C@H](O)CC1", strip_stereo_blocks(diol)[0])
    assert [(r.text, r.span, r.multiplicity, r.units) for r in records] == [
        ("(1r,4r)", (0, 8), 1, (("a", 1), ("a", 4)))]

    name = "bis[(1r,4r)-4-methylcyclohexyl]phosphane"
    records = _stereo_block_units(name, TT, strip_stereo_blocks(name)[0])
    assert [(r.text, r.span, r.multiplicity, r.units) for r in records] == [
        ("(1r,4r)", (4, 12), 2, (("a", 1), ("a", 4), ("a", 6), ("a", 9)))]


def test_block_units_of_a_name_with_two_blocks():
    from orthonym.namer import _stereo_block_units
    name = "[(1r,4r)-4-methylcyclohexyl]bis[(1s,4s)-4-methylcyclohexyl]phosphane"
    records = _stereo_block_units(name, TCC, strip_stereo_blocks(name)[0])
    assert [(r.text, r.multiplicity, r.units) for r in records] == [
        ("(1r,4r)", 1, (("a", 1), ("a", 4))),
        ("(1s,4s)", 2, (("a", 6), ("a", 9), ("a", 13), ("a", 16)))]
    assert [name[s:e] for s, e in (r.span for r in records)] == ["(1r,4r)-", "(1s,4s)-"]


def test_no_reading_is_none():
    from orthonym.namer import _stereo_block_units
    name = "(4-methylcyclohexyl)[(1r,4r)-4-methylcyclohexyl]phosphane"
    assert _stereo_block_units(name, TT, strip_stereo_blocks(name)[0]) is None
