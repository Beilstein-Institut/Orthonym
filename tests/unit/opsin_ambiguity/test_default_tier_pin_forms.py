"""The default tier ships the preferred IUPAC name or declines: a name that is valid and is not the
PIN is labelled below it.

A ring prefix that cites its locant makes a name read cleanly by OPSIN; whether the rest
of the name is the PIN is a separate question, answered here for the groups that sit on a ring
heteroatom and for a mononuclear parent with two prefixes.
"""
import pytest

from ._helpers import assert_clean, name_row

pytestmark = pytest.mark.opsin_gate


# Names based on suffixes (the Blue Book): "The suffix 'carbaldehyde' is used
# when the -CHO group is attached to a carbon atom of a ring or ring system, or to a heteroatom";
# the chalcogen analogue -CHS takes 'carbothialdehyde', Table 6.4,:35036).
FORMYL_ON_RING_HETEROATOM = [
    ("O=CN1CC1", "aziridine-1-carbaldehyde"),
    ("O=CN1CCC1", "azetidine-1-carbaldehyde"),
    ("O=CN1CCCC1", "pyrrolidine-1-carbaldehyde"),
    ("O=CN1CCOCC1", "morpholine-4-carbaldehyde"),
    ("S=CN1CC1", "aziridine-1-carbothialdehyde"),
]


@pytest.mark.parametrize("smiles,pin", FORMYL_ON_RING_HETEROATOM)
def test_formyl_group_on_a_ring_heteroatom_is_a_carbaldehyde(smiles, pin):
    row = name_row(smiles, tier="pin")
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row
    assert_clean(smiles, tier="pin", must_contain=pin)


# (the Blue Book): "The carboxy group can be attached to any atom, carbon or
# heteroatom, of any parent hydride"; (:29886) 'pyrrolidine-1-carboxylic acid (PIN)';
# the chalcogen analogues are named with the suffixes and the parent hydride,:30270).
CARBOXY_ON_RING_HETEROATOM = [
    ("C1CN1C(=O)SC2=CC=CC=C2", "S-phenyl aziridine-1-carbothioate"),
    ("O=C(SC)N1CC1", "S-methyl aziridine-1-carbothioate"),
    ("O=C(SC)N1CCOCC1", "S-methyl morpholine-4-carbothioate"),
    ("O=C([Se]C)N1CCCC1", "Se-methyl pyrrolidine-1-carboselenoate"),
]


@pytest.mark.parametrize("smiles,pin", CARBOXY_ON_RING_HETEROATOM)
def test_carboxy_group_on_a_ring_heteroatom_is_a_suffix_of_the_ring(smiles, pin):
    row = name_row(smiles, tier="pin")
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


# a ring that carries a substituent, or is not a single ring, has no spelling of its own in the
# producer: the methane name round-trips and is kept below the PIN, so the default tier declines
NO_RING_PARENT_NAME = [
    ("O=CN1CCc2ccccc21", "(2,3-dihydro-1H-indol-1-yl)methanone"),
    ("O=C(SC)N1C(C)CC1", "S-methyl (2-methylazetidin-1-yl)methanethioate"),
]


@pytest.mark.parametrize("smiles,name", NO_RING_PARENT_NAME)
def test_a_name_that_is_not_the_pin_is_not_shipped_as_one(smiles, name):
    # (the Blue Book) and (:29876), quoted above
    wide = name_row(smiles, tier="best-effort")
    assert wide["name"] == name and wide["tier"] == "systematic_verified", wide
    assert name_row(smiles, tier="pin")["tier"] == "abstain"


# (the Blue Book): "For mononuclear parent hydrides with two or more substituents
# the first cited substituent never has enclosing marks unless it includes a locant. The second and
# further substituents are each enclosed with parentheses even for simple substituents."
TWO_PREFIXES_ON_A_MONONUCLEAR_PARENT = [
    ("O=C(C1CO1)C1CCO1", "(oxetan-2-yl)(oxiranyl)methanone"),
    ("C1CO[C@H]1C(=O)c1ccccc1", "[(2R)-oxetan-2-yl](phenyl)methanone"),
    ("O=C(c1ccccc1)C1CCCO1", "(oxolan-2-yl)(phenyl)methanone"),
    ("OC(c1ccccc1)C1CCO1", "(oxetan-2-yl)(phenyl)methanol"),
    # the first prefix is simple: it stays bare
    ("O=C(C1CC1)C1CCO1", "cyclopropyl(oxetan-2-yl)methanone"),
]


@pytest.mark.parametrize("smiles,pin", TWO_PREFIXES_ON_A_MONONUCLEAR_PARENT)
def test_second_prefix_on_a_mononuclear_parent_is_enclosed(smiles, pin):
    row = name_row(smiles, tier="pin")
    assert (row["name"], row["tier"]) == (pin, "pin_verified"), row


# Parentheses (the Blue Book) and (:7444): the marks go round the whole
# compound prefix; one that opens with marks of its own takes the next mark in the nesting order
@pytest.mark.parametrize("smiles,pin", [
    ("OCC1CC1CN1CC1", "{2-[(aziridin-1-yl)methyl]cyclopropyl}methanol"),
    ("OCC1(CN2CC2)CC1", "{1-[(aziridin-1-yl)methyl]cyclopropyl}methanol"),
])
def test_compound_prefix_on_a_ring_prefix_is_enclosed_as_a_whole(smiles, pin):
    assert name_row(smiles, tier="pin")["name"] == pin
