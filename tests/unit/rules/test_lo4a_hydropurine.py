"""LO4a: the hydropurine parent reader.

Two defects kept every hydropurine from a name at the default tier.

1. ``bridged_fused_pin.parents`` could not read purine at all (a purine skeleton named by no table):
   purine has the traditional numbering, (the Blue Book) "Anthracene, phenanthrene,
   acridine, carbazole, xanthene and its chalcogen analogues, purine, and cyclopenta[a]phenanthrene are
   exceptions; traditional numberings are retained", with the fusion carbon atoms numbered 4 and 5, not 4a
   and 5a (Table 2.8 (16) purine, ":11593-:11595 (special numbering, 7H-isomer shown; the PIN is
   7H-purine)"). The set of parents whose fusion carbon atoms carry plain numbers
   (``parents.TRADITIONAL_NUMBERING``) did not list it, so the lettered locants the fusion numbering
   gives disagreed with the table's and the parent was dropped.
2. ``purine.name_substituted_purine`` matched the purine skeleton with any-bond SMARTS and wrote the
   MANCUDE parent for any state: ``NC1=NC(N)c2[nH]cnc2N1`` (sp3 C6) came out as '7H-purine-2,6-diamine',
   the hydro prefixes dropped, a different molecule, which the self-consistency gate then rejected, so
   the default tier abstained. (the Blue Book) "a mancude ring or ring system, i.e., one
   that contains the maximum number of noncumulative double bonds": the parent written there is the
   mancude one, so the namer declines a core that is not.

The hydro state is then named by the builder: SUBSTITUENT GROUPS MODIFIED BY THE PREFIXES
'HYDRO' OR 'DEHYDRO' (:16870),:16876 "lowest locants to fixed numbering of a polycyclic ring system,
... then to heteroatoms in heterocycles, and then to indicated hydrogen" (the hydro prefixes stand in
front of the parent, with the lowest indicated hydrogen: '6,7-dihydro-3H-purine-2,6-diamine').
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)

pytestmark = pytest.mark.opsin_gate

HYDRO_ROWS = [
    # the item row (fu4 record section 6, 'the parent reader for a hydropurine')
    ("NC1=NC(N)c2[nH]cnc2N1", "6,7-dihydro-3H-purine-2,6-diamine"),
    ("NC1=NC(N)c2nc[nH]c2N1", "6,9-dihydro-3H-purine-2,6-diamine"),
    ("NC1=NC2=C(N=CN2)C(N)N1", "6,9-dihydro-1H-purine-2,6-diamine"),
]
# the mancude purines keep their names (the older namer, which this change does not reach)
MANCUDE_ROWS = [
    ("Nc1ncnc2[nH]cnc12", "9H-purin-6-amine"),
    ("Nc1ncnc2c1ncn2C", "9-methyl-9H-purin-6-amine"),
    ("Nc1nc2[nH]cnc2c(N)n1", "9H-purine-2,6-diamine"),
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
    ("O=c1[nH]c(N)nc2[nH]cnc12", "2-amino-1,9-dihydro-6H-purin-6-one"),
]


@pytest.mark.parametrize("smiles,pin", HYDRO_ROWS)
def test_hydropurine_is_named_by_the_builder(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
    assert name_default(smiles).get("producer") == "inner:tier_a_ring"


@pytest.mark.parametrize("smiles,pin", MANCUDE_ROWS)
def test_mancude_purine_names_are_unchanged(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_purine_is_a_parent_with_the_traditional_numbering():
    from orthonym.rules.bridged_fused_pin import parents, selection
    assert "purine" in parents.TRADITIONAL_NUMBERING
    mol = Chem.MolFromSmiles("NC1=NC(N)c2[nH]cnc2N1")
    found = parents.fused_parent(mol, selection.ring_system(mol))
    assert found is not None and found.name == "purine"
    # N1 C2 N3 C4 C5 C6 N7 C8 N9: the fusion carbons carry the plain numbers 4 and 5
    for numbering in found.numberings:
        assert {loc for loc in numbering.values() if isinstance(loc, str)} == set()
        assert sorted(numbering.values()) == list(range(1, 10))


def test_builder_names_the_hydro_state_and_the_nucleoside_like_n9_substituent():
    from orthonym.rules.bridged_fused_pin import build_fused
    got = build_fused(Chem.MolFromSmiles("NC1=NC(N)c2[nH]cnc2N1"))
    assert got is not None and got[0] == "6,7-dihydro-3H-purine-2,6-diamine", got
    got = build_fused(Chem.MolFromSmiles("C=C1NC(N)=Nc2c1ncn2C"))
    assert got is not None and got[0] == "9-methyl-6-methylidene-6,9-dihydro-1H-purin-2-amine", got


def test_the_older_purine_namer_declines_a_hydro_state():
    """The any-bond SMARTS matched the hydro state and the namer wrote the mancude parent."""
    from orthonym.rules.purine import name_substituted_purine
    assert name_substituted_purine(Chem.MolFromSmiles("NC1=NC(N)c2[nH]cnc2N1")) is None
    assert name_substituted_purine(Chem.MolFromSmiles("C=C1NC(N)=Nc2c1ncn2C")) is None
    assert name_substituted_purine(Chem.MolFromSmiles("Nc1ncnc2[nH]cnc12")) == "9H-purin-6-amine"


def test_a_hydropurine_nucleotide_is_not_named_at_the_default_tier():
    """The second hydropurine row of the record is a 9-substituted nucleoside triphosphate: its principal
    group is the phosphate, a group the builder does not spell, so the default tier keeps declining it
    (best-effort keeps its systematic name). Pinned so that the decline is a decision, not an accident."""
    smiles = "C=C1NC(N)=Nc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)OP(=O)(O)O)C(O)C1O"
    assert name_default(smiles).get("tier") != "pin_verified"
    b = name_breadth(smiles)
    assert b.get("name") and b.get("tier") in ("systematic_verified", "pin_unverified"), b
