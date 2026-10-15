"""LO4a class H (second row): an acyl group that carries another group is named as an acyl group.

``substituent_naming._name_polyfunctional_acyclic_substituent_impl`` cited the oxo on the free-valence
carbon of a chain as a detachable prefix, so '-CO-CH(OH)-CH3' on a ring was '2-hydroxy-1-oxopropyl', a
valid name that is never the preferred one: the default tier declined every molecule that carried one,
whatever its ring (the pteridine row of the record, ``CC(O)C(=O)C1=NC2C(=O)NC(N)=NC2NC1``, is one).

 (the Blue Book, heading Acyl groups derived from carboxylic acids with
names retained only for general nomenclature): "substitution on acyl groups is identical to that of
acids... Preferred IUPAC names are systematic substitutive names", with 'propanoyl (preferred prefix)'
and '1-oxopropyl' as the alternative that is not preferred (:30562). Printed substituted acyl prefixes:
'3-chloropropanoyl' (:33117), '2-methylidenebutanoyl' (:31400), 'oxoacetyl' (:30516) and, with the one
possible locant omitted, 'chloroacetyl' ((chloroacetyl)oxyl (PIN),:40694; (:3031)
"All locants are omitted for parent compounds when all substitutable hydrogen atoms have the same
locant").
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PTERIN = "C1=NC2C(=O)NC(N)=NC2NC1"          # the tetrahydropteridinone ring of the record, the acyl on C6
PIN_ROWS = [
    # the item row
    ("CC(O)C(=O)C1=NC2C(=O)NC(N)=NC2NC1",
     "2-amino-6-(2-hydroxypropanoyl)-4a,7,8,8a-tetrahydropteridin-4(3H)-one"),
    ("C[C@H](O)C(=O)C1=NC2C(=O)NC(N)=NC2NC1",
     "2-amino-6-[(2S)-2-hydroxypropanoyl]-4a,7,8,8a-tetrahydropteridin-4(3H)-one"),
    ("OCC(=O)C1=NC2C(=O)NC(N)=NC2NC1", "2-amino-6-(hydroxyacetyl)-4a,7,8,8a-tetrahydropteridin-4(3H)-one"),
    ("ClCC(=O)C1=NC2C(=O)NC(N)=NC2NC1", "2-amino-6-(chloroacetyl)-4a,7,8,8a-tetrahydropteridin-4(3H)-one"),
    ("NCCC(=O)C1=NC2C(=O)NC(N)=NC2NC1",
     "2-amino-6-(3-aminopropanoyl)-4a,7,8,8a-tetrahydropteridin-4(3H)-one"),
    ("CCC(O)C(=O)C1=NC2C(=O)NC(N)=NC2NC1",
     "2-amino-6-(2-hydroxybutanoyl)-4a,7,8,8a-tetrahydropteridin-4(3H)-one"),
    # the other LO3 row: its builder name carried the acyl too (so it was not offered) and the older
    # namer's descriptor spelling of pteridine ('pyrazino[2,3-d]pyrimidine') was not recorded as non-PIN
    # (fused_rings._retained_name_of_skeleton read only PIN_PARENT_NAMES; Table 2.8 (7) 'pteridine (PIN)',
    #:11547)
    ("CC(O)C(=O)C1=Nc2c([nH]c(=O)[nH]c2=O)NC1",
     "6-(2-hydroxypropanoyl)-7,8-dihydropteridine-2,4(1H,3H)-dione"),
    ("Cc1cn(CCO)c2nc(=O)[nH]c(=O)c-2n1", "8-(2-hydroxyethyl)-6-methylpteridine-2,4(3H,8H)-dione"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_acyl_group_with_a_second_group_is_named_as_an_acyl_group(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def _acyl_fragment_name(smiles):
    """The fragment namer's name of the acyl group that hangs on an aromatic carbon or a ring nitrogen."""
    from rdkit import Chem

    from orthonym.assembly.substituent_naming import name_substituent_fragment
    mol = Chem.MolFromSmiles(smiles)
    x, carbonyl = mol.GetSubstructMatch(Chem.MolFromSmarts("[c,N;R][CX3;!R](=O)[#6]"))[:2]
    cut = Chem.RWMol(mol)
    cut.RemoveBond(x, carbonyl)
    frag = next(set(f) for f in Chem.GetMolFrags(cut) if carbonyl in f)
    parent = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in frag]
    return name_substituent_fragment(mol, sorted(frag), carbonyl, parent)


@pytest.mark.parametrize("smiles,expected", [
    ("CC(O)C(=O)c1ccccc1", "2-hydroxypropanoyl"),
    ("C[C@H](O)C(=O)c1ccccc1", "(2S)-2-hydroxypropanoyl"),
    ("NCCC(=O)c1ccccc1", "3-aminopropanoyl"),
    ("OCC(O)C(=O)c1ccccc1", "2,3-dihydroxypropanoyl"),
    ("OCCCC(=O)c1ccccc1", "4-hydroxybutanoyl"),
    ("OC(=O)CCC(=O)c1ccccc1", "3-carboxypropanoyl"),
    ("CC(=O)C(=O)c1ccccc1", "2-oxopropanoyl"),
    # two carbons: acetyl, one kind of prefix, no locant
    ("ClCC(=O)c1ccccc1", "chloroacetyl"),
    ("ClC(Cl)C(=O)c1ccccc1", "dichloroacetyl"),
    ("O=CC(=O)c1ccccc1", "oxoacetyl"),
    # the saturated chain with no second group is the acyl tier's (unchanged): 'propanoyl'
    ("CCC(=O)c1ccccc1", "propanoyl"),
])
def test_the_fragment_namer_gives_the_acyl_name(smiles, expected):
    assert _acyl_fragment_name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # two different prefixes on the one carbon of an acetyl group need the enclosing marks of
    #: the older spelling stays (a recorded non-PIN name, so the default tier declines)
    ("OC(Cl)C(=O)c1ccccc1", "2-chloro-2-hydroxy-1-oxoethyl"),
    # an acyl group on a nitrogen is an acylamino prefix, whose spelling (amido or acylamino,
    # is not chosen here
    ("CC(O)C(=O)N1CCCC1", "2-hydroxy-1-oxopropyl"),
])
def test_the_older_spelling_stays(smiles, expected):
    assert _acyl_fragment_name(smiles) == expected


def test_the_acyl_name_is_not_recorded_as_non_pin():
    """The wrapper records a name that cites an oxo on the free-valence carbon as non-PIN; an acyl name that
    contains an oxo elsewhere ('5-oxopentanoyl', an acyloxy group of a phospholipid) is not that spelling."""
    from orthonym.assembly.acyl_prefix import is_acyl_prefix
    assert is_acyl_prefix("5-oxopentanoyl") and is_acyl_prefix("oxoacetyl") and is_acyl_prefix("chloroacetyl")
    assert not is_acyl_prefix("2-hydroxy-1-oxopropyl") and not is_acyl_prefix("3-oxopropyl")
    from tests.support.pin_tiers import name_default
    smiles = "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)OC(=O)CCCC=O"
    d = name_default(smiles)
    assert (d.get("tier"), d.get("name")) == (
        "pin_verified", "2-[({(2R)-3-(hexadecanoyloxy)-2-[(5-oxopentanoyl)oxy]propoxy}hydroxyphosphoryl)oxy]"
                        "-N,N,N-trimethylethan-1-aminium"), d
