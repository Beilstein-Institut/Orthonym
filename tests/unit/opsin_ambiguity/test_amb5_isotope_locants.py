"""An isotope descriptor cites its locant when more than one atom could carry the label
,.

 Omission of locants (the Blue Book), (:44180): "In preferred IUPAC
names, locants are omitted if no locants are necessary in unmodified names. However, if isotopic
modification requires a locant to specify its position, then all locants must be specified and
none are omitted."
"""
import pytest

from ._helpers import assert_clean, name_row

# the names are the shipped ones: the OPSIN validity gate is on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,locant_descriptor", [
    ("C1C(=O)NC(=O)[15NH]1", "(1-15N)"),                  # two N in the ring
    ("O=c1cc[nH][13c](=O)[nH]1", "(2-13C)"),              # C-2 between two N, four C in the ring
    ("CC1=C[15N+](=C(C(=N1)C)C)[O-]", "(1-15N)"),         # two N in the ring
    ("[2H]C1N(CC(CN1C)C2=CC=CC=C2)C", "(2-2H1)"),         # C-2 between two N
])
def test_isotope_in_a_ring_with_several_atoms_of_the_element_cites_its_locant(smiles,
                                                                              locant_descriptor):
    # (the Blue Book): "Locants are omitted when there is only one atom of a
    # given element." The ring holds several atoms of the labelled element, so the locant is
    # cited,:44202: "Locants are not omitted when there is a possibility of isomers").
    for tier in ("best-effort", "pin"):
        row = name_row(smiles, tier)
        if tier == "best-effort" or row["tier"] not in (None, "abstain"):
            assert locant_descriptor in row["name"], row
    assert_clean(smiles, tier="best-effort")


@pytest.mark.parametrize("smiles", [
    "C1C(=O)NC(=O)[15NH]1",
    "O=c1cc[nH][13c](=O)[nH]1",
    "CC1=C[15N+](=C(C(=N1)C)C)[O-]",
    "[2H]C1N(CC(CN1C)C2=CC=CC=C2)C",
])
def test_isotope_names_are_clean(smiles):
    assert_clean(smiles, tier="pin")


@pytest.mark.parametrize("smiles,name", [
    ("[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]", "(2H6)benzene"),   #,:44196
    ("[2H]C([2H])([2H])C#N", "(2H3)acetonitrile"),                  # example,:44182
    ("[15n]1ccccc1", "(15N)pyridine"),                              # example,:44194
])
def test_book_omissions_stay(smiles, name):
    assert name_row(smiles, tier="pin")["name"] == name


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_labelled_ring_prefix_cites_the_isotope_locant(tier):
    # Letter and/or numeral locants (the Blue Book),:44210: "When locants are
    # needed for defining the structure of the parent structure or of a unit of structure as
    # defined by its appropriate enclosing marks, then all locants must be cited for the
    # parent structure or that structural unit." The unit '[(...)aziridin-1-yl]' cites the
    # locant 1, so its descriptor cites the locant 2. OPSIN raises no warning on the name
    # '[(13C)aziridin-1-yl]acetic acid', so the census cannot see this row.
    assert name_row("OC(=O)CN1C[13CH2]1", tier)["name"] == "[(2-13C)aziridin-1-yl]acetic acid"


@pytest.mark.parametrize("smiles,name", [
    ("[3H]N1C=CC(=N1)OC", "3-methoxy(1-3H)-1H-pyrazole"),
    ("[2H]N1CCCCC1", "(1-2H)piperidine"),
    ("[2H][15N]1C=CC=C1", "(1-2H,1-15N)-1H-pyrrole"),
])
def test_hydrogen_isotope_on_a_ring_nitrogen_cites_the_ring_locant(smiles, name):
    # (the Blue Book): "Locants are not omitted when there is a possibility of
    # isomers"; the hydrogen of the ring nitrogen is one of the hydrogens of the ring, and OPSIN
    # places a locant-free (3H) on a different atom of '1H-pyrazole' (a warning), so the locant
    # of the ring nitrogen is cited
    assert name_row(smiles, tier="pin")["name"] == name


@pytest.mark.parametrize("smiles,name", [
    ("[2H]C([2H])(c1ccccc1)n1ccnc1", "1-[phenyl(2H2)methyl]-1H-imidazole"),
    ("[2H]C(c1ccccc1)OCC(=O)O", "[phenyl(2H1)methoxy]acetic acid"),
])
def test_label_on_the_benzylic_carbon_is_cited_on_phenylmethyl(smiles, name):
    # (the Blue Book): "When the nuclide is located at a position in a retained
    # name that is not numbered a systematic name that identifies separately the relevant atom is
    # used for the IUPAC preferred name"; '[phenyl(13C)methyl]hydrazine (PIN)' (:44259). A
    # deuterium or a tritium is such a nuclide as a 13C is
    assert name_row(smiles, tier="pin")["name"] == name


@pytest.mark.parametrize("smiles", [
    "[3H]N1C=CC(=N1)OCC2=C(C=CN2)N3C(=O)N(N=N3)C",
    "[2H]C1=NC2=NNC(=C2C(=N1)N)C3=CC=C(C=C3)OC4=CC=CC=C4",
    "[2H]N1CCC2(CC1CC3=CC=CC=C3)CN(C4=NN=C(C=C24)C5=CC=CC=C5O)CC6=C(C=C(C=C6)OC)OC",
    "[2H][15N]1C(C(=C(C1=O)CC)C)S(=O)(=O)C2=CC=C(C=C2)C",
    "[2H]C([2H])(C1=CC=CC=C1)C2=NC3=C(C=C2)C4=C(O3)C(=CC=C4)C5=NC6=CC=CC=C6N5CC7=C(C=C(C=C7C(C)C)C)C(C)C",
    "[2H]C1(OC2=C(O1)C3=C(C=C2)SC(=N3)N4CNC[C@H]4C#CC)[2H]",
    "[2H]C1=[N+](C(=C2N1C=CC2(C3=CC=CC=C3)C4=CC=CC=C4)C)C5=CC=CC=C5C",
    "CCOC(=O)C1=CN2C(=C1C)C(=O)N[14CH]=N2",
    "C1=CC=C(C(=C1)[C@@H]2[C@@](O2)(CN3C=N[14CH]=N3)C4=CC=C(C=C4)F)Cl",
])
def test_isotope_names_of_larger_structures_are_clean(smiles):
    # and (see above): the descriptor cites the locant of its position
    assert_clean(smiles, tier="best-effort")
