"""A chain prefix cites the locant of its free valence, (2)).

 Substituent groups derived from parent hydrides with 'ene' or 'yne' endings,
(the Blue Book): "Accordingly, for acyclic parent structures, all locants for the free
valences, including '1', must be cited."
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
from orthonym.jvm_bridge import opsin_stdout

from ._helpers import assert_clean, name_row

# the names are the shipped ones: the OPSIN validity gate is on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate


def _same_radical_graph(smiles: str, name: str) -> bool:
    """OPSIN's parse of ``name`` has the canonical SMILES of ``smiles``, radical electrons
    included (the full InChIKey does not tell a radical from its closed-shell twin)."""
    text, served = opsin_stdout(name, allow_radicals=True)
    assert served, "in-process OPSIN not available"
    a, b = Chem.MolFromSmiles(smiles), Chem.MolFromSmiles(text.strip())
    return a is not None and b is not None and Chem.MolToSmiles(a) == Chem.MolToSmiles(b)


@pytest.mark.parametrize("parent,n", [
    ("3-ethoxy-1,1-difluoroprop-1-ene", 3),
    ("1-chloro-3-methylbut-1-yne", 4),
    ("1,1-dichloro-3,3-dimethylhexa-1,5-diene", 6),
    ("1-methoxy-1-oxopent-4-yne", 5),
    ("but-2-ene", 4),
    ("(4Z)-4-oxido-1,6-dioxa-4,5-diazahepta-1,4-dien-4-ium", 0),
])
def test_converter_declines_a_name_that_carries_locants(parent, n):
    #, the Blue Book (see above): the locants of a parent name belong to the
    # numbering of the capped molecule, which does not number the free valence, so a string
    # converter has no free-valence locant to cite and declines
    assert not parent_to_prefix(parent, n, attach_locant=ATTACH_LOCANT_UNKNOWN)


@pytest.mark.parametrize("parent,n,prefix", [
    ("ethene", 2, "ethenyl"),
    ("ethyne", 2, "ethynyl"),
])
def test_converter_keeps_the_two_atom_stems(parent, n, prefix):
    # Retained names for substituent groups derived from unsaturated acyclic parent
    # hydrides (the Blue Book): "The systematic names ethenyl, ethenylidene,
    # prop-2-en-1-yl, prop-2-en-1-ylidene, and prop-2-en-1-ylidyne, respectively, are the
    # preferred IUPAC names."
    assert parent_to_prefix(parent, n, attach_locant=ATTACH_LOCANT_UNKNOWN) == prefix


@pytest.mark.parametrize("parent,n,prefix", [
    ("(4-methylphenyl)diazene", 7, "(4-methylphenyl)diazenyl"),
    ("(2-chlorophenyl)diazene", 6, "(2-chlorophenyl)diazenyl"),
])
def test_converter_keeps_a_name_whose_locants_are_inside_enclosing_marks(parent, n, prefix):
    # Substitution of diazene (the Blue Book): "The prefix 'diazenyl' is a
    # preselected prefix"; '(methyldiazenyl)acetic acid (PIN)' (:38740). The locants of a nested
    # prefix are complete in its own numbering; the parent 'diazene' cites none.
    assert parent_to_prefix(parent, n, attach_locant=ATTACH_LOCANT_UNKNOWN) == prefix


@pytest.mark.parametrize("smiles", [
    "CCOCC(=C(F)F)c1ccccc1F",
    "CC(=C(F)F)n1c(=O)n(CC=C(F)F)c(=O)n(CC=C(F)F)c1=O",
    "C=CCC(C)(C)C(OP(=O)(O)O)=C(Cl)Cl",
    "CCCCC(=C(F)F)c1cc(S(=O)(=O)Nc2ccccc2)ccc1C",
    "CC=C(C(F)(F)F)NC1=NC(=NC(=N1)NC2CC3(C2)COC3)C4=C(C(CCC4)O)F",
    "CC(C)(C#CCl)Nc1ccccc1",
    "CC(C)(C#CO)c1ccc(OCCO)o1",
    "C=C(C(=N)OC)N1CCNCC1",
    "C#CC[C@@H](NC(=O)C1=C[C@H](O)[C@H](NC(=O)C2(C3(O)CCC3)CC2)C1)C(=O)OC",
    "C=C(C)C(C(=O)OOC(N)=O)c1ccccc1C(F)(F)F",
    "CC(C#CS(C)(C)C)c1ccc(OC2CCCCO2)cc1",
    "CO/N=[N+](\\[O-])CC(=O)N1CCN(S(=O)(=O)c2ccccc2)C1",
])
def test_unsaturated_chain_prefix_is_clean(smiles):
    #, the Blue Book (see above): the prefix is named from the structure, with
    # the locant of its free valence
    assert_clean(smiles)


@pytest.mark.parametrize("smiles", [
    "CC(C=C(Cl)Cl)=NNC(N)=O",
    "C/C(=C\\1/C=CC=CN1C(=C)C(=NCC(F)F)NN=C)/CC2=CC=CC=C2F",
])
def test_ylidene_chain_prefix_is_clean(smiles):
    #, the Blue Book (see above): the same rule governs the 'ylidene' prefix
    # of an unsaturated chain
    assert_clean(smiles)


@pytest.mark.parametrize("smiles", [
    "CC(C)CC(CCc1ccccc1)OS(N)(=O)=O",
    "CN(C)/C=N/S(=O)(=O)c1cc(F)c(CO)c(F)c1",
    "CON(C)S(=O)(=O)c1ccc(C(=O)NCc2ccccc2)cc1",
    "CNS(=O)C(c1ccccc1)(c1ccc(F)cc1)c1ccc(F)cc1",
])
def test_chain_prefix_with_a_heteroatom_spine_is_clean(smiles):
    #, the Blue Book (see above): a prefix whose chain carries skeletal
    # heteroatoms is named by the same rule
    assert_clean(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("C=CN1CCOCC1", "4-ethenylmorpholine"),
    ("C#CN1CCOCC1", "4-ethynylmorpholine"),
    ("CC=CN1CCOCC1", "4-(prop-1-en-1-yl)morpholine"),
])
def test_cited_free_valence_locants_stay(smiles, name):
    # (the Blue Book, see above): 'ethenyl' and 'ethynyl' are preferred prefixes;
    # (:17207, see above): 'prop-1-en-1-yl' cites the locant of its free valence
    assert name_row(smiles, tier="pin")["name"] == name


@pytest.mark.parametrize("smiles", [
    "[B]1C2=CC=CC=C2C(=O)OC1=O",
    "[B]1C2=CC=CC=C2C(=O)C3=CC=CC=C31",
    "[B]1CC([C@@H]2N1CCC2)(C3=CC=CC=C3)C4=CC=CC=C4",
    "[B]1C(=O)CC2=C(N1O)C=CC=N2",
    "[B]1C2=C(CC=C2)C=C1COOCC",
    "[B]1C2=C(C=CC(=C2)C(=O)NN)C=NO1",
    "C1=CC=C(C=C1)[Si]2C3=CC=CC=C3C(=O)C4=CC=CC=C42",
    "C1C2=CC=CC=C2CC3=CC=CC=C3[Sb]C4=CC=CC=C4CC5=CC=CC=C51",
    "C1[C]C2CC1C2",
    "CC1=CC=CC=C1N2[C]C3(CCC2(CC3)C)C",
    "CC(C)C1=C(C(=CC=C1)C(C)C)N2CC3CCCN(C3)[C]2",
    "[B]1C=CC(=O)OO1",
    "[B]1N(C(=CO1)O)S(=O)(=O)C2=CC=C(C=C2)C",
    "C1C(=O)NC[Si]O1",
    "[B]1C=CC=CC=C1COPC2=CC=CC(=C2)OC",
])
def test_ring_radical_name_is_clean(smiles):
    # Monovalent radicals, (the Blue Book): "A radical formally derived
    # by the removal of one hydrogen atom from any position of a parent hydride... is named by
    # adding the suffix 'yl' to the name of the parent hydride"; the position of the free
    # valence is cited General methodology for naming substituent groups,:15814: "the
    # locant '1' must be cited")
    row = assert_clean(smiles)
    assert _same_radical_graph(smiles, row["name"]), row["name"]


@pytest.mark.parametrize("smiles,name", [
    ("[CH2]C", "ethyl"),
    ("[CH2]CC", "propyl"),
    ("[CH]1CCCCC1", "cyclohexyl"),
    ("[CH3]", "methyl"),
])
def test_contracted_radical_name_stays(smiles, name):
    # Monovalent radicals, (the Blue Book): "A radical formally derived
    # by the removal of one hydrogen atom from a mononuclear parent hydride of an element of Group
    # 14, from a terminal atom of an unbranched acyclic hydrocarbon, or from any position of a
    # monocyclic saturated hydrocarbon ring is named by replacing the 'ane' ending of the
    # systematic name of the parent hydride by 'yl'."
    assert name_row(smiles, tier="pin")["name"] == name


@pytest.mark.parametrize("parent", [
    "(2R)-2-(2,5-difluorophenyl)pyrrolidine",
    "2-methylpiperidine",
])
def test_converter_declines_a_ring_name_that_carries_locants(parent):
    # General methodology for naming substituent groups, method (2) (the Blue Book):
    # "the locant '1' must be cited"; a ring name that cites locants numbers the capped
    # molecule, so its ending is not edited into the prefix
    assert not parent_to_prefix(parent, 5, attach_locant=ATTACH_LOCANT_UNKNOWN)


def test_ring_prefix_of_a_substituted_ring_cites_its_free_valence():
    # (2) (the Blue Book): the nitrogen of the pyrrolidine carries the free valence
    smi = "C[C@]1(C[C@H]1N2CCC[C@@H]2C3=C(C=CC(=C3)F)F)S(=O)(=O)NC(=O)C4=C5C=CC=CN5N=C4"
    assert_clean(smi, must_contain="pyrrolidin-1-yl")
