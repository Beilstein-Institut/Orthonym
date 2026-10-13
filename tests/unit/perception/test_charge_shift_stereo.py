"""The configured stereo elements of an input that each charge form of its species keeps.

A form of a delocalised cation is written from a Kekule structure whose double bonds are not the drawing's. A bond
that is single in it carries no E/Z, and the standard InChIKey that admits a form (``accepted_charge_forms``) does
not see the bonds of a polymethine, so the key cannot say that a form lost them: the choice among the forms'
names took the form that named best by its other criteria and shipped names with no stereodescriptor where the
input had four (rh-forms review, rows r1863, r2090, r1026, r643, r1117 of the 333 multi-form rows of the union).
``charge_shift.kept_stereo`` counts, on each form's own mol and against the drawn input, the elements that are
configured in both with the same configuration; ``assembly.form_choice.beats`` ranks that count first.

 STERODESCRIPTORS (the Blue Book), "The configuration of an organic compound is systematically indicated
by one or more affixes added to a name that does not itself prescribe configuration; such affixes are called
'stereodescriptors'" (:44570); Omission of stereodescriptors (:44635), "From nine-membered rings onwards,
stereodescriptors are required" (:44637).
"""
import random

import pytest
from rdkit import Chem

from orthonym import namer
from orthonym.perception.centres_bridge import _smiles_output_order
from orthonym.perception.charge_shift import kept_stereo, stereo_elements

pytestmark = pytest.mark.unit

#: a vinamidinium, the smallest polymethine: the +1 sits on one of the two nitrogen atoms, and in the other form
#: the double bonds are N+=C and C=C one place along, so the C=C of the drawing is a single bond
TRANS = "CN(C)/C=C/C=[N+](C)C"
CIS = "CN(C)/C=C\\C=[N+](C)C"
#: the same with a tetrahedral centre on the side chain
CHIRAL = "C[C@@H](O)CN(C)/C=C/C=[N+](C)C"


def _written(smiles):
    """(the string RDKit writes for ``smiles`` without canonical ordering, the atom order of that string): the
    form of the same drawing that is no longer in the input's atom order."""
    mol = Chem.MolFromSmiles(smiles)
    string = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
    return string, tuple(_smiles_output_order(mol))


@pytest.mark.parametrize("smiles", [TRANS, CIS])
def test_the_form_with_the_shifted_charge_keeps_none_of_the_polymethine_bond_and_the_drawing_keeps_it(smiles):
    forms = namer._charge_forms(smiles)
    assert forms is not None and len(forms) == 2
    kept = kept_stereo(Chem.MolFromSmiles(smiles), forms)
    drawn = next(k for k, (_string, order) in enumerate(forms) if order is None)
    assert kept[drawn] == 1 and kept[1 - drawn] == 0, (forms, kept)


def test_the_two_drawings_of_one_species_count_the_same_elements():
    # the elements are keyed by the atoms of the input, so the count of a form does not depend on the string
    # that draws it
    assert len(stereo_elements(Chem.MolFromSmiles(TRANS))) == len(stereo_elements(Chem.MolFromSmiles(CIS))) == 1
    assert len(stereo_elements(Chem.MolFromSmiles(CHIRAL))) == 2


def test_a_chiral_centre_is_kept_by_every_form_and_only_the_bond_decides():
    forms = namer._charge_forms(CHIRAL)
    assert forms is not None and len(forms) == 2
    elements = stereo_elements(Chem.MolFromSmiles(CHIRAL))
    assert sorted(kind for kind, _atoms in elements) == ['=', '@']
    kept = kept_stereo(Chem.MolFromSmiles(CHIRAL), forms)
    drawn = next(k for k, (_string, order) in enumerate(forms) if order is None)
    assert kept[drawn] == 2 and kept[1 - drawn] == 1, kept


@pytest.mark.parametrize("smiles", [
    "C[C@H](N)/C=C/C(Cl)=C(\\F)Br",
    "C[C@@H]1CC[C@H](C)C[C@H]1/C=C\\C(=O)N[C@@H](C)c1ccccc1",
    "F/C=C/[C@H](Cl)[C@@H](Br)/C=C\\I",
])
def test_the_elements_do_not_depend_on_how_a_string_numbers_and_writes_its_atoms(smiles):
    # the property that makes a comparison of two forms' mols meaningful: a random writing of the same molecule,
    # read back and keyed through its atom order, has the same elements with the same configurations
    random.seed(7)
    mol = Chem.MolFromSmiles(smiles)
    wanted = stereo_elements(mol)
    assert len(wanted) >= 3
    for _ in range(40):
        string = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        order = tuple(_smiles_output_order(mol))
        assert stereo_elements(Chem.MolFromSmiles(string), order) == wanted, string


def test_a_form_that_has_the_bond_with_the_other_configuration_does_not_keep_it():
    # the geometry is compared, not only that a configuration is present
    trans, cis = Chem.MolFromSmiles("F/C=C/Cl"), Chem.MolFromSmiles("F/C=C\\Cl")
    same = [("F/C=C/Cl", None)]
    flipped = [("F/C=C\\Cl", None)]
    assert kept_stereo(trans, same) == (1,) and kept_stereo(trans, flipped) == (0,)
    assert kept_stereo(cis, flipped) == (1,) and kept_stereo(cis, same) == (0,)


def test_a_form_that_has_the_centre_with_the_other_configuration_does_not_keep_it():
    a, b = Chem.MolFromSmiles("C[C@H](N)O"), Chem.MolFromSmiles("C[C@@H](N)O")
    assert kept_stereo(a, [("C[C@H](N)O", None)]) == (1,) and kept_stereo(a, [("C[C@@H](N)O", None)]) == (0,)
    assert kept_stereo(b, [("C[C@@H](N)O", None)]) == (1,)
    # the same configuration written from another atom keeps it
    string, order = _written("C[C@H](N)O")
    assert kept_stereo(a, [(string, order)]) == (1,)


def test_a_molecule_without_stereo_has_nothing_to_keep():
    mol = Chem.MolFromSmiles("Cc1c[nH]c[nH+]1")
    assert stereo_elements(mol) == {}
    forms = namer._charge_forms("Cc1c[nH]c[nH+]1")
    assert forms is not None and kept_stereo(mol, forms) == (0, 0)


def test_a_form_string_that_cannot_be_read_is_not_counted():
    assert kept_stereo(Chem.MolFromSmiles(TRANS), [("not a smiles ((", None)]) is None
