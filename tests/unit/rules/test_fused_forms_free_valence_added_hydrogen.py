"""A free valence on a ring nitrogen that has no hydrogen in the mancude parent takes added hydrogen
(``rules.fused_forms``, PB lane R).

the Blue Book (:16070) "When no hydrogen atoms are present... it is necessary to use 'added
indicated hydrogen'"; (:24687) "This method is preferred over the use of nondetachable hydro
prefixes for preferred IUPAC names."; (:24695) 'pyridin-1(2H)-yl (preferred prefix)';
'1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one (PIN)' (:7467). A parent that has indicated hydrogen keeps
it ('2,3-dihydro-1H-isoindol-2-yl (preferred prefix)':17380).
"""
import pytest
from rdkit import Chem

from orthonym.assembly import substituent_enumerator as se
from orthonym.rules.fused_forms import fused_system_form

pytestmark = pytest.mark.unit


def _prefix(smiles, parent, attach):
    mol = Chem.MolFromSmiles(smiles)
    cut = Chem.RWMol(mol)
    cut.RemoveBond(parent, attach)
    frag = set(next(f for f in Chem.GetMolFrags(cut) if attach in f))
    core = se._ring_system_core_atoms(mol, attach, frag, mol.GetRingInfo())
    form = fused_system_form(mol, sorted(core), attach, [])
    return None if form is None else form.prefix(attach, 1)


@pytest.mark.parametrize("smiles,parent,attach,expected", [
    ("CC(=O)N1CCCc2ccccc21", 1, 3, "3,4-dihydroquinolin-1(2H)-yl"),                     # no indicated hydrogen
    ("CC(=O)N1CCc2ccccc2C1", 1, 3, "3,4-dihydroisoquinolin-2(1H)-yl"),
    ("CC(=O)N1CC2CCCN2CC1", 1, 3, "hexahydropyrrolo[1,2-a]pyrazin-2(1H)-yl"),
    ("CC(=O)N1CCc2ccccc21", 1, 3, "2,3-dihydro-1H-indol-1-yl"),                          # indicated hydrogen
    ("CC(=O)N1c2ccccc2Sc2ccccc21", 1, 3, "10H-phenothiazin-10-yl"),
    ("CC(=O)n1c2ccccc2c2ccccc21", 1, 3, "9H-carbazol-9-yl"),
    ("CC(=O)C1CCCc2ccccc21", 1, 3, "1,2,3,4-tetrahydronaphthalen-1-yl"),                 # a carbon: unchanged
])
def test_the_prefix_of_a_ring_nitrogen_with_no_hydrogen_in_the_parent(smiles, parent, attach, expected):
    assert _prefix(smiles, parent, attach) == expected
