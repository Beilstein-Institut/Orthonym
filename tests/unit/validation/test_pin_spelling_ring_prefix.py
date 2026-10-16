""" (a): a prefix of the parent hydride methane cites no locant (PB lane R).

the Blue Book "The locant '1' is omitted: (a) in substituted mononuclear parent hydrides"
('chloromethane (PIN)'). The PIN tier's promotion re-run assembled 'N-{1-[1-(...)azetidin-3-yl]methyl}-
1H-indazole-3-carboxamide' for the amide of '1-[1-(...)azetidin-3-yl]methanamine' (the locant of the
methanamine kept in the prefix); the check lowers such a name instead of shipping it as the PIN.
"""
import pytest
from rdkit import Chem

from orthonym.validation import pin_spelling as ps

pytestmark = pytest.mark.unit

MOL = Chem.MolFromSmiles("C")


def _rules(name):
    return [f.rule for f in ps.check_pin_spelling(MOL, name)]


@pytest.mark.parametrize("name", [
    "N-{1-[1-(6-cyclopropylpyridazin-3-yl)azetidin-3-yl]methyl}-1H-indazole-3-carboxamide",
    "1-[1-(pyridin-2-yl)piperidin-4-yl]methyl benzoate",
])
def test_a_methyl_prefix_with_the_locant_one_fails(name):
    assert "P-14.3.4" in _rules(name)


@pytest.mark.parametrize("name", [
    "N-{[1-(6-cyclopropylpyridazin-3-yl)azetidin-3-yl]methyl}-1H-indazole-3-carboxamide",
    "1-[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine",        # the parent keeps its locant (:2869)
    "1-[(4-bromophenyl)methyl]-1H-indole",                         # the locant belongs to the ring
    "1-(13C)methyl(2-13C)benzene",                                 # an isotope descriptor, not a prefix
    "2-[1-(phenylmethoxy)ethyl]pyridine",
])
def test_names_without_the_defect_pass(name):
    assert "P-14.3.4" not in _rules(name)


# --: a ring S with =O is written with its lambda --------------------------------------

@pytest.mark.parametrize("smiles,name,fails", [
    ("OC1CCS(=O)(=O)C1", "1,1-dioxothiolan-3-ol", True),                              # the Blue Book,:33290
    ("CC1C(SCCN1C2CS(=O)(=O)CC2O)C", "4-(2,3-dimethylthiomorpholin-4-yl)-1,1-dioxothiolan-3-ol", True),
    ("OC1CCS(=O)(=O)C1", "3-hydroxy-1λ6-thiolane-1,1-dione", False),                  # the lambda is cited
    ("O=S1(=O)CCCC1", "thiolane 1,1-dioxide", False),                                 # functional class: gold rows
    ("O=C1NS(=O)(=O)c2ccccc21", "saccharin", False),                                  # retained name
    ("CCS(=O)(=O)c1ccccc1", "(ethanesulfonyl)benzene", False),                        # not a ring S
    ("O=C1CCS(=O)(=O)C1", "3-oxo-1λ6-thiolane-1,1-dione", False),
])
def test_ring_chalcogen_oxo_prefix_needs_lambda(smiles, name, fails):
    got = [f.rule for f in ps.check_pin_spelling(Chem.MolFromSmiles(smiles), name)]
    assert ("P-14.1.3" in got) is fails, got
