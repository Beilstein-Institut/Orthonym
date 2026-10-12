"""A diazonium group that is not the suffix of its parent is the prefix 'diazonio'.

 CATIONIC PREFIX NAMES (the Blue Book), example:42318: 'diazyn-1-ium-1-yl
(preselected prefix) diazonio'. The prefix cites its locant like every prefix,
the Blue Book).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.universal_substituent import (
    _diazo_nitrogens,
    _diazonium_nitrogens,
    name_universal_substitutive,
)

from ._helpers import name_row

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("N#[N+]c1c[nH]c(=O)[nH]c1=O", "5-diazonio-2,4-dioxo-1,2,3,4-tetrahydropyrimidine"),
    ("N#[N+]c1ccc2ccccc2c1", "2-diazonionaphthalene"),
    ("N#[N+]c1ccccc1", "diazoniobenzene"),
])
def test_universal_engine_cites_the_diazonio_prefix_at_its_locant(smiles, name):
    assert name_universal_substitutive(Chem.MolFromSmiles(smiles)).name == name


@pytest.mark.parametrize("smiles", ["N#[N+]c1ccccc1", "C[N+]#N", "CC(=O)C(C(C)=O)[N+]#N"])
def test_diazonium_suffix_names_stay(smiles):
    # (the Blue Book,:41615): 'methanediazonium (PIN)',
    # '2,4-dioxopentane-3-diazonium (PIN)'; benzene has one kind of position
    name = name_row(smiles, tier="pin")["name"]
    assert name.endswith("diazonium"), name


@pytest.mark.parametrize("smiles", ["N#[N+]c1ccccc1", "C[N+]#N"])
def test_diazonium_is_not_a_diazo_group(smiles):
    # a diazo twin R(-)-[N+]#N carries a carbanion; R-[N+]#N alone is the cation
    mol = Chem.MolFromSmiles(smiles)
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms()
                    if a.GetSymbol() == "N" and a.GetFormalCharge() == 1)
    assert _diazo_nitrogens(mol, nitrogen) is None
    assert _diazonium_nitrogens(mol, nitrogen) is not None


@pytest.mark.parametrize("smiles,name", [
    ("C=[N+]=[N-]", "diazomethane"),
    ("CC(=O)C(C(C)=O)=[N+]=[N-]", "3-diazopentane-2,4-dione"),
])
def test_diazo_group_stays_the_diazo_prefix(smiles, name):
    # DIAZO COMPOUNDS (the Blue Book),:25919: "Compounds containing a group =N2
    # attached to a single carbon atom are named by adding the prefix 'diazo'"; 'diazomethane (PIN)'
    assert name_row(smiles, tier="pin")["name"] == name
