"""The resonance-supplier trace sees each of the three sites, and nothing else (the instrument of the resonance tests)."""
import pytest
from rdkit import Chem

from orthonym import namer
from orthonym.rules import ions
from orthonym.validation import protonation_identity as pi
from tests.support.resonance_spy import supplier_calls

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles", ["Cc1c[nH]c[nH+]1", "CCCCn1cc[n+](C)c1"])
def test_spy_sees_site_1(smiles):
    namer._charge_forms.cache_clear()
    with supplier_calls() as calls:
        namer._charge_forms(smiles)
    assert [c[0] for c in calls] == ["_charge_forms"]


def test_spy_sees_nothing_for_a_neutral_molecule():
    namer._charge_forms.cache_clear()
    with supplier_calls() as calls:
        namer._charge_forms("CCO")
    assert calls == []


@pytest.mark.parametrize("smiles", ["C[N+]1=CNc2ccccc21", "C[n+]1cc[nH]c1"])
def test_spy_sees_site_2(smiles):
    mol = Chem.MolFromSmiles(smiles)
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    with supplier_calls() as calls:
        ions._ring_cation_hydron_site(mol, ring, 1)
    assert [c[0] for c in calls] == ["_ring_cation_hydron_site"]


@pytest.mark.parametrize("smiles", ["C[n+]1c[nH]c2ccccc21", "C[N+]1=CNc2ccccc21"])
def test_spy_sees_site_3(smiles):
    with supplier_calls() as calls:
        pi._analogue_inchis(Chem.MolFromSmiles(smiles))
    assert [c[0] for c in calls] == ["_analogue_inchis"]


def test_spy_reports_heavy_atoms_and_restores_the_supplier():
    from rdkit.Chem import rdchem
    before = (Chem.ResonanceMolSupplier, rdchem.ResonanceMolSupplier)
    with supplier_calls() as calls:
        pi._analogue_inchis(Chem.MolFromSmiles("C[n+]1c[nH]c2ccccc21"))
    assert calls == [("_analogue_inchis", 10)]
    assert (Chem.ResonanceMolSupplier, rdchem.ResonanceMolSupplier) == before


def test_spy_restores_the_supplier_after_an_error():
    from rdkit.Chem import rdchem
    before = rdchem.ResonanceMolSupplier
    with pytest.raises(ZeroDivisionError):
        with supplier_calls():
            1 / 0
    assert rdchem.ResonanceMolSupplier is before


@pytest.mark.parametrize("smiles", ["Cc1c[nH]c[nH+]1", "CCCCn1cc[n+](C)c1"])
def test_a_forbidden_supplier_is_recorded_and_not_built(smiles):
    namer._charge_forms.cache_clear()
    with supplier_calls(forbid=True) as calls:
        assert namer._charge_forms(smiles) is None          # the site's own error handling swallows the refusal
    assert [c[0] for c in calls] == ["_charge_forms"]
    namer._charge_forms.cache_clear()
    assert len(namer._charge_forms(smiles)) == 2            # the real supplier is back after the block
