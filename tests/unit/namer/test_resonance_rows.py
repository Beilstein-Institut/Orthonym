"""The constructed ladders of the resonance tests have the sizes their names give, and the flags-0 family builder
reproduces the committed ladder rows."""
import pytest
from rdkit import Chem

from tests.support import resonance_rows
from tests.support.resonance_rows import (
    FLAGS0_GROUPS,
    azide_ladder,
    benzimidazolium_ladder,
    flags0_family,
    imidazolium_ladder,
    imidazolium_with_work,
    n_azide_ladder,
    rows,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("build,core", [(azide_ladder, 3), (imidazolium_ladder, 5), (benzimidazolium_ladder, 9),
                                        (n_azide_ladder, 5)])
@pytest.mark.parametrize("n", [1, 2, 5, 9])
def test_a_ladder_adds_six_heavy_atoms_per_phenylene(build, core, n):
    mol = Chem.MolFromSmiles(build(n))
    assert mol is not None and mol.GetNumHeavyAtoms() == core + 6 * n


def test_fixture_readers_return_the_committed_rows():
    assert [r["id"] for r in rows("user_molecules.tsv")] == ["u1", "u2", "u3"]
    assert all(r["smiles"] for r in rows("flags0_ladders.tsv"))


def test_smiles_lines_takes_the_first_token_and_skips_blank_lines(tmp_path, monkeypatch):
    (tmp_path / "x.smi").write_text("CCO\n\nC[n+]1ccccc1\tsecond column\n")
    monkeypatch.setattr(resonance_rows, "FIXTURES", tmp_path)
    assert resonance_rows.smiles_lines("x.smi") == ["CCO", "C[n+]1ccccc1"]


def _family_id(row_id):
    """``<core>_<group>_n<n>_k<k>`` -> (core, group, n, k), else None."""
    parts = row_id.split("_")
    if len(parts) == 4 and parts[2][:1] == "n" and parts[3][:1] == "k" and parts[2][1:].isdigit() and parts[3][1:].isdigit():
        return parts[0], parts[1], int(parts[2][1:]), int(parts[3][1:])
    return None


@pytest.mark.parametrize("row", [r for r in rows("flags0_ladders.tsv") if _family_id(r["id"])], ids=lambda r: r["id"])
def test_flags0_family_reproduces_the_committed_ladder_rows(row):
    core, group, n, k = _family_id(row["id"])
    core = {"imid": "imidazolium", "benzimid": "benzimidazolium"}[core]
    assert flags0_family(core, group, k, n) == row["smiles"]


@pytest.mark.parametrize("row", [r for r in rows("flags0_ladders.tsv")
                                 if r["id"].startswith("imid_") and r["id"].endswith("_k5")], ids=lambda r: r["id"])
def test_flags0_family_reproduces_the_committed_five_group_rows(row):
    group = row["id"].split("_")[1]
    assert group in FLAGS0_GROUPS
    assert flags0_family("imidazolium", group, 5, 0) == row["smiles"]


@pytest.mark.parametrize("core", ["imidazolium", "benzimidazolium"])
@pytest.mark.parametrize("group", FLAGS0_GROUPS)
def test_every_flags0_family_molecule_parses_and_carries_k_groups(core, group):
    group_atoms = Chem.MolFromSmiles("C" + resonance_rows._GROUPS[group]).GetNumHeavyAtoms() - 1
    for n in (0, 1, 2):
        base = Chem.MolFromSmiles(flags0_family(core, group, 0, n))
        assert base is not None and sum(a.GetFormalCharge() for a in base.GetAtoms()) == 1
        for k in range(1, 6):
            mol = Chem.MolFromSmiles(flags0_family(core, group, k, n))
            assert mol is not None
            assert mol.GetNumHeavyAtoms() == base.GetNumHeavyAtoms() + k * group_atoms


@pytest.mark.parametrize("count", range(13, 45))
def test_imidazolium_with_work_has_the_work_it_is_asked_for(count):
    from orthonym.perception.charge_shift import pi_work
    mol = Chem.MolFromSmiles(imidazolium_with_work(count))
    assert mol is not None and pi_work(mol) == count
