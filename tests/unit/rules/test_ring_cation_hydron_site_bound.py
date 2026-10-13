"""_ring_cation_hydron_site asks RDKit only when another ring heteroatom carries a hydrogen and the molecule is
within the work bound; above the bound the site comes from a matching of the pi bonds.

 "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: "A cation derived formally by adding one or more hydrons
to any position of a neutral parent hydride... is named by replacing the final letter
'e' of the parent hydride name, if any, by the suffix 'ium'": the added hydron sits on a
ring heteroatom that has a hydrogen. A resonance structure at RDKit flags 0 moves bond
orders and charges and keeps the hydrogen count of every atom, so a drawing with no such
atom has no site and the search is skipped.

RDKit's enumeration without a flag costs what the acyclic groups that carry charge or N=X bonds cost (x7 to x13 for each
azide, diazonium, azo or nitroso group) and its C++ call cannot be interrupted, so it is asked only when
``charge_shift.pi_work(mol)`` is at most ``RDKIT_RESONANCE0_MAX_WORK``. Above that count the matching of
``charge_shift.hydron_sites`` answers, and an undecided answer is "no site": RDKit is never asked in its place. Every
test on a molecule that RDKit does not finish in a second builds no supplier (``supplier_calls(forbid=True)``), so a
regression fails at once instead of hanging. Rows taken from the benchmark corpora are in
tests_local/unit/rules/test_ring_cation_hydron_site_bound_local.py.
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

import orthonym
from orthonym.perception import charge_shift
from orthonym.perception.charge_shift import RDKIT_RESONANCE0_MAX_WORK as B0
from orthonym.perception.charge_shift import pi_work
from orthonym.rules import ions
from tests.support.jars import jar_or_skip
from tests.support.resonance_rows import benzimidazolium_ladder, imidazolium_with_work, rows
from tests.support.resonance_spy import supplier_calls

pytestmark = pytest.mark.unit

# another ring heteroatom with a hydrogen: the search finds a site
SITE_FOUND = [
    "C[N+]1=CNc2ccccc21",
    "C[n+]1cc[nH]c1",
    "C[n+]1c[nH]c2ccccc21",
    "C[n+]1cc[nH]n1",
    "C[n+]1cnc2[nH]cnc2c1",
    "C[n+]1ccc[nH]c1=O",
    "C[N+]1=C(N)NC=C1",
    "C[n+]1c(-c2ccccc2)[nH]c2ccccc21",
]
# no other ring heteroatom carries a hydrogen: no site, no search
NO_HYDROGEN_ELSEWHERE = [
    "C[n+]1ccccc1",
    "C[n+]1ccn(C)c1",
    "Cn1cc[n+](C)c1",
    "C[n+]1cnn(C)c1",
    "C[n+]1ccsc1",
    "C[n+]1c(C=Cc2cccs2)sc2ccccc21",
    "C[n+]1ccccc1[O-]",
    "C[N+]1=CN=CC=C1",
]
# another ring heteroatom carries a hydrogen but the charge cannot move onto it
HYDROGEN_BUT_NO_SITE = ["C[n+]1cc[nH]c1[O-]"]
# the charge moves between atoms of a conjugated chain, with no ring cation centre
CHARGE_MOVES_ACYCLIC = ["CN(C)C=CC=[N+](C)C", "NC(=[NH2+])N"]

RING_CATIONS = SITE_FOUND + NO_HYDROGEN_ELSEWHERE + HYDROGEN_BUT_NO_SITE


def _centres(mol):
    """The ring atoms the namer asks about: neither H nor C, charge +1, no hydrogen."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and a.GetAtomicNum() not in (1, 6)
            and a.GetFormalCharge() == 1 and a.GetTotalNumHs() == 0]


def _search(mol, centre):
    with supplier_calls() as calls:
        site = ions._ring_cation_hydron_site(mol, ions._ring_system_of(mol, centre), centre)
    return site, calls


@pytest.mark.parametrize("smiles", NO_HYDROGEN_ELSEWHERE)
def test_no_other_ring_hydrogen_means_no_enumeration(smiles):
    mol = Chem.MolFromSmiles(smiles)
    centres = _centres(mol)
    assert centres
    for centre in centres:
        site, calls = _search(mol, centre)
        assert site is None and calls == []


@pytest.mark.parametrize("smiles", SITE_FOUND)
def test_the_hydron_site_is_the_other_ring_heteroatom_with_a_hydrogen(smiles):
    mol = Chem.MolFromSmiles(smiles)
    (centre,) = _centres(mol)
    site, calls = _search(mol, centre)
    assert site is not None and [c[0] for c in calls] == ["_ring_cation_hydron_site"]
    atom = mol.GetAtomWithIdx(site[1])
    assert site[1] != centre and atom.IsInRing()
    assert atom.GetAtomicNum() not in (1, 6) and atom.GetTotalNumHs() >= 1


@pytest.mark.parametrize("smiles", HYDROGEN_BUT_NO_SITE)
def test_a_ring_hydrogen_admits_the_search_and_the_search_may_find_nothing(smiles):
    mol = Chem.MolFromSmiles(smiles)
    (centre,) = _centres(mol)
    assert ions._other_ring_heteroatom_with_hydrogen(mol, ions._ring_system_of(mol, centre), centre)
    site, calls = _search(mol, centre)
    assert site is None and len(calls) == 1


@pytest.mark.parametrize("smiles", RING_CATIONS)
def test_the_pre_check_never_skips_a_search_that_finds_a_site(smiles, monkeypatch):
    mol = Chem.MolFromSmiles(smiles)
    for centre in _centres(mol):
        ring_system = ions._ring_system_of(mol, centre)
        admitted = ions._other_ring_heteroatom_with_hydrogen(mol, ring_system, centre)
        gated = ions._ring_cation_hydron_site(mol, ring_system, centre)
        with monkeypatch.context() as patch:
            patch.setattr(ions, "_other_ring_heteroatom_with_hydrogen", lambda *args: True)
            ungated = ions._ring_cation_hydron_site(mol, ring_system, centre)
        assert admitted or ungated is None
        assert (gated is None) == (ungated is None)
        if gated is not None:
            assert gated[1] == ungated[1]
            assert Chem.MolToSmiles(gated[0]) == Chem.MolToSmiles(ungated[0])


@pytest.mark.parametrize("smiles", RING_CATIONS + CHARGE_MOVES_ACYCLIC)
def test_flags_0_structures_keep_the_hydrogen_count_of_every_atom(smiles):
    mol = Chem.MolFromSmiles(smiles)
    supplier = Chem.ResonanceMolSupplier(mol, 0, 64)
    structures = [supplier[i] for i in range(len(supplier))]
    drawn = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    assert structures and all([a.GetTotalNumHs() for a in s.GetAtoms()] == drawn for s in structures)
    if smiles in SITE_FOUND or smiles in CHARGE_MOVES_ACYCLIC:
        assert len(structures) >= 2


@pytest.mark.parametrize("smiles", ["C[n+]1ccccc1", "C[n+]1c(C=Cc2cccs2)sc2ccccc21"])
def test_a_cation_without_a_hydron_site_keeps_the_drawn_centre_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    (centre,) = _centres(mol)
    with supplier_calls() as calls:
        name = ions._ring_cation_hydron_site_name(mol, centre, "drawn-centre-name")
    assert name == "drawn-centre-name" and calls == []


# ---------------------------------------------------------------------------------------------------------------------
# the work bound: RDKit up to RDKIT_RESONANCE0_MAX_WORK, the matching above it
# ---------------------------------------------------------------------------------------------------------------------

#: every molecule takes the matching route / the RDKit route
MATCHING_ONLY = -1
RDKIT_ONLY = 10 ** 6

LADDER_ROWS = rows("flags0_ladders.tsv")
NH_ROWS = [r for r in LADDER_ROWS if "[nH]" in r["smiles"]]
HELPER_SIDE = [r for r in NH_ROWS if r["side"] == "helper"]
RDKIT_SIDE = [r for r in NH_ROWS if r["side"] == "rdkit"]
USER_MOLECULES = rows("user_molecules.tsv")
RELATIVES = [r for r in rows("controls.tsv") if r["id"] in {"p_azido_tpp", "mol2_nozn", "mol3_cation"}]

#: the 5-membered charge-separated twin of a zwitterion: RDKit finds the NH as the site, the matching would have to
#: move the anion too and is undecided
ZWITTERION = "C[n+]1[nH]n(C)c(=O)c1C([O-])=NC"
HARD_TIMEOUT_S = 60
CPU_LIMIT_S = 5.0


def _site(mol, centre):
    return ions._ring_cation_hydron_site(mol, ions._ring_system_of(mol, centre), centre)


def _only_centre(mol):
    (centre,) = _centres(mol)
    return centre


def _inchi_key(mol):
    return Chem.MolToInchiKey(mol)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 9])
def test_the_benzimidazolium_ladder_finds_the_same_site_on_both_sides_of_the_bound(n):
    # n = 9 does not return in 100 s through RDKit: the helper side forbids the supplier
    mol = Chem.MolFromSmiles(benzimidazolium_ladder(n))
    helper_side = pi_work(mol) > B0
    with supplier_calls(forbid=helper_side) as calls:
        site = _site(mol, _only_centre(mol))
    assert site is not None and site[1] == 2                         # the NH of the imidazole ring
    assert (calls == []) == helper_side
    if not helper_side:
        assert [c[0] for c in calls] == ["_ring_cation_hydron_site"]


@pytest.mark.parametrize("count", [B0 - 1, B0, B0 + 1, B0 + 4])
def test_rdkit_is_asked_up_to_the_bound_and_never_above_it(count, monkeypatch):
    mol = Chem.MolFromSmiles(imidazolium_with_work(count))
    assert pi_work(mol) == count
    centre = _only_centre(mol)
    helper_side = count > B0
    with supplier_calls(forbid=helper_side) as calls:
        got = _site(mol, centre)
    assert (calls == []) == helper_side
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", RDKIT_ONLY)
    want = _site(mol, centre)                                        # the molecules are cheap for RDKit
    assert got is not None and want is not None and got[1] == want[1]
    assert Chem.MolToSmiles(got[0]) == Chem.MolToSmiles(want[0])


@pytest.mark.parametrize("bound, helper_side", [(MATCHING_ONLY, True), (RDKIT_ONLY, False)])
def test_the_function_reads_the_bound_when_it_runs(bound, helper_side, monkeypatch):
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", bound)
    mol = Chem.MolFromSmiles("C[n+]1cc[nH]c1")
    with supplier_calls(forbid=helper_side) as calls:
        site = _site(mol, _only_centre(mol))
    assert site is not None and (calls == []) == helper_side


@pytest.mark.parametrize("smiles", SITE_FOUND + [benzimidazolium_ladder(n) for n in (1, 2, 3, 4, 5)])
def test_the_matching_route_gives_the_site_and_the_structure_of_the_rdkit_route(smiles, monkeypatch):
    mol = Chem.MolFromSmiles(smiles)
    centre = _only_centre(mol)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", RDKIT_ONLY)
    with supplier_calls() as rdkit_calls:
        want = _site(mol, centre)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    with supplier_calls(forbid=True) as matching_calls:
        got = _site(mol, centre)
    assert [c[0] for c in rdkit_calls] == ["_ring_cation_hydron_site"] and matching_calls == []
    assert want is not None and got is not None and got[1] == want[1]
    assert Chem.MolToSmiles(got[0]) == Chem.MolToSmiles(want[0])


@pytest.mark.parametrize("row", HELPER_SIDE, ids=lambda r: r["id"])
def test_charge_dense_cations_build_no_supplier(row):
    # imidazolium-NH + 5 diazonium groups, benzimidazolium-NH + 5 azides and their relatives (W0 33 to 57): RDKit does
    # not finish them in 25 s
    mol = Chem.MolFromSmiles(row["smiles"])
    assert pi_work(mol) > B0
    centre = _only_centre(mol)
    with supplier_calls(forbid=True) as calls:
        site = _site(mol, centre)
    assert calls == []
    assert site is not None and site[1] != centre
    assert mol.GetAtomWithIdx(site[1]).GetTotalNumHs() == 1 and mol.GetAtomWithIdx(site[1]).GetSymbol() == "N"
    assert site[0].GetAtomWithIdx(centre).GetFormalCharge() == 0
    assert site[0].GetAtomWithIdx(site[1]).GetFormalCharge() == 1


@pytest.mark.parametrize("row", RDKIT_SIDE, ids=lambda r: r["id"])
def test_charge_dense_cations_within_the_bound_still_use_rdkit_and_the_matching_agrees(row, monkeypatch):
    mol = Chem.MolFromSmiles(row["smiles"])
    assert pi_work(mol) <= B0
    centre = _only_centre(mol)
    with supplier_calls() as calls:
        want = _site(mol, centre)
    assert [c[0] for c in calls] == ["_ring_cation_hydron_site"]
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    with supplier_calls(forbid=True) as matching_calls:
        got = _site(mol, centre)
    assert matching_calls == []
    # the same site and the same species; a dipole (azide) may be drawn in either of its two forms
    assert want is not None and got is not None and got[1] == want[1]
    assert _inchi_key(got[0]) == _inchi_key(want[0])


@pytest.mark.parametrize("smiles", ["C[n+]1cc[nH]c1", "C[N+]1=CNc2ccccc21", benzimidazolium_ladder(4)])
def test_the_matching_route_gives_the_name_of_the_rdkit_route(smiles, monkeypatch):
    jar_or_skip()
    mol = Chem.MolFromSmiles(smiles)
    centre = _only_centre(mol)
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", RDKIT_ONLY)
    want = ions._ring_cation_hydron_site_name(mol, centre, "")
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    with supplier_calls() as calls:
        got = ions._ring_cation_hydron_site_name(mol, centre, "")
    assert "_ring_cation_hydron_site" not in [c[0] for c in calls]      # the name's other checks may ask for their own
    assert got == want != ""


def test_an_undecided_matching_means_no_site_and_never_a_supplier(monkeypatch):
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    mol = Chem.MolFromSmiles(ZWITTERION)
    centre = _only_centre(mol)
    why = []
    assert charge_shift.hydron_sites(mol, centre, ions._ring_system_of(mol, centre), why=why) is None
    assert why == ["anion"]
    with supplier_calls(forbid=True) as calls:
        assert _site(mol, centre) is None
        assert ions._ring_cation_hydron_site_name(mol, centre, "drawn-centre-name") == "drawn-centre-name"
    assert calls == []


def test_the_same_zwitterion_within_the_bound_still_gets_its_site_from_rdkit():
    mol = Chem.MolFromSmiles(ZWITTERION)
    centre = _only_centre(mol)
    assert pi_work(mol) <= B0
    with supplier_calls() as calls:
        assert _site(mol, centre) is not None
    assert [c[0] for c in calls] == ["_ring_cation_hydron_site"]


def test_a_bonded_metal_above_the_bound_is_no_site_and_never_a_supplier():
    # the matching does not draw the charge and bond moves of a bonded metal: undecided, and RDKit is not asked
    smiles = "C[n+]1c[nH]cc1-" + "-".join(["c1ccc(cc1)"] * 4 + ["c1ccc([Sn](C)(C)C)cc1"])
    mol = Chem.MolFromSmiles(smiles)
    assert pi_work(mol) > B0
    why = []
    centre = _only_centre(mol)
    assert charge_shift.hydron_sites(mol, centre, ions._ring_system_of(mol, centre), why=why) is None
    assert why == ["metal"]
    with supplier_calls(forbid=True) as calls:
        assert _site(mol, centre) is None
    assert calls == []


def test_an_error_in_the_matching_is_no_site_and_never_a_supplier(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("the matching failed")

    monkeypatch.setattr(charge_shift, "hydron_sites", fail)
    mol = Chem.MolFromSmiles(benzimidazolium_ladder(5))
    assert pi_work(mol) > B0
    with supplier_calls(forbid=True) as calls:
        assert _site(mol, _only_centre(mol)) is None
    assert calls == []


@pytest.mark.parametrize("smiles", [benzimidazolium_ladder(4), HELPER_SIDE[0]["smiles"], HELPER_SIDE[-1]["smiles"]])
def test_the_matching_route_does_not_depend_on_the_atom_order(smiles, monkeypatch):
    # the structure, and the atom that carries the hydron, are the same for every numbering of the input
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    drawn = Chem.MolFromSmiles(smiles)
    seen = set()
    for seed in range(6):
        order = list(range(drawn.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        mol = Chem.RenumberAtoms(drawn, order)
        with supplier_calls(forbid=True) as calls:
            structure, site = _site(mol, _only_centre(mol))
        assert calls == []
        atom = structure.GetAtomWithIdx(site)
        seen.add((Chem.MolToSmiles(structure), atom.GetSymbol(), atom.GetTotalNumHs(), atom.GetFormalCharge()))
    assert len(seen) == 1


_CHILD = """
import json, sys, time
from rdkit import Chem
from rdkit.Chem import rdchem
from orthonym.rules import ions
built = []
def refuse(mol, *args, **kwargs):
    built.append(mol.GetNumHeavyAtoms())
    raise RuntimeError("a resonance supplier must not be built here")
Chem.ResonanceMolSupplier = refuse
rdchem.ResonanceMolSupplier = refuse
out = {}
for name, smiles in json.loads(sys.argv[1]).items():
    mol = Chem.MolFromSmiles(smiles)
    centres = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing() and a.GetAtomicNum() not in (1, 6)
               and a.GetFormalCharge() == 1 and a.GetTotalNumHs() == 0]
    start = time.process_time()
    names = [ions._ring_cation_hydron_site_name(mol, c, "drawn-centre-name") for c in centres]
    out[name] = {"centres": len(centres), "names": names, "cpu": time.process_time() - start}
print(json.dumps({"built": built, "rows": out}))
"""


def test_the_user_molecules_leave_the_hydron_site_within_seconds():
    # a child process: RDKit's C++ call holds the GIL and ignores signals, so only a kill from outside ends a hang
    source_root = str(Path(orthonym.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [source_root, os.environ.get("PYTHONPATH")])))
    payload = json.dumps({r["id"]: r["smiles"] for r in USER_MOLECULES + RELATIVES})
    try:
        done = subprocess.run([sys.executable, "-c", _CHILD, payload], capture_output=True, text=True,
                              timeout=HARD_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail("_ring_cation_hydron_site_name did not return for the user molecules in %d s" % HARD_TIMEOUT_S,
                    pytrace=False)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["built"] == []
    assert sum(entry["centres"] for entry in result["rows"].values()) >= 1       # the benzothiazolium of molecule 3
    for name, entry in result["rows"].items():
        assert entry["names"] == ["drawn-centre-name"] * entry["centres"], name
        assert entry["cpu"] < CPU_LIMIT_S, (name, entry["cpu"])
