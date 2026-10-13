"""_analogue_inchis asks RDKit for the other drawings of a charge only when ``charge_shift.pi_work`` is at most
RDKIT_RESONANCE0_MAX_WORK; above that count the other drawings are the placements of the +1 of each mobile nitrogen
cation, found by matching the pi bonds.

 "Acid salts of polybasic organic acids are named in two ways" (the Blue Book),:31593: "(1) by
substitutive nomenclature in which the free acid is cited as a prefix to the name of the anion;" and:31596 "Method (1)
generates preferred IUPAC names, except when the structure of the acid salt is unknown." So the hydron of an acid salt
has a site, and a name that moves it to another acid group, or the +1 of a cation to another atom than the input's
 "General rule for systematically naming cationic centers in parent hydrides",:41368: the cation is formed
"by adding one or more hydrons to any position of a neutral parent hydride"), names another species. Two structures are
one species when their neutral analogues share a standard InChI in some drawing of each charge.

RDKit's enumeration without a flag costs what the acyclic groups that carry charge or N=X bonds cost (x7 to x13 for each
azide, diazonium, azo or nitroso group) and its C++ call cannot be interrupted. Every test on a molecule that RDKit
does not finish in a second builds no supplier (``supplier_calls(forbid=True)``), so a regression fails at once instead
of hanging. The matching moves the +1 of a nitrogen and no -1, so above the bound the set of analogues can be narrower
than RDKit's and a verdict stricter, never silently looser: an undecided matching leaves the drawn analogue only.
"""
import random
import time

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly import memo
from orthonym.perception import charge_shift
from orthonym.perception.charge_shift import RDKIT_RESONANCE0_MAX_WORK as B0
from orthonym.perception.charge_shift import pi_work
from orthonym.validation import protonation_identity as pi
from tests.support.jars import jar_or_skip
from tests.support.resonance_rows import (
    azide_ladder,
    benzimidazolium_ladder,
    imidazolium_ladder,
    imidazolium_with_work,
    n_azide_ladder,
    rows,
)
from tests.support.resonance_spy import supplier_calls

pytestmark = pytest.mark.unit

#: every molecule takes the matching route / the RDKit route
MATCHING_ONLY = -1
RDKIT_ONLY = 10 ** 6
CPU_LIMIT_S = 5.0

LADDER_ROWS = rows("flags0_ladders.tsv")
HELPER_SIDE = [r for r in LADDER_ROWS if r["side"] == "helper"]
RDKIT_SIDE = {r["id"]: r["smiles"] for r in LADDER_ROWS if r["side"] == "rdkit"}
USER_ROWS = [r for r in rows("user_molecules.tsv")] + [
    r for r in rows("controls.tsv") if r["id"] in {"p_azido_tpp", "mol2_nozn", "mol3_cation"}]


@pytest.fixture(autouse=True)
def _no_process_memo(monkeypatch):
    # a verdict or a name computed earlier in the process must not hide the call under test
    monkeypatch.setattr(memo, "_PROCESS_MAX", 0)


def _chain(n):
    """n para-linked phenylenes, to be attached with a '-'."""
    return imidazolium_ladder(n).split("-", 1)[1]


def _expected_analogues(mol):
    """The analogue of the drawn structure and one for each placement of the +1 (the matching route's answer)."""
    out = {pi._neutral_analogue_inchi(mol)}
    out |= {pi._neutral_analogue_inchi(m) for m in charge_shift.charge_placements(mol) or []}
    return out - {None}


def _analogues_by(route, smiles, monkeypatch):
    """``_analogue_inchis`` of ``smiles`` with every molecule forced onto one route ("matching" or "rdkit")."""
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY if route == "matching" else RDKIT_ONLY)
    with supplier_calls(forbid=route == "matching") as calls:
        out = pi._analogue_inchis(Chem.MolFromSmiles(smiles))
    assert (calls == []) == (route == "matching")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# above the bound: no supplier
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("row", HELPER_SIDE, ids=lambda r: r["id"])
def test_a_charge_dense_cation_is_not_enumerated_by_rdkit(row):
    # imidazolium-NH + 5 diazonium groups, benzimidazolium-NH + 5 azides and their relatives (W0 33 to 57): RDKit does
    # not finish them in 25 s
    mol = Chem.MolFromSmiles(row["smiles"])
    assert pi_work(mol) > B0
    start = time.process_time()
    with supplier_calls(forbid=True) as calls:
        out = pi._analogue_inchis(mol)
    assert calls == [] and time.process_time() - start < CPU_LIMIT_S
    assert out == _expected_analogues(mol)
    if "[nH]" in row["smiles"]:
        assert len(out) >= 2                   # the second nitrogen of the azolium takes the +1
    else:
        assert len(out) == 1                   # a pyridinium and a hexa-azide have no second nitrogen to take it


@pytest.mark.parametrize("row", USER_ROWS, ids=lambda r: r["id"])
def test_the_user_molecules_and_their_relatives_are_not_enumerated_by_rdkit(row):
    mol = Chem.MolFromSmiles(row["smiles"])
    assert pi_work(mol) > B0
    start = time.process_time()
    with supplier_calls(forbid=True) as calls:
        out = pi._analogue_inchis(mol)
    assert calls == [] and time.process_time() - start < CPU_LIMIT_S
    assert out == _expected_analogues(mol) and len(out) >= 1


@pytest.mark.parametrize("smiles", [HELPER_SIDE[0]["smiles"], HELPER_SIDE[-1]["smiles"], imidazolium_ladder(4)])
def test_the_matching_route_does_not_depend_on_the_atom_order(smiles):
    drawn = Chem.MolFromSmiles(smiles)
    seen = set()
    for seed in range(4):
        order = list(range(drawn.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        with supplier_calls(forbid=True) as calls:
            seen.add(frozenset(pi._analogue_inchis(Chem.RenumberAtoms(drawn, order))))
        assert calls == []
    assert len(seen) == 1


# ---------------------------------------------------------------------------------------------------------------------
# at or below the bound: RDKit, as it is asked for a small molecule
# ---------------------------------------------------------------------------------------------------------------------

def test_a_small_cation_still_uses_rdkit():
    mol = Chem.MolFromSmiles("C[n+]1c[nH]c2ccccc21")
    assert pi_work(mol) <= B0
    with supplier_calls() as calls:
        out = pi._analogue_inchis(mol)
    assert calls == [("_analogue_inchis", 10)] and len(out) == 2


@pytest.mark.parametrize("name", sorted(RDKIT_SIDE))
def test_charge_dense_cations_within_the_bound_still_use_rdkit(name):
    mol = Chem.MolFromSmiles(RDKIT_SIDE[name])
    assert pi_work(mol) <= B0
    with supplier_calls() as calls:
        pi._analogue_inchis(mol)
    assert [c[0] for c in calls] == ["_analogue_inchis"]


@pytest.mark.parametrize("n", range(1, 10))
def test_the_imidazolium_ladder_has_both_charge_placements_on_either_side_of_the_bound(n):
    # the two nitrogens of the imidazolium give two neutral analogues at every length; n = 9 does not return through
    # RDKit
    mol = Chem.MolFromSmiles(imidazolium_ladder(n))
    helper_side = pi_work(mol) > B0
    with supplier_calls(forbid=helper_side) as calls:
        out = pi._analogue_inchis(mol)
    assert len(out) == 2
    assert (calls == []) == helper_side


def test_the_imidazolium_ladder_crosses_the_bound():
    works = [pi_work(Chem.MolFromSmiles(imidazolium_ladder(n))) for n in range(1, 10)]
    assert works == sorted(works) and works[0] <= B0 < works[-1]


@pytest.mark.parametrize("count", [B0 - 1, B0, B0 + 1, B0 + 4])
def test_rdkit_is_asked_up_to_the_bound_and_never_above_it(count, monkeypatch):
    smiles = imidazolium_with_work(count)
    assert pi_work(Chem.MolFromSmiles(smiles)) == count
    helper_side = count > B0
    with supplier_calls(forbid=helper_side) as calls:
        got = pi._analogue_inchis(Chem.MolFromSmiles(smiles))
    assert (calls == []) == helper_side
    assert got == _analogues_by("rdkit", smiles, monkeypatch)        # the molecules are cheap for RDKit


@pytest.mark.parametrize("route", ["matching", "rdkit"])
def test_the_function_reads_the_bound_when_it_runs(route, monkeypatch):
    assert len(_analogues_by(route, "C[n+]1c[nH]c2ccccc21", monkeypatch)) == 2


# ---------------------------------------------------------------------------------------------------------------------
# the matching against RDKit
# ---------------------------------------------------------------------------------------------------------------------

#: cations whose analogues the matching gives exactly as RDKit does (no azide, diazo or nitroso dipole whose -1 moves)
SAME_ANALOGUES = (
    ["C[n+]1c[nH]c2ccccc21", "C[N+]1=CNc2ccccc21", "NC(=[NH2+])NC", "CN(C)C=CC=[N+](C)C", "Cc1ccnn2c(C)n[n+](C)c12"]
    + [imidazolium_ladder(n) for n in range(1, 6)] + [benzimidazolium_ladder(n) for n in range(1, 6)]
    + [RDKIT_SIDE[name] for name in ("imid_nitroso_k5", "imid_azo_k5", "tetradiazonium_pyridinium", "imid_oxime_k5")])
#: cations with a dipole whose -1 RDKit moves and the matching does not: the matching's set is a subset
NARROWER_ANALOGUES = (
    [azide_ladder(n) for n in (1, 2, 3)] + [n_azide_ladder(n) for n in (1, 2)] + ["CN(C)N=[N+]=[N-]"]
    + [RDKIT_SIDE[name] for name in ("pentanitroso_pyridinium", "imid_azide_2")])


@pytest.mark.parametrize("smiles", SAME_ANALOGUES)
def test_the_matching_route_gives_the_analogues_of_the_rdkit_route(smiles, monkeypatch):
    assert _analogues_by("matching", smiles, monkeypatch) == _analogues_by("rdkit", smiles, monkeypatch)


@pytest.mark.parametrize("smiles", NARROWER_ANALOGUES)
def test_the_matching_route_never_adds_an_analogue_rdkit_lacks(smiles, monkeypatch):
    got = _analogues_by("matching", smiles, monkeypatch)
    assert got and got <= _analogues_by("rdkit", smiles, monkeypatch)


# ---------------------------------------------------------------------------------------------------------------------
# an undecided or failing matching leaves the drawn analogue only, and RDKit is never asked in its place
# ---------------------------------------------------------------------------------------------------------------------

def test_an_undecided_matching_leaves_the_drawn_analogue_only(monkeypatch):
    monkeypatch.setattr(charge_shift, "RDKIT_RESONANCE0_MAX_WORK", MATCHING_ONLY)
    monkeypatch.setattr(charge_shift, "charge_placements", lambda mol, **kwargs: None)
    mol = Chem.MolFromSmiles("C[n+]1c[nH]c2ccccc21")
    with supplier_calls(forbid=True) as calls:
        out = pi._analogue_inchis(mol)
    assert calls == [] and out == {pi._neutral_analogue_inchi(mol)}


def test_an_error_in_the_matching_leaves_the_drawn_analogue_only(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("the matching failed")

    monkeypatch.setattr(charge_shift, "charge_placements", fail)
    mol = Chem.MolFromSmiles(imidazolium_ladder(5))
    assert pi_work(mol) > B0
    with supplier_calls(forbid=True) as calls:
        out = pi._analogue_inchis(mol)
    assert calls == [] and out == {pi._neutral_analogue_inchi(mol)}


def test_a_bonded_metal_above_the_bound_leaves_the_drawn_analogue_only():
    # the matching does not draw the charge and bond moves of a bonded metal: undecided, and RDKit is not asked
    smiles = "[nH+]1c[nH]cc1-" + "-".join(["c1ccc(cc1)"] * 4 + ["c1ccc([Sn](C)(C)C)cc1"])
    mol = Chem.MolFromSmiles(smiles)
    assert pi_work(mol) > B0
    why = []
    assert charge_shift.charge_placements(mol, why=why) is None and why == ["metal"]
    with supplier_calls(forbid=True) as calls:
        out = pi._analogue_inchis(mol)
    assert calls == [] and out == {pi._neutral_analogue_inchi(mol)}


# ---------------------------------------------------------------------------------------------------------------------
# the same species, and the verdict
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 7])
def test_another_charge_placement_is_the_same_species_on_either_side_of_the_bound(n):
    # the +1 drawn on the other nitrogen of the imidazolium: each analogue set holds the other's drawn analogue
    a = Chem.MolFromSmiles(imidazolium_ladder(n))
    b = Chem.MolFromSmiles("[nH]1c[nH+]cc1-" + _chain(n))
    helper_side = pi_work(a) > B0
    assert helper_side == (n == 7)
    with supplier_calls(forbid=helper_side) as calls:
        assert pi._same_up_to_neutral_tautomer(a, b) is True
    assert (calls == []) == helper_side


def test_above_the_bound_another_regioisomer_is_another_species():
    a = Chem.MolFromSmiles("C[n+]1c[nH]cc1-" + _chain(7))
    b = Chem.MolFromSmiles("C[n+]1c[nH]c(-" + _chain(7) + ")c1")
    assert pi_work(a) > B0
    with supplier_calls(forbid=True) as calls:
        assert pi._same_up_to_neutral_tautomer(a, b) is False
    assert calls == []


def _pair(family, n):
    chain = _chain(n)
    return {
        # a neutral tautomer inside a molecule that carries a charge (a nitro group): one species
        "pyridone": ("O=c1[nH]cc([N+](=O)[O-])cc1-" + chain, "Oc1ncc([N+](=O)[O-])cc1-" + chain),
        # the hydron on the other carboxyl group: another species
        "acid": ("O=C([O-])Cc2cc(-" + chain + ")ccc2C(=O)O", "O=C(O)Cc2cc(-" + chain + ")ccc2C(=O)[O-]"),
        # a neutral tautomer beside a cation whose +1 is drawn on the other nitrogen: one species
        "indazole": ("C[n+]1c[nH]cc1-" + chain + "-c1ccc2[nH]ncc2c1", "Cn1c[nH+]cc1-" + chain + "-c1ccc2n[nH]cc2c1"),
    }[family]


#: (family, phenylenes, verdict); the first two lengths of each family are at or below the bound, the others above it
VERDICTS = [("pyridone", 2, "n/a"), ("pyridone", 3, "n/a"), ("pyridone", 5, "n/a"),
            ("acid", 2, "mismatch"), ("acid", 4, "mismatch"), ("acid", 6, "mismatch"),
            ("indazole", 1, "n/a"), ("indazole", 3, "n/a"), ("indazole", 5, "n/a")]


@pytest.mark.parametrize("family,n,verdict", VERDICTS)
def test_the_verdict_is_the_same_on_either_side_of_the_bound(family, n, verdict):
    smiles, parsed = _pair(family, n)
    works = {pi_work(Chem.MolFromSmiles(smiles)) > B0, pi_work(Chem.MolFromSmiles(parsed)) > B0}
    assert len(works) == 1
    helper_side = works.pop()
    with supplier_calls(forbid=helper_side) as calls:
        assert pi.protonation_site_verdict(smiles, parsed) == verdict
    if helper_side:
        assert calls == []
    else:
        assert {c[0] for c in calls} == {"_analogue_inchis"}


def test_the_verdicts_cross_the_bound():
    for family, short, long_ in (("pyridone", 2, 5), ("acid", 2, 6), ("indazole", 1, 5)):
        assert max(pi_work(Chem.MolFromSmiles(s)) for s in _pair(family, short)) <= B0
        assert min(pi_work(Chem.MolFromSmiles(s)) for s in _pair(family, long_)) > B0


def test_an_undecided_matching_makes_the_verdict_stricter_never_looser(monkeypatch):
    # the same species with the +1 drawn on the other nitrogen: n/a when the matching places the charge, a mismatch when
    # it cannot decide (the drawn analogues differ), never n/a without the evidence
    smiles, parsed = _pair("indazole", 4)
    assert pi.protonation_site_verdict(smiles, parsed) == "n/a"
    monkeypatch.setattr(charge_shift, "charge_placements", lambda mol, **kwargs: None)
    with supplier_calls(forbid=True) as calls:
        assert pi.protonation_site_verdict(smiles, parsed) == "mismatch"
    assert calls == []


# ---------------------------------------------------------------------------------------------------------------------
# the naming path: acid salts reach _analogue_inchis below the bound, with RDKit, and keep their names
# ---------------------------------------------------------------------------------------------------------------------

#: acid salts whose hydron can sit on another carboxyl group than the name places it: the verdict asks for the analogues
#: of the input and of the parse (_same_up_to_neutral_tautomer)
ACID_SALTS = [
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "sodium (2-carboxyphenyl)acetate"),
    ("O=C([O-])CCc1ccccc1C(=O)O.[Na+]", "sodium 3-(2-carboxyphenyl)propanoate"),
    ("O=C([O-])Cc1ccc(C(=O)O)cc1.[Na+]", "sodium (4-carboxyphenyl)acetate"),
    ("O=C([O-])Cc1cccc(C(=O)O)c1.[Na+]", "sodium (3-carboxyphenyl)acetate"),
    ("O=C([O-])Cc1ccccc1C(=O)O.[K+]", "potassium (2-carboxyphenyl)acetate"),
    ("O=C([O-])Cc1ccccc1C(=O)O", "(2-carboxyphenyl)acetate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ACID_SALTS)
def test_an_acid_salt_below_the_bound_reaches_rdkit_and_keeps_its_name(smiles, name):
    jar_or_skip()
    assert pi_work(Chem.MolFromSmiles(smiles)) <= B0
    with supplier_calls() as calls:
        named = Orthonym(style="pin").name_tiered(smiles)
    assert "_analogue_inchis" in {c[0] for c in calls}
    assert (named["tier"], named["name"]) == ("pin_verified", name)
