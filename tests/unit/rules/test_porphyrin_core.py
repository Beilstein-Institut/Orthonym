"""The porphyrin core as a ring-system query with the 1-24 numbering, and the guards of the
producer that names substituted free-base porphyrins (rules/porphyrin.py).

Governing text, the tetrapyrrole recommendations that Blue Book (reference 17) points to
(<https://iupac.qmul.ac.uk/tetrapyrrole/TP1.html>):

.1 "The fundamental macrocyclic tetrapyrrolic ring system shown in Fig. 2 is named porphyrin. It
is tautomeric with respect to the location of the two hydrogen atoms not involved in the peripheral
conjugated system.... However, for nomenclature purposes, the name "porhyrin" [sic] implies that the
saturated nitrogen atoms are at positions 21 and 23 unless specifically indicated otherwise."

.2 "The 1-24 numbering system... is adopted for the porphyrin nucleus and is shown in Fig. 2. The
2,3,7,8,12,13,17 and 18 positions have commonly been referred to generically as "beta- positions"...
positions at 1,4,6,9,11,14,16 and 19... "alpha- positions," while those at 5,10,15 and 20 are referred
to generically as "meso- positions.""

Blue Book INTRODUCTION: "Preferred IUPAC names (PINs) are not identified for the compounds in
this Chapter", so nothing here is a PIN ruling.

The molecules are constructed from names (tests/fixtures/porphyrin/*.tsv); the user molecules u1 and u2
come from tests/fixtures/resonance/user_molecules.tsv.
"""
import csv
import random
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym.rules import porphyrin
from tests.support.jars import jar_or_skip
from tests.support.resonance_rows import rows as resonance_rows
from tests.support.rt_assert import _independent_parse

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "porphyrin"


def _table(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


SUBSTITUTED = {r["id"]: r["smiles"] for r in _table("substituted_porphyrins.tsv")}
DECLINES = _table("decline_cases.tsv")
DECLINE_BY_ID = {r["id"]: r["smiles"] for r in DECLINES}

#.2, Fig. 2, written out: ring A = N21 with C1-C4, ring B = N22 with C6-C9, ring C = N23 with
# C11-C14, ring D = N24 with C16-C19; the meso carbons 5, 10, 15, 20 link the rings.
ALPHA = (1, 4, 6, 9, 11, 14, 16, 19)
BETA = (2, 3, 7, 8, 12, 13, 17, 18)
MESO = (5, 10, 15, 20)
NITROGEN = (21, 22, 23, 24)
RING_BONDS = {
    21: ((1, 2), (2, 3), (3, 4)),
    22: ((6, 7), (7, 8), (8, 9)),
    23: ((11, 12), (12, 13), (13, 14)),
    24: ((16, 17), (17, 18), (18, 19)),
}
N_BONDS = {21: (1, 4), 22: (6, 9), 23: (11, 14), 24: (16, 19)}
MESO_BONDS = ((4, 5), (5, 6), (9, 10), (10, 11), (14, 15), (15, 16), (19, 20), (20, 1))


def test_the_table_is_the_tetrapyrrole_numbering():
    expected = set()
    for nitrogen in NITROGEN:
        expected |= {frozenset(b) for b in RING_BONDS[nitrogen]}
        expected |= {frozenset((nitrogen, c)) for c in N_BONDS[nitrogen]}
    expected |= {frozenset(b) for b in MESO_BONDS}
    assert len(expected) == 28
    assert {frozenset(b) for b in porphyrin._CORE_EDGES} == expected and len(porphyrin._CORE_EDGES) == 28

    query = porphyrin._core_query()
    assert query.GetNumAtoms() == 24 and query.GetNumBonds() == 28
    by_locant = {a.GetAtomMapNum(): a for a in query.GetAtoms()}
    assert sorted(by_locant) == list(range(1, 25))
    assert all(by_locant[n].GetAtomicNum() == 7 for n in NITROGEN)
    assert all(by_locant[c].GetAtomicNum() == 6 for c in range(1, 21))
    assert all(b.GetSmarts() == "~" for b in query.GetBonds())  # any bond: Kekule and aromatic drawings

    neighbours = {loc: {n.GetAtomMapNum() for n in a.GetNeighbors()} for loc, a in by_locant.items()}
    assert all(len(neighbours[c]) == 3 for c in ALPHA)      # the beta carbon, the nitrogen and a meso carbon
    assert all(len(neighbours[c]) == 2 for c in BETA + MESO + NITROGEN)
    assert {n: neighbours[n] for n in NITROGEN} == {n: set(N_BONDS[n]) for n in NITROGEN}
    assert all(neighbours[m] <= set(ALPHA) and len(neighbours[m]) == 2 for m in MESO)


def test_the_numbering_agrees_with_opsin_for_every_ordered_pair():
    """132 ordered pairs of (methyl, ethyl) on the 12 substitutable positions: OPSIN's numbering of
    `p-methyl-q-ethylporphyrin` is one of the numberings of the producer's graph match."""
    jar_or_skip("opsin")
    positions = BETA + MESO
    checked = 0
    for p in positions:
        for q in positions:
            if p == q:
                continue
            smiles = _independent_parse(f"{p}-methyl-{q}-ethylporphyrin")
            assert smiles, (p, q)
            mol = _molecule(smiles)
            numberings = porphyrin.core_numberings(mol)
            core = set(numberings[0])
            attach = {}
            for atom in mol.GetAtoms():
                if atom.GetIdx() in core:
                    continue
                for nb in atom.GetNeighbors():
                    if nb.GetIdx() in core:
                        attach["methyl" if atom.GetDegree() == 1 else "ethyl"] = nb.GetIdx()
            images = {(loc[attach["methyl"]], loc[attach["ethyl"]]) for loc in numberings}
            assert (p, q) in images, (p, q, sorted(images))
            checked += 1
    assert checked == 132


def _locant_sets(mol):
    """For each numbering, the sorted locants of the core atoms that carry a substituent."""
    out = set()
    for loc in porphyrin.core_numberings(mol):
        out.add(tuple(sorted(
            loc[i] for i in loc
            if any(nb.GetIdx() not in loc for nb in mol.GetAtomWithIdx(i).GetNeighbors()))))
    return out


@pytest.mark.parametrize("name", ["5,10,15,20-tetraphenylporphyrin", "2-methyl-5-phenylporphyrin",
                                  "2-ethyl-3-(4-methylphenyl)porphyrin"])
def test_locants_come_from_map_numbers_not_atom_order(name):
    mol = _molecule(SUBSTITUTED[name])
    reference = _locant_sets(mol)
    assert reference
    for seed in range(8):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        assert _locant_sets(Chem.RenumberAtoms(mol, order)) == reference, seed


def test_a_trans_drawing_has_four_numberings_and_the_cis_drawing_none():
    tpp = _molecule(SUBSTITUTED["5,10,15,20-tetraphenylporphyrin"])
    assert len(porphyrin.core_numberings(tpp)) == 4
    # the graph itself matches 8 ways (D4 symmetry); only the 4 with the drawn N-H on N21 and N23 are names
    assert len(tpp.GetSubstructMatches(porphyrin._core_query(), uniquify=False)) == 8
    for loc in porphyrin.core_numberings(tpp):
        assert sorted(loc[i] for i in loc if loc[i] > 20 and tpp.GetAtomWithIdx(i).GetTotalNumHs() == 1) == [21, 23]
    assert porphyrin.core_numberings(_molecule(DECLINE_BY_ID["cis_TPP"])) == []


@pytest.mark.parametrize("row", DECLINES, ids=lambda r: r["id"])
def test_decline_reason(row):
    assert porphyrin.decline_reason(_molecule(row["smiles"])) == row["code"]


def test_every_decline_class_of_the_producer_has_a_row():
    classes = {r["code"] for r in DECLINES}
    assert classes == {"no_core", "fragments", "multiple_cores", "charged", "radical", "isotope",
                       "n_substituted", "metal", "not_mancude", "nh_pattern", "fused_ring",
                       "suffix_group", "stereo", "bare", "unaccounted"}


def test_a_branch_the_shared_collector_cannot_identify_is_declined(monkeypatch):
    """`get_fused_heterocycle_substituents` skips a branch it cannot name, so an incomplete collection would
    name a different molecule; the guard declines instead (rules/fused_rings._exocyclic_atoms_accounted)."""
    import orthonym.rules.fused_rings as fused_rings
    mol = _molecule(SUBSTITUTED["5,10,15,20-tetraphenylporphyrin"])
    assert porphyrin.decline_reason(mol) is None
    monkeypatch.setattr(fused_rings, "_exocyclic_atoms_accounted", lambda *_args, **_kwargs: False)
    assert porphyrin.decline_reason(mol) == "unaccounted"


def test_the_charged_rows_are_net_charged_or_carry_a_charged_core_atom():
    """A charge-separated group whose charges cancel is not a charge; a net charge and a charge on a
    core atom are."""
    for row_id in ("dication_TPP", "tetrakis_N_methylpyridinium_4_yl", "zwitterionic_core_net_0"):
        mol = _molecule(DECLINE_BY_ID[row_id])
        net = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        core = set(mol.GetSubstructMatches(porphyrin._core_query())[0])
        charged_core = any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in core)
        assert net != 0 or charged_core, row_id
        assert porphyrin.decline_reason(mol) == "charged", row_id
    zwitter = _molecule(DECLINE_BY_ID["zwitterionic_core_net_0"])
    assert sum(a.GetFormalCharge() for a in zwitter.GetAtoms()) == 0      # the second branch alone declines it


def _user_row(row_id):
    return next(r for r in resonance_rows("user_molecules.tsv") if r["id"] == row_id)["smiles"]


@pytest.mark.parametrize("smiles", [
    SUBSTITUTED["5,10,15,20-tetrakis(4-nitrophenyl)porphyrin"],
    SUBSTITUTED["2-nitro-5,10,15,20-tetraphenylporphyrin"],
    SUBSTITUTED["2-(azidomethyl)-5,10,15,20-tetraphenylporphyrin"],
    _user_row("u1"),
    _user_row("u2").split(".")[0],          # u2 without its free [Zn+2]: the free base
], ids=["tetrakis_nitrophenyl", "nitro_TPP", "azidomethyl_TPP", "u1", "u2_free_base"])
def test_charge_separated_groups_are_not_charged(smiles):
    mol = _molecule(smiles)
    assert any(a.GetFormalCharge() for a in mol.GetAtoms())               # the groups do carry charges
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 0
    assert porphyrin.decline_reason(mol) is None


def test_a_metal_on_a_peripheral_group_is_declined_even_when_net_neutral():
    """Net charge 0, a charge-separated pair outside the core, no N-metal bond on the core: only the
    metal guard declines it."""
    mol = _molecule(DECLINE_BY_ID["TPP_pyridyl_PdCl2_covalent"])
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 0
    assert any(a.GetFormalCharge() for a in mol.GetAtoms())
    assert porphyrin.decline_reason(mol) == "metal"


def test_the_free_ion_and_the_core_n_metal_bond_keep_their_own_reason_codes():
    assert porphyrin.decline_reason(_molecule(DECLINE_BY_ID["TPP_plus_free_Zn2+"])) == "fragments"
    assert porphyrin.decline_reason(_molecule(DECLINE_BY_ID["ZnTPP_bonded"])) == "n_substituted"


def test_the_bare_parent_and_a_substituted_porphyrin_are_in_or_out_of_scope():
    assert porphyrin.decline_reason(_molecule(DECLINE_BY_ID["bare_porphyrin"])) == "bare"
    assert porphyrin.decline_reason(_molecule(SUBSTITUTED["5,10,15,20-tetraphenylporphyrin"])) is None


def test_an_ordinary_molecule_is_rejected_before_any_search():
    assert porphyrin.decline_reason(Chem.MolFromSmiles("CCO")) == "no_core"
    assert porphyrin.decline_reason(Chem.MolFromSmiles("c1ccccc1")) == "no_core"
    assert porphyrin.core_numberings(Chem.MolFromSmiles("CCO")) == []
