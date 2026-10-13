"""Names of substituted free-base porphyrins on the retained parent ``porphyrin`` (rules/porphyrin.py).

The molecules are constructed from names (tests/fixtures/porphyrin/substituted_porphyrins.tsv: the SMILES
is OPSIN's reading of the expected name at plan time). Every name is checked three ways: it equals the
expected name, OPSIN reads it back to the input's tautomer-specific canonical SMILES, and the full
InChIKey of that reading equals the input's.

Governing text: the tetrapyrrole recommendations cited by Blue Book (reference 17),
.1 (the name implies the hydrogens on N21 and N23),.2 (the 1-24 numbering) and
.7 "Systematic Names for Substituted Porphyrins": "When there is a choice the starting point and
direction of numbering around the porphyrin nucleus are chosen so as to give lowest locants to the
following structural factors (if present), considered successively in the order listed until a decision
is reached: (a) hydrogen atoms expressed as "indicated hydrogen" (b) principal groups named as suffix
(c) substituents named as prefixes and hydro prefixes all considered together in one series in ascending
numerical order (d) the substituent named as a prefix, which is cited first in a name." The same
criteria are (b), (c), (f) and (g). Blue Book INTRODUCTION: "Preferred IUPAC names (PINs)
are not identified for the compounds in this Chapter".
"""
import csv
import random
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.porphyrin import decline_reason, name_substituted_porphyrin
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import _independent_parse, name_is_rt_exact

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "porphyrin"


def _table(name):
    with open(FIXTURES / name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


ROWS = _table("substituted_porphyrins.tsv")
DECLINES = _table("decline_cases.tsv")


def _molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


def _canonical(smiles):
    return Chem.MolToSmiles(_molecule(smiles))


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_expected_names(row):
    assert name_substituted_porphyrin(_molecule(row["smiles"])) == row["expected_name"]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_the_name_reads_back_to_the_input_tautomer(row):
    jar_or_skip("opsin")
    name = name_substituted_porphyrin(_molecule(row["smiles"]))
    parsed = _independent_parse(name)
    assert parsed, name
    # canonical SMILES is tautomer-specific: it keeps the drawn N-H
    assert _canonical(parsed) == _canonical(row["smiles"]), name
    assert name_is_rt_exact(name, row["smiles"]), name


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_the_name_does_not_depend_on_atom_order(row):
    mol = _molecule(row["smiles"])
    for seed in range(8):
        order = list(range(mol.GetNumAtoms()))
        random.Random(seed).shuffle(order)
        assert name_substituted_porphyrin(Chem.RenumberAtoms(mol, order)) == row["expected_name"], seed


def test_the_expected_names_are_what_opsin_reads():
    """Guards the fixture: the stored SMILES is OPSIN's reading of the expected name."""
    jar_or_skip("opsin")
    for row in ROWS:
        parsed = _independent_parse(row["expected_name"])
        assert parsed and _canonical(parsed) == _canonical(row["smiles"]), row["expected_name"]


def test_every_in_scope_row_is_in_scope():
    assert [r["id"] for r in ROWS if decline_reason(_molecule(r["smiles"])) is not None] == []


@pytest.mark.parametrize("row", [r for r in DECLINES if r["code"] == "suffix_group"], ids=lambda r: r["id"])
def test_no_senior_group_name_is_built(row):
    # a carboxylic acid, phenol, amine, sulfonic acid, aldehyde or nitrile anywhere makes another parent
    # senior,: the name is another one, e.g. the multiplicative
    # 4,4',4'',4'''-(porphyrin-5,10,15,20-tetrayl)tetrabenzoic acid
    assert name_substituted_porphyrin(_molecule(row["smiles"])) is None


@pytest.mark.parametrize("row", DECLINES, ids=lambda r: r["id"])
def test_a_declined_molecule_gets_no_name_from_this_module(row):
    assert name_substituted_porphyrin(_molecule(row["smiles"])) is None


def test_two_trans_tautomers_are_two_names():
    """One constitution drawn as its two trans tautomers..1 puts the retained name's hydrogens on N21
    and N23, so only the numberings that do so are names of the molecule (4 per drawing);.7 (c) then
    takes the lowest locant: 2 for the drawing with N-H on rings A and C, 7 for the other (B and D)."""
    jar_or_skip("opsin")
    by_id = {r["id"]: r["smiles"] for r in ROWS}
    first, second = by_id["2-methylporphyrin"], by_id["7-methylporphyrin"]
    mols = [_molecule(first), _molecule(second)]
    infos = [inchi.MolToInchi(m) for m in mols]
    keys = [inchi.MolToInchiKey(m) for m in mols]
    assert keys == ["AZYZJSPCUMGVDB-UHFFFAOYSA-N", "CIEFSSKFAANGNM-UHFFFAOYSA-N"]
    assert infos[0].split("/")[:3] == infos[1].split("/")[:3]            # the same formula and connection layer
    assert [i.split("/h")[1] for i in infos][0] != [i.split("/h")[1] for i in infos][1]   # the N-H sites differ
    assert _canonical(first) != _canonical(second)
    names = [name_substituted_porphyrin(m) for m in mols]
    assert names == ["2-methylporphyrin", "7-methylporphyrin"]
    assert name_is_rt_exact(names[0], first) and name_is_rt_exact(names[1], second)
    assert not name_is_rt_exact(names[0], second) and not name_is_rt_exact(names[1], first)


_POSITIONS = (2, 3, 7, 8, 12, 13, 17, 18, 5, 10, 15, 20)


def _methyl_and_ethyl_locants(mol):
    """(methyl, ethyl) locant pairs over the numberings that put the drawn N-H on N21 and N23."""
    from orthonym.rules.porphyrin import core_numberings
    numberings = core_numberings(mol)
    core = set(numberings[0])
    attach = {}
    for atom in mol.GetAtoms():
        if atom.GetIdx() in core:
            continue
        for neighbour in atom.GetNeighbors():
            if neighbour.GetIdx() in core:
                attach["methyl" if atom.GetDegree() == 1 else "ethyl"] = neighbour.GetIdx()
    return {(loc[attach["methyl"]], loc[attach["ethyl"]]) for loc in numberings}


def test_every_ordered_pair_of_methyl_and_ethyl_is_named_as_the_criteria_say():
    """The 132 molecules `p-methyl-q-ethylporphyrin` (OPSIN's reading)..7 (c) takes the numbering with
    the lowest set of locants and (d) breaks a tie by the prefix cited first, which is 'ethyl' (under 'e'):
    the expected name is worked out here from the candidate numberings alone, and it reads back."""
    jar_or_skip("opsin")
    checked = 0
    for p in _POSITIONS:
        for q in _POSITIONS:
            if p == q:
                continue
            smiles = _independent_parse(f"{p}-methyl-{q}-ethylporphyrin")
            mol = _molecule(smiles)
            methyl, ethyl = min(_methyl_and_ethyl_locants(mol), key=lambda pair: (sorted(pair), pair[1]))
            name = name_substituted_porphyrin(mol)
            assert name == f"{ethyl}-ethyl-{methyl}-methylporphyrin", (p, q, name)
            parsed = _independent_parse(name)
            assert parsed and _canonical(parsed) == _canonical(smiles), (p, q, name)
            checked += 1
    assert checked == 132


def test_seeded_three_substituent_porphyrins_read_back():
    """60 seeded porphyrins with three different substituents at three of the 12 positions, written in
    arbitrary order, which exercise the tie-break of the numberings (tert-butyl cited under b, a locant-led
    prefix under its stem): the producer's name reads back to the input's tautomer-specific canonical SMILES
    and is the same name for a reshuffled atom order."""
    jar_or_skip("opsin")
    rng = random.Random(20261013)
    prefixes = ["methyl", "ethyl", "tert-butyl", "bromo", "phenyl", "(4-methylphenyl)", "nitro", "methoxy"]
    for _ in range(60):
        positions = rng.sample(_POSITIONS, 3)
        subs = rng.sample(prefixes, 3)
        name_in = "-".join(f"{p}-{s}" for p, s in zip(positions, subs)) + "porphyrin"
        smiles = _independent_parse(name_in)
        assert smiles, name_in
        mol = _molecule(smiles)
        name = name_substituted_porphyrin(mol)
        assert name, name_in
        parsed = _independent_parse(name)
        assert parsed and _canonical(parsed) == _canonical(smiles), (name_in, name)
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        assert name_substituted_porphyrin(Chem.RenumberAtoms(mol, order)) == name, name_in
