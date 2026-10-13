"""A partially ionized polycarboxylic acid on one ring: the ionized group is the suffix, the
free acid a 'carboxy' prefix.

 "Acid salts of polybasic organic acids are named in two ways" (the Blue Book):
"(1) by substitutive nomenclature in which the free acid is cited as a prefix to the name of the
anion" (:31593) and "Method (1) generates preferred IUPAC names, except when the structure of
the acid salt is unknown." (:31596); the examples are 'potassium 6-carboxyhexanoate (PIN)' and
'ammonium 3-carboxypropanoate (PIN)' (:31602,:31607), and 'sodium hydrogen
2-(carboxylatomethyl)benzoate (PIN)' (:31610) for the method (2) form of a salt whose
structure is not known. An anion is senior to an acid, '4 Anions' before
'7 Acids',:18167), and benzoic acid is the retained parent of a benzene ring with one such
group,:29717, 'C6H5-COOH benzoic acid (PIN)'). So the monoanion of phthalic acid is
'2-carboxybenzoate', not '2-carboxy-1-[oxido(oxo)methyl]benzene' (the carboxylate as an
anionic prefix of a carboxylic acid that is not its own parent) and not
'hydrogen benzene-1,2-dicarboxylate' (method (2), which does not place the hydron).

Numbering, (:3219): ring heteroatoms, indicated hydrogen, then the ionized groups as the
suffix (c,:3256), then the prefixes.
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=C([O-])c1ccccc1C(=O)O", "2-carboxybenzoate"),
    ("O=C([O-])c1ccc(C(=O)O)cc1", "4-carboxybenzoate"),
    ("O=C([O-])c1cccc(C(=O)O)c1", "3-carboxybenzoate"),
    ("O=C([O-])c1cc(C(=O)O)cc(C(=O)O)c1", "3,5-dicarboxybenzoate"),
    ("O=C([O-])c1ccc(Cl)cc1C(=O)O", "2-carboxy-4-chlorobenzoate"),
    ("O=C([O-])c1ccc(O)cc1C(=O)O", "2-carboxy-4-hydroxybenzoate"),
    # two ionized groups and one free: the suffix is '-dicarboxylate'
    ("O=C([O-])c1cc(C(=O)[O-])cc(C(=O)O)c1", "5-carboxybenzene-1,3-dicarboxylate"),
    # a ring heteroatom fixes locant 1; the ionized group takes the lower locant of its two
    ("O=C(O)c1cccnc1C(=O)[O-]", "3-carboxypyridine-2-carboxylate"),
    ("O=C([O-])c1cccnc1C(=O)O", "2-carboxypyridine-3-carboxylate"),
    ("O=C([O-])c1ccncc1C(=O)O", "3-carboxypyridine-4-carboxylate"),
    # the proven constructions are untouched
    ("OC(=O)C1CCC(CC1)C(=O)[O-]", "4-carboxycyclohexane-1-carboxylate"),
    ("O=C([O-])c1ccccc1CC(=O)O", "2-(carboxymethyl)benzoate"),
    ("OC(=O)CCCCCC(=O)[O-]", "6-carboxyhexanoate"),
])
def test_partial_acid_anion_on_a_ring(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


@pytest.mark.parametrize("smiles,name", [
    ("O=C(O)c1ccccc1C(=O)[O-].[NH4+]", "ammonium 2-carboxybenzoate"),
    ("O=C([O-])c1ccccc1C(=O)O.[Na+]", "sodium 2-carboxybenzoate"),
    ("O=C(O)c1cc(C(=O)[O-])cc(C(=O)[O-])c1.[Na+].[Na+]", "disodium 5-carboxybenzene-1,3-dicarboxylate"),
    ("O=C([O-])c1ccccc1C(=O)O.O=C([O-])c1ccccc1C(=O)O.[Ca+2]", "calcium bis(2-carboxybenzoate)"),
])
def test_partial_acid_anion_in_a_salt(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


@pytest.mark.parametrize("smiles", [
    "O=C([O-])c1ccccc1C(=O)O",
    "O=C([O-])c1ccc(C(=O)O)cc1",
])
def test_no_oxido_oxo_methyl_name_at_the_wider_tiers(smiles):
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    for tier in ("valid", "best-effort"):
        r = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
        assert "oxido(oxo)methyl" not in (r["name"] or ""), (tier, r)


@pytest.mark.parametrize("smiles", [
    "O=C([O-])c1ccccc1C(=O)[O-]",                  # fully ionized: 'benzene-1,2-dicarboxylate'
    "O=C(O)c1ccccc1C(=O)O",                        # neutral: nothing is ionized
    "O=C([O-])c1ccc2ccccc2c1C(=O)O",               # fused ring: no simple cycle
    "O=C([O-])c1ccccc1-c1ccccc1C(=O)O",            # two rings
    "O=C([O-])c1ccccc1C(=O)O.[Na+]",               # more than one fragment
])
def test_builder_declines_outside_its_scope(smiles):
    from rdkit import Chem

    from orthonym.rules.acid_salt_anions import name_ring_acid_salt_anion
    mol = Chem.MolFromSmiles(smiles)
    site = next(({"atom_idx": a.GetIdx()} for a in mol.GetAtoms() if a.GetFormalCharge() == -1),
                {"atom_idx": 0})
    assert name_ring_acid_salt_anion(mol, site) == ""
