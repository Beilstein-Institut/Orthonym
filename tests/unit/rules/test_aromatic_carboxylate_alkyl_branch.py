"""The aromatic carboxylate handler names a ring substituent by its carbon count only when the
substituent is a plain n-alkyl.

'propyl' for three carbons is another group when they form an isopropyl or a propenyl: the name
'sodium 4-propylbenzoate' read back as the n-propyl compound, the gate rejected it, and the default
tier declined sodium 4-isopropylbenzoate although its acid, '4-(propan-2-yl)benzoic acid', is
named. A branched or unsaturated substituent now leaves the handler to the neutral-acid route,
which names it ('4-(propan-2-yl)benzoate'; propan-2-yl is the preferred prefix of the isopropyl
group,, the Blue Book,:15946; '2-methylpropyl (preferred prefix) (not isobutyl)',
:16412). Benzoic acid is the retained parent,:29717).
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CC(C)c1ccc(cc1)C(=O)[O-].[Na+]", "sodium 4-(propan-2-yl)benzoate"),
    ("CC(C)(C)c1ccc(cc1)C(=O)[O-].[Na+]", "sodium 4-tert-butylbenzoate"),
    ("CC(C)Cc1ccc(cc1)C(=O)[O-]", "4-(2-methylpropyl)benzoate"),
    ("CCC(C)c1ccc(cc1)C(=O)[O-]", "4-(butan-2-yl)benzoate"),
    ("C=Cc1ccc(cc1)C(=O)[O-]", "4-ethenylbenzoate"),
    # the n-alkyl rings the handler still names itself
    ("CCc1ccc(cc1)C(=O)[O-].[Na+]", "sodium 4-ethylbenzoate"),
    ("CCCc1ccc(cc1)C(=O)[O-].[Na+]", "sodium 4-propylbenzoate"),
    ("CCCCc1ccc(cc1)C(=O)[O-]", "4-butylbenzoate"),
    ("Cc1cc(C)c(cc1)C(=O)[O-]", "2,4-dimethylbenzoate"),
])
def test_alkyl_ring_substituent_of_a_benzoate(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


def test_only_a_plain_n_alkyl_branch_is_counted():
    from rdkit import Chem

    from orthonym.rules.ions import _branch_all_carbon

    def ok(smiles, start):
        mol = Chem.MolFromSmiles(smiles)
        ring = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}   # the benzene ring
        return _branch_all_carbon(mol, start, ring)

    assert ok("CCCc1ccccc1", 2)             # propyl
    assert ok("Cc1ccccc1", 0)               # methyl
    assert not ok("CC(C)c1ccccc1", 1)       # isopropyl: a branch
    assert not ok("C=Cc1ccccc1", 1)         # ethenyl: a double bond
    assert not ok("C1CCCCC1c1ccccc1", 5)    # cyclohexyl: a ring
    assert not ok("COc1ccccc1", 1)          # a heteroatom
