"""Amidinium cations take the cationic suffix of the amidine group, 'imidamidium'.

 (heading **** "Cationic compounds derived from neutral compounds
expressed by suffixes are named in two ways", the Blue Book), method (1),:41431:
"Cationic suffixes derived from names of acids named by a suffix, amides, imides, nitriles,
amines, and imines are formed by adding the suffix 'ium' to the basic suffix, as indicated in
Table 7.4" (:41417: 'amide, carboxamide' -> 'amidium, carboxamidium'). An amidine is named as an
amide by functional replacement, '-imidamide' and '-carboximidamide' "Suffixes for amidines",:34173,:34175), and
the Blue Book's own cation is '5lambda5-quinolizin-5-ylium-2-carboximidamidium (PIN)'
(:42217), drawn with =NH2(+) and -NH2 on one carbon. Neither name nor drawing places the
hydron on one nitrogen, so the names below are the cations of ethanimidamide and of
benzenecarboximidamide; the name does not read '1-amino-1-azaniumylideneethane'.

OPSIN reads the suffix with the hydron on the amide nitrogen (CC(=N)[NH3+]), which has another
fixed-H InChI than the delocalised amidinium (CC(N)=[NH2+]); validation/protonation_identity
leaves the site open for a name that cites 'imidamidium' (a single amidine group, one onium
centre), and keeps every other protonation site a mismatch. A free amine on the same molecule
is not the cation: '5-aminopentanimidamidium' is the amidine, '5-amino-5-iminopentan-1-aminium'
the amine.
"""
import pytest

from tests.support.jars import jar_or_skip

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CC(N)=[NH2+]", "ethanimidamidium"),
    ("CC(=[NH2+])N", "ethanimidamidium"),                      # the other drawing
    ("CCC(N)=[NH2+]", "propanimidamidium"),
    ("NC(=[NH2+])c1ccccc1", "benzenecarboximidamidium"),
    ("CN(C)C(=[NH2+])c1ccccc1", "N,N-dimethylbenzenecarboximidamidium"),
    ("C[NH+]=C(N)c1ccccc1", "N-methylbenzenecarboximidamidium"),
    ("NC(=[NH2+])CCCCN", "5-aminopentanimidamidium"),
    ("NC(=N)CCCC[NH3+]", "5-amino-5-iminopentan-1-aminium"),
    ("CC(N)=[NH2+].[Cl-]", "ethanimidamidium chloride"),
    ("NC(=[NH2+])c1ccccc1.[Br-]", "benzenecarboximidamidium bromide"),
])
def test_amidinium_name(smiles, name):
    jar_or_skip()
    from orthonym import Orthonym
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


@pytest.mark.parametrize("smiles", [
    "CC(N)=[NH2+]",
    "NC(=[NH2+])c1ccccc1",
    "NC(=[NH2+])CCCCN",
])
def test_no_azaniumylidene_name_is_offered_at_the_wider_tiers(smiles):
    jar_or_skip()
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    for tier in ("valid", "best-effort"):
        r = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
        assert "azaniumylidene" not in (r["name"] or ""), (tier, r)


def test_mono_ionized_polyamine_emitter_declines_a_doubly_bonded_nitrogen():
    """'-aminium' names a nitrogen joined by a single bond: the =NH2(+) of an amidinium is not
    that group (the emitter named it '1-aminoethan-1-aminium', the cation of another molecule)."""
    from rdkit import Chem

    from orthonym.rules.ions import emit_mono_ionized_polyfunctional
    mol = Chem.MolFromSmiles("CC(N)=[NH2+]")
    ion = next(a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge() == 1)
    assert emit_mono_ionized_polyfunctional(mol, ion, "aminium") == ""
    # a single-bonded protonated amine beside a sibling amine is still named
    diamine = Chem.MolFromSmiles("NCC[NH3+]")
    ion = next(a.GetIdx() for a in diamine.GetAtoms() if a.GetFormalCharge() == 1)
    assert emit_mono_ionized_polyfunctional(diamine, ion, "aminium") == "2-aminoethan-1-aminium"


def test_site_of_the_hydron_is_open_only_for_the_amidine_group():
    """validation/protonation_identity: the fixed-H InChIs of the delocalised amidinium and of the
    amide-nitrogen reading differ; a name citing 'imidamidium' places no hydron, any other
    name, and any other onium centre, keeps the mismatch."""
    from orthonym.validation.protonation_identity import protonation_site_verdict as v
    assert v("CC(N)=[NH2+]", "CC(=N)[NH3+]", "ethanimidamidium") == "ok"
    assert v("CC(N)=[NH2+]", "CC(=N)[NH3+]", "1-aminoethan-1-aminium") == "mismatch"
    assert v("CC(N)=[NH2+]", "CC(=N)[NH3+]", None) == "mismatch"
    # the hydron on the amine of a molecule that also holds an amidine is another species
    assert v("NC(=[NH2+])CCCCN", "N=C(N)CCCC[NH3+]", "5-aminopentanimidamidium") == "mismatch"
    # two onium centres: out of scope of the open site
    assert v("NC(=[NH2+])CCC[NH3+]", "N=C([NH3+])CCC[NH3+]", "4-azaniumylbutanimidamidium") == "mismatch"
