""" breadth — sulfinyl / sulfonyl substituent on a heteroaromatic parent.

Benzene names a -S(=O)R / -S(=O)(=O)R substituent as (R)sulfinyl / (R)sulfonyl
(`(methanesulfinyl)benzene`), but every HETEROaromatic parent dropped it: the
heterocycle substituent collector counted the R carbons and dropped the S + its
=O, so the whole molecule abstained (caught the atom-drop). The recursive
`name_substituent(..., allow_mancude=True)` already builds these prefixes; the
heterocycle path now routes a ring-borne S(=O)-substituent through it, behind a
gate-independent atom-coverage guard. best-effort-gated -> PIN default byte-identical.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.unit


def _be():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.parametrize("smi,expected", [
    ("CS(=O)c1ccncc1", "4-(methanesulfinyl)pyridine"),
    ("CS(=O)(=O)c1ccncc1", "4-(methanesulfonyl)pyridine"),
])
def test_heteroaryl_sulfinyl_names_at_best_effort(smi, expected):
    assert _be().name_tiered(smi)["name"] == expected


def test_pin_default_byte_identical():
    """The PIN default no longer abstains on pyridine-sulfinyl: the PIN tier gained the class
    and names it with the same 'methanesulfinyl' prefix as the benzene control below
    (the Blue Book '1-(methanesulfinyl)-2-(methylsulfanyl)ethane (PIN)', under the
    heading "Classes denoted by the senior atom in heterane nomenclature"). The name is the
    best-effort tier's own, so the two tiers agree byte for byte; OPSIN 2.9.0 parses it to the
    input's full InChIKey (an InChIKey)."""
    assert Orthonym(style="pin").name("CS(=O)c1ccncc1") == "4-(methanesulfinyl)pyridine"
    assert _be().name_tiered("CS(=O)c1ccncc1")["name"] == "4-(methanesulfinyl)pyridine"


def test_benzene_control_unchanged():
    assert Orthonym(style="pin").name("CS(=O)c1ccccc1") == "(methanesulfinyl)benzene"


def test_complex_arm_names_substitutively():
    """ Wave F (core-namer item 3): a benzyl/aryl-methyl arm now names
    SUBSTITUTIVELY via the sulfinyl/sulfonyl-rooted guard in
    ``name_substituent`` -> ``4-(benzylsulfinyl)pyridine`` (was the stale
    abstain pin, and before the guard the 'a'-replacement
    ``4-(2-(cyclohexa-1,3,5-trien-1-yl)-1-oxo-1-thiaethyl)pyridine``). OPSIN
    round-trips to the input InChIKey (change-asserted-value verified)."""
    from rdkit import Chem
    from rdkit.Chem import inchi
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "O=S(Cc1ccccc1)c1ccncc1"
    r = _be().name_tiered(smi)
    # '4-(benzylsulfinyl)pyridine' became '4-(phenylmethanesulfinyl)pyridine': the CH2 between the ring and the
    # sulfur is the parent of the sulfinyl group and the prefix is the acid-stem form
    # 'methanesulfinyl', the Blue Book; '1-methyl-4-[(R)-phenyl(18O1)methanesulfonyl]benzene
    # (PIN)':46030; '1-(methanesulfinyl)-2-(methylsulfanyl)ethane (PIN)':18284). change-asserted-value:
    # OPSIN reads the new name back to the input (below); mutation =
    # tests/unit/rules/test_ring_methane_sulfonyl_suffix.py.
    assert r["name"] == "4-(phenylmethanesulfinyl)pyridine"
    assert "oxo" not in r["name"] and "thiaethyl" not in r["name"]
    got = opsin_parse(r["name"])
    assert got and (inchi.MolToInchiKey(Chem.MolFromSmiles(got))
                    == inchi.MolToInchiKey(Chem.MolFromSmiles(smi)))
