"""LO4c: a bridged fused name may cite five or more identical suffixes ('-pentol').

``bridged_fused_pin._assemble`` spelled the suffix multiplier from a table that ended at four
('tetrol'), so a bridged fused ring system carrying five hydroxy groups - the tetrodotoxin skeleton,
5,9-epoxy-7,10a-methano-oxocino[4,5-d]pyrimidine - was declined at the last step of the build, after
the ring system, the parent, the hydrogenation and the numbering had all been found. The best-effort
tier then named the molecule by the von Baeyer fallback, with 'hydroxy' prefixes and no suffix, at
systematic_verified, and the default tier declined it.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* Selection of bridges / "Selection of fused ring system to be bridged"
  (:14257): the fused ring system, before bridging, must (a) "contain the maximum number of rings"
  (:14261), (b) "include the maximum number of skeletal atoms" (:14271), (c) "have the fewer
  heteroatoms in the fused ring system before bridging" (:14279). The tetrodotoxin skeleton is read
  as oxocino[4,5-d]pyrimidine (three heteroatoms) with an epoxy and a methano bridge, not as the
  CAS-style [1,3]dioxocino[6,5-d]pyrimidine with two methano bridges (four heteroatoms).
* Bridging nomenclature (:23816): "A bridged fused system... is used to generate names
  for structures that cannot be named by normal fusion nomenclature. A properly formed fusion name
  is first created; then additional rings are created by using bridges."
* numerical terms and (c) (:7619) elision of the terminal 'a' of the
  multiplying prefix before 'o': 'benzenehexol (PIN, ' (:7625), so 'penta' + 'ol' =
  'pentol'.

ASSUMED (the book prints no tetrodotoxin name): the hydrogenation and added-hydrogen spelling
'1,4,4a,5,6,7-hexahydro... -4,6,9,10,11(10H)-pentol' is the builder's own for this class 'Added indicated
hydrogen',:24687, and:24641 for hydro prefixes; is a von Baeyer numbering rule), checked here only by the independent read-back to the full InChIKey.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: smiles, PIN. The three tetrodotoxin-type rows of the fused-guanidine census (A3, left_cause_after,
#: internal notes) that the builder can name; a fourth, the pentacyclic
#: anhydro cage, has no reading with divalent bridges only (``best_splits`` finds none).
TTX_ROWS = [
    ("C[C@]1(O)[C@@H]2O[C@]3(O)O[C@H]1C(O)[C@@]1(NC(N)=N[C@@H](O)[C@@H]21)[C@@H]3O",
     "(4S,4aR,5R,6S,7S,9S,10S,10aR)-2-amino-6-methyl-1,4,4a,5,6,7-hexahydro-5,9-epoxy-"
     "7,10a-methanooxocino[4,5-d]pyrimidine-4,6,9,10,11(10H)-pentol"),
    ("NC1=N[C@H](O)[C@H]2[C@H]3O[C@]4(O)O[C@@H](C(O)[C@@]2(N1)[C@@H]4O)[C@]3(O)CO",
     "(4R,4aR,5R,6S,7S,9S,10S,10aR)-2-amino-6-(hydroxymethyl)-1,4,4a,5,6,7-hexahydro-5,9-epoxy-"
     "7,10a-methanooxocino[4,5-d]pyrimidine-4,6,9,10,11(10H)-pentol"),
    ("NC1=NC(O)C2C3OC4(O)O[C@@H](C(O)C2(N1)[C@@H]4O)C3(O)CO",
     "(7S,10S)-2-amino-6-(hydroxymethyl)-1,4,4a,5,6,7-hexahydro-5,9-epoxy-"
     "7,10a-methanooxocino[4,5-d]pyrimidine-4,6,9,10,11(10H)-pentol"),
]


def test_the_suffix_multiplier_reaches_five():
    from orthonym.rules.bridged_fused_pin import MAX_SUFFIX_COUNT
    assert MAX_SUFFIX_COUNT >= 6


@pytest.mark.parametrize("smiles,pin", TTX_ROWS)
def test_the_tetrodotoxin_type_rows_are_named_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", TTX_ROWS)
def test_the_name_replaces_the_von_baeyer_fallback(smiles, pin):
    """At START the best-effort name was the 'tetracyclo' von Baeyer one (systematic_verified, no
    '-ol' suffix); it is the builder's now, and the default tier ships it."""
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier"), b.get("producer")) == (pin, "pin_verified", "inner:tier_a_ring"), b
    assert "tetracyclo" not in b["name"]
    d = name_default(smiles)
    assert (d.get("name"), d.get("tier"), d.get("is_pin")) == (pin, "pin_verified", True), d


#: a fused (not bridged) parent with six hydroxy groups: the same cap, the spelling the book prints for the
#: parent ('octahydroindolizine',:17026 total hydrogenation; the retained 'indolizine'
#: Table 2.8:11709) and for the suffix ('benzenehexol (PIN, ':7625). Named at START by no producer
#: at the default tier (the multiplier stopped at four).
POLYOL_ROWS = [
    ("OC1CC(O)N2C(O)C(O)C(O)C(O)C12", "octahydroindolizine-1,3,5,6,7,8-hexol"),
]


@pytest.mark.parametrize("smiles,pin", POLYOL_ROWS)
def test_a_fused_polyol_with_six_suffixes(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
