"""LO4b: a geminal diol, two -OH on one ring atom, in the bridged fused PIN builder.

The saxitoxin family carries the hydrate of a ketone, C(OH)2, on a ring carbon (C10 of the
'pyrrolo[1,2-c]purine' skeleton, ``test_lo4b_tetravalent_fusion.py``); ``principal_suffix`` returned None for
any ring atom that carried the suffix twice, so 19 of the 29 rows stopped there.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* 'Added indicated hydrogen' (:24687): hydrogen added "as the consequence of the addition of a
  suffix"; 'naphthalen-4a(2H)-ol (PIN)' (:26864). A geminal diol (two -OH on one ring atom) needs the atom to
  be a >CH2 group of the parent, as a ketone does,:28386); Citation of locants (:2867)
  cites the locant once per group, as 'cyclohexane-1,1-dicarboximidamide (PIN)' (:34236) does for a suffix
  repeated on one ring atom.

ASSUMED (the book prints no geminal diol on a mancude parent): the atom takes the indicated or added
hydrogen as a ketone does (``hydro.lacks_hydrogen``, n_h = 2). The ketone has the book's own example:
'2,3-dihydro-1H-inden-1-one (PIN; see ' (:28977), where the carbonyl atom carries the indicated
hydrogen '1H', so 'octahydro-1H-indene-1,1-diol' is spelled the same way. A geminal diol on a fully
saturated naphthalene is labelled below the PIN by the independent spelling check ('octahydronaphthalene-
1,1(2H)-diol', pin_unverified), and the default tier declines it.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

SAXITOXIN_ROWS = [
    # a geminal diol on C10 (the hydrate of the ketone) and a second -OH on C9:, the diol atom
    # takes the indicated hydrogen '10H'
    ("NC1=N[C@H]2[C@H](CO)N=C(N)N3CC(O)C(O)(O)[C@]23N1",
     "(3aS,4R,10aS)-2,6-diamino-4-(hydroxymethyl)-3a,4,8,9-tetrahydro-1H,10H-pyrrolo[1,2-c]purine-9,10,10-triol"),
    # the geminal diol with the hydroxymethyl and no stereo
    ("NC1=NC2C(CO)N=C(N)N3CCC(O)(O)C23N1",
     "2,6-diamino-4-(hydroxymethyl)-3a,4,8,9-tetrahydro-1H,10H-pyrrolo[1,2-c]purine-10,10-diol"),
]


@pytest.mark.parametrize("smiles,pin", SAXITOXIN_ROWS)
def test_saxitoxin_rows_with_a_geminal_diol(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# geminal diols on skeletons that have nothing to do with the tetravalent atom: the suffix repeated on one
# ring atom:2867,:34236)
GEMINAL_ROWS = [
    ("OC1(O)CCC2CCCCC12", "octahydro-1H-indene-1,1-diol"),
    ("OC1(O)CC2CCCCC2N1", "octahydro-2H-indole-2,2-diol"),
]


@pytest.mark.parametrize("smiles,pin", GEMINAL_ROWS)
def test_a_geminal_diol_on_a_ring_atom(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


