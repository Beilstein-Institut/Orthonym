"""LO4c: a fused ring system of a parent component and two or more first-order attached components
is named by fusion nomenclature (``bridged_fused_pin.poly_fusion``).

``hetero_fusion`` builds the name of two components (the parent and one attached component, a
benzo unit merged beforehand); a skeleton of three or four components, each fused directly to the
parent, had no parent name in the builder, so the fused-guanidine rows below (the benzazepine-fused
imidazoline, i.e. epinastine and its two enantiomers, and an imidazo-pyrido-pyrimidine) were named
only by the von Baeyer fallback and the default tier declined them.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* Constructing polycomponent fusion names (:12706): "When several nonidentical components
  must be considered, one, and only one, can be the parent component. All other components are
  attached components. A component attached directly to the parent component is called a
  'first-order attached component'."
* Order of seniority for selecting parent components (:12956), (b) (:12970) the
  location that needs no attached component higher than first-order, (c) (:12976) "the location that
  results in the maximum number of first-order attached components", (d) (:12984) "a location that
  permits the expression of the maximum number of identical attached components with multiplicative
  prefixes".
* Order of citation of fusion prefixes, (:13040): "All attached
  components are cited in front of the parent component... If there are two or more different
  components, or sets of components, attached to a lower order component, they are cited in
  alphabetical order" ('furo[3,2-b]thieno[2,3-e]pyridine (PIN)',:13044); the multiplying prefix
  is not part of the alphabetical order:12909): 'dibenzo' before 'imidazo'.
* Order of seniority of locants (letters and numbers) (:13076): (a) "parent component
  letters as a set" (:13080; 'furo[3,2-h]pyrrolo[3,4-a]carbazole (PIN)',:13086, is declined by the
  multiparent guard, see the decline rows), (b) "Parent component letters in order of citation"
  (:13088, 'furo[3,4-b]thieno[2,3-e]pyrazine (PIN)',:13094).
* Identical attached components (:13467): "Two or more components that are identical and
  both fused to a parent component are indicated by the use of multiplying prefixes... If
  abbreviated sets of locants are used, the letters are separated by a comma... the locants of the
  second component are primed": 'difuro[3,2-b:3',4'-e]pyridine (PIN)' (:13479), 'dibenzo[c,e]oxepine
  (PIN)' (:13481).
* Heteromonocyclic rings fused to a benzene ring (:13437): a benzo heterocycle is one
  component "However, this approach is not used if it disrupts a multiparent system... or the use
  of multiplicative prefixes (see ": two benzene rings on one azepine are 'dibenzo', one
  benzene ring is the benzo name (``hetero_fusion``).
* (:13451) "A multiparent name is preferred to a fused ring system, when there is a
  choice."
"""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import hetero_fusion, parents, selection
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _poly_fusion():
    from orthonym.rules.bridged_fused_pin import poly_fusion
    return poly_fusion


def _skeleton(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


#: names the book prints in PIN-marked (or PIN-listed) examples of the class, on OPSIN 2.9.0's
#: structure of the name; the comment is the line of the Blue Book that prints it
BOOK_POLY_NAMES = [
    ('O1C=CC2=NC3=C(C=C21)SC=C3', 'furo[3,2-b]thieno[2,3-e]pyridine'),  #:13044
    ('S1C=CC=2C1=NC=1C(N2)=COC1', 'furo[3,4-b]thieno[2,3-e]pyrazine'),  #:13094
    ('O1CC=CC2=C1C=1C(=NC2)C=COC1', "dipyrano[4,3-b:2',3'-d]pyridine"),  #:13098
    ('O1NC=C2C1=C1N(C=C2)C=CO1', '[1,2]oxazolo[5,4-c][1,3]oxazolo[3,2-a]pyridine'),  #:13074
    ('O1C=CC2=NC=3C(C=C21)=COC3', "difuro[3,2-b:3',4'-e]pyridine"),  #:13479
    ('C1=CC=CC2=COC=C3C(=C21)C=CC=C3', 'dibenzo[c,e]oxepine'),  #:13481
    ('C1=CC=CC2OC=C3C(=C21)C=CC=C3', 'dibenzo[b,d]pyran'),  #:13463
]


@pytest.mark.parametrize("smiles,name", BOOK_POLY_NAMES)
def test_the_book_polycomponent_names_are_rebuilt_by_the_rules(smiles, name):
    assert _poly_fusion().rule_name(_skeleton(smiles)) == name


#: the benzazepine-fused imidazoline of epinastine, imidazo[1,5-a] on a dibenzo[c,f]azepine
EPINASTINE_SKELETON = "C1=NCC2c3ccccc3Cc3ccccc3N12"


def test_two_benzo_prefixes_and_an_imidazo_prefix_on_azepine():
    """ (:13467) 'dibenzo' for the two identical benzene rings, (:13040)
    alphabetical citation ('benzo' before 'imidazo', the multiplying prefix not counted)."""
    assert _poly_fusion().rule_name(_skeleton(EPINASTINE_SKELETON)) == "dibenzo[c,f]imidazo[1,5-a]azepine"


#: three identical attached components on one parent: skeleton, name. (:13467) "To
#: distinguish between two or more components of the same order, the locants of the second component
#: are primed (or double primed if the first is primed, etc.), the third double primed, and so on":
#: the second set takes one prime, the third two. The benzo row is the carbo case (letters only, no
#: locants, so no primes at all).
THREE_IDENTICAL_COMPONENTS = [
    ("N1C2OCCC2C2OCCC2C2OCCC21", "trifuro[2,3-b:2',3'-d:2'',3''-f]azepine"),
    ("N1C2SCCC2C2SCCC2C2SCCC21", "trithieno[2,3-b:2',3'-d:2'',3''-f]azepine"),
    ("N1c2ccccc2-c2ccccc2-c2ccccc12", "tribenzo[b,d,f]azepine"),
]


@pytest.mark.parametrize("smiles,name", THREE_IDENTICAL_COMPONENTS)
def test_the_third_identical_hetero_component_takes_two_primes(smiles, name):
    """ Identical attached components (:13467): the i-th identical component's locants carry
    i primes. The old spelling gave the third component one prime ('2',3'-f'), a name that repeats the
    second component's locant set and that the rule does not give. The name is also read back by OPSIN
    2.9.0 to the skeleton it was built from."""
    from orthonym.validation.opsin_roundtrip import extended_smiles_or_unavailable
    key = _skeleton(smiles)
    assert _poly_fusion().rule_name(key) == name
    line = None
    for ih in ("", "1H-", "3H-", "9H-"):
        line = extended_smiles_or_unavailable(ih + name)
        if line:
            break
    assert line, name
    back = Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(line.split()[0])))
    assert Chem.MolToSmiles(back) == Chem.MolToSmiles(key)


#: whole molecules of the class, named at both tiers; each was abstained at the default tier before the
#: third component's locants took two primes (OPSIN 2.9.0 did not read the one-prime spelling, so the
#: verified name was never offered). The mancude oxepine row carries no hydro prefix; ASSUMED (the book
#: prints no trifuro name): the indicated hydrogen and hydro prefix spelling of the other two rows is the
#: builder's own:24633), checked here by the OPSIN read-back to the full InChIKey.
THREE_COMPONENT_MOLECULES = [
    ("c1cc2c(o1)Oc1ccoc1-c1ccoc1-2", "trifuro[2,3-b:2',3'-d:2'',3''-f]oxepine"),
    ("C1=C2C(=c3ccoc3=Nc3ccoc32)OC1", "2H-trifuro[2,3-b:2',3'-d:2'',3''-f]azepine"),
    ("CN1C2CCOC2C2CCOC2C2CCOC21", "7-methyldodecahydro-2H-trifuro[2,3-b:2',3'-d:2'',3''-f]azepine"),
]


@pytest.mark.parametrize("smiles,pin", THREE_COMPONENT_MOLECULES)
def test_three_identical_hetero_components_are_named_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_parent_chain_reaches_the_polycomponent_producer():
    mol = Chem.MolFromSmiles(EPINASTINE_SKELETON)
    got = parents.fused_parent(mol, selection.ring_system(mol))
    assert got is not None
    assert (got.name, got.source) == ("dibenzo[c,f]imidazo[1,5-a]azepine", "poly_fusion_name+opsin")


@pytest.mark.parametrize("name,why", [
    ("cyclopropa[b]dicyclopenta[2,3:6,7]oxepino[4,5-e]pyridine", "second-order attached components (:13060)"),
    ("furo[3,2-g]dipyrano[2,3-b:3',4',5'-de]quinoline", "peri-fusion, an atom in three rings (:13058)"),
    ("benzo[1,2-b:4,5-b']difuran", "a multiparent name (P-25.3.5.3 :13451)"),
    ("dipyrido[3,2-b:2',3'-e]pyridine", "the parent's ring repeated: a multiparent reading (P-25.3.7 :13530)"),
    # the book prints this PIN (:13086) but a pyrrole ring of the parent (carbazole) repeats the
    # attached pyrrolo: the conservative multiparent guard (``hetero_fusion._multi``, the one the
    # two-component producer applies) declines it; a decline never emits a wrong name
    ("furo[3,2-h]pyrrolo[3,4-a]carbazole", "multiparent guard, the book's own PIN left declined"),
])
def test_names_outside_the_class_are_declined(name, why):
    from orthonym.validation.opsin_roundtrip import extended_smiles_or_unavailable
    line = None
    for ih in ("", "1H-", "2H-", "3H-", "4H-"):
        line = extended_smiles_or_unavailable(ih + name)
        if line:
            break
    assert line, name
    assert _poly_fusion().rule_name(_skeleton(line.split()[0])) is None, why


def test_one_benzene_ring_is_left_to_the_benzo_name():
    """ (:13437): one benzene ring on a heteromonocycle of five or more members is a benzo
    name, built by ``hetero_fusion``; the polycomponent producer does not name that reading."""
    key = _skeleton("C1C=NC=2N1C1=C(C=CC2)C=CC=C1")      # imidazo[1,2-a][1]benzazepine
    assert _poly_fusion().rule_name(key) is None
    assert hetero_fusion.rule_name(key) == "imidazo[1,2-a][1]benzazepine"


#: smiles, PIN. The epinastine rows (3-amino-hydro-dibenzo[c,f]imidazo[1,5-a]azepine): the parent by
#: /, the indicated hydrogen and hydro prefixes as the builder spells them
#: 'Indicated hydrogen, added indicated hydrogen and hydro prefixes':24633), '-amine':26168) for the ring -NH2 of the guanidine carbon
#: (``seniority.ring_skeleton_units_resolved``). The last row: the parent imidazo[1,2-a]pyrido[3,4-e]
#: pyrimidine with the letters a,e, the lower set (a):13080; b,d is the other numbering),
#: and the ketone as '-5(1H)-one'. ASSUMED (the book prints none of the four): the hydrogenation spelling.
PIN_ROWS = [
    ("NC1=NCC2c3ccccc3Cc3ccccc3N12", "9,13b-dihydro-1H-dibenzo[c,f]imidazo[1,5-a]azepin-3-amine"),
    ("NC1=NC[C@H]2c3ccccc3Cc3ccccc3N12", "(13bR)-9,13b-dihydro-1H-dibenzo[c,f]imidazo[1,5-a]azepin-3-amine"),
    ("NC1=NC[C@@H]2c3ccccc3Cc3ccccc3N12", "(13bS)-9,13b-dihydro-1H-dibenzo[c,f]imidazo[1,5-a]azepin-3-amine"),
    ("Cc1ccccc1CN1C(=O)C2=C(CCN(Cc3ccccc3)C2)N2CCN=C12",
     "7-benzyl-4-[(2-methylphenyl)methyl]-2,4,6,7,8,9-hexahydroimidazo[1,2-a]pyrido[3,4-e]pyrimidin-5(1H)-one"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_the_fused_guanidine_rows_are_named_at_both_tiers(smiles, pin):
    """The reproducers of the record: default tier abstained, best-effort named them by the
    von Baeyer fallback at systematic_verified."""
    assert_pin_at_both_tiers(smiles, pin)


def test_the_reproducers_best_effort_producer_is_the_builder():
    for smiles, pin in PIN_ROWS:
        b = name_breadth(smiles)
        assert (b.get("name"), b.get("tier"), b.get("producer")) == (pin, "pin_verified", "inner:tier_a_ring"), b
        assert name_default(smiles).get("name") == pin


SPARTEINE = "C1CCN2C[C@@H]3C[C@@H](CN4CCCC[C@@H]34)[C@H]2C1"
SPARTEINE_PIN = "(7S,7aR,14S,14aS)-dodecahydro-2H,6H-7,14-methanodipyrido[1,2-a:1',2'-e][1,5]diazocine"


def test_a_natural_product_parent_takes_the_bridged_fused_pin_once_the_builder_names_it():
    """Gold row P14C- (``benchmarks/the gold set/packs/np_stereoparents.json``) moves from 'sparteine' to
    its bridged fused name. Sparteine is a (a) alkaloid parent (the Blue Book,
    ':57147 sparteine (named systematically by CAS)'). Its ring system, 7,14-methanodipyrido[1,2-a:1',2'-e]
    [1,5]diazocine, had no builder parent (three components) and the natural-product route kept the parent
    name; ``poly_fusion`` names the fused parent (two identical attached components on a diazocine parent,
     :13467), and the route offers the builder's name first, as it does for the seven gold rows
    of commit 6e2f89fdd (kaurane, gibbane,...). (:50943): "Preferred IUPAC names (PINs) are not
    identified for the compounds in this Chapter", so the book is silent; ASSUMED: that commit's policy
    (the systematic name is the PIN where the strict path builds it and OPSIN reads it back) carries over.
    The mutation check is the same test run on the START code (the name there is 'sparteine')."""
    assert_pin_at_both_tiers(SPARTEINE, SPARTEINE_PIN)
