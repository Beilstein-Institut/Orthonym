"""A ring prefix cites the locant of its free valence with (2)).

 Citation of locants (the Blue Book): "In preferred IUPAC names, if any locants
are essential for defining the structure of the parent structure or of a unit of structure as
defined by its appropriate enclosing marks, then all locants must be cited for the parent
structure or that structural unit."
"""
import pytest
from rdkit import Chem

from orthonym.rules.ring_substituents import get_ring_substituent_name

from ._helpers import ambiguity_messages, assert_clean, name_row, round_trips

# the names are the shipped ones: the OPSIN validity gate is on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,ring,attach,expected", [
    ("COC(=O)CN1CC1", (5, 6, 7), 5, "aziridin-1-yl"),
    ("O=C(C1CC1)C1CCO1", (5, 6, 7, 8), 5, "oxetan-2-yl"),
    ("CC(C#N)N1CCC1", (4, 5, 6, 7), 4, "azetidin-1-yl"),
])
def test_get_ring_substituent_name_cites_locant(smiles, ring, attach, expected):
    # General methodology for naming substituent groups, method (2)
    # (the Blue Book): "...except for mononuclear parent hydrides or the suffix
    # 'ylidyne', the locant '1' must be cited."
    m = Chem.MolFromSmiles(smiles)
    assert get_ring_substituent_name(m, ring, attachment_point=attach) == expected


@pytest.mark.parametrize("smiles,must", [
    ("COC(=O)CN1CC1", "aziridin-1-yl"),
    ("CC(C#N)N1CCC1", "azetidin-1-yl"),
    ("O=C(C1CC1)C1CCO1", "oxetan-2-yl"),
    ("C1CC(O1)C1=CON=N1", "oxetan-2-yl"),
    ("COC=NC1=NC=CO1", "1,3-oxazol-2-yl"),
    ("OC1=NNC=C1N1CC1", "aziridin-1-yl"),
    ("OCC1COC1", "oxetan-3-yl"),
])
def test_flagged_ring_prefixes_are_clean(smiles, must):
    # (2), the Blue Book (see above): the free-valence locant is cited
    assert_clean(smiles, must_contain=must)


@pytest.mark.parametrize("smiles,must", [
    ("COC=NC1=NC=CN1", "1H-imidazol-2-yl"),
    ("COC=NC1=CC=CO1", "furan-2-yl"),
    ("COC=NC1=CC=CN1", "1H-pyrrol-2-yl"),
    ("COC=NC1=NOC=C1", "1,2-oxazol-3-yl"),
    ("COC=NC1=NOC=N1", "1,2,4-oxadiazol-3-yl"),
    ("NCC(Cc1ccccc1)Nc1ncccn1", "pyrimidin-2-yl"),
    ("c1ccc(C=Nc2ccccn2)cc1", "pyridin-2-yl"),
])
def test_ring_prefix_on_an_amine_nitrogen_cites_locant(smiles, must):
    # (2), the Blue Book (see above): a ring joined to a nitrogen is a
    # ring prefix like any other, with the locant of its free valence
    assert_clean(smiles, must_contain=must)


def test_polysubstituted_ring_prefixes_are_preferred_names():
    # (2), the Blue Book (see above); multiplying prefix 'tris' with the
    # enclosing marks of around a prefix that cites a locant
    row = assert_clean("O=C1C=C(N2CC2)C(=O)C(N2CC2)=C1N1CC1", tier="pin")
    assert row["name"] == "2,3,5-tris(aziridin-1-yl)cyclohexa-2,5-diene-1,4-dione"
    assert row["tier"] == "pin_verified"


def test_ring_on_carbon_branch_fallback_cites_locant():
    # (2), the Blue Book (see above): the ring atom bonded to the branch
    # atom carries the free valence
    from orthonym.assembly.composer import _name_c_attached_ring_substituent_fallback
    mol = Chem.MolFromSmiles("CC(O)C1CN1")
    ring = {3, 4, 5}
    assert _name_c_attached_ring_substituent_fallback(
        mol, sorted(ring), ring, {0, 1, 2}, 1) == "(aziridin-2-yl)"


def test_single_ring_classification_cites_locant(monkeypatch):
    # (2), the Blue Book (see above): when the ring-system namer has no name,
    # the single-ring lookup still receives the ring atom bonded to the parent
    from orthonym.perception.chains import classify_substituent
    monkeypatch.setattr("orthonym.rules.ring_substituents.name_ring_system_substituent",
                        lambda *args, **kwargs: None)
    mol = Chem.MolFromSmiles("CC(O)C1CN1")
    info = classify_substituent(mol, [3, 4, 5], {0, 1, 2})
    assert info["type"] == "ring" and info["name"] == "aziridin-2-yl"


@pytest.mark.parametrize("smiles,must", [
    ("Nc1ccccc1P(c1ccccc1)c1cccc2ccccc12", "(naphthalen-1-yl)(phenyl)phosphanyl"),
    ("Nc1ccccc1P(c1cccc2ccccc12)c1cccc2ccccc12", "di(naphthalen-1-yl)phosphanyl"),
    ("Nc1ccccc1P(c1ccccc1)c1ccc2ccccc2c1", "(naphthalen-2-yl)(phenyl)phosphanyl"),
])
def test_naphthalene_on_phosphorus_cites_locant(smiles, must):
    # (2), the Blue Book (see above); 'naphthalen-2-yl (preferred prefix)'
    # (the Blue Book). Default tier too: the name is the preferred name.
    assert_clean(smiles, tier="pin", must_contain=must)


@pytest.mark.parametrize("smiles,name", [
    ("OCC1CO1", "oxiranylmethanol"),
    ("OCC1CCC1", "cyclobutylmethanol"),
    ("OCC1CC1", "cyclopropylmethanol"),
    ("OCc1ccccc1", "phenylmethanol"),
])
def test_symmetric_rings_keep_locant_free_form(smiles, name):
    # (the Blue Book): "The locant is omitted in monosubstituted symmetrical
    # parent hydrides or parent compounds where there is only one kind of substitutable
    # hydrogen."; example '(oxiranylmethoxy)silane (PIN)' (:18929)
    assert name_row(smiles)["name"] == name


@pytest.mark.parametrize("smiles,must", [
    ("OC(=O)CN1C[13CH2]1", "aziridin-1-yl"),
    ("OCC1[13CH2]O1", "oxiran-2-yl"),
    ("OCC1O[13CH2]1", "oxiran-2-yl"),
    ("OCC1OC1[2H]", "oxiran-2-yl"),
])
def test_labelled_ring_never_locant_free(smiles, must):
    # Letter and/or numeral locants (the Blue Book),:44210: "When locants are
    # needed for defining the structure of the parent structure or of a unit of structure as
    # defined by its appropriate enclosing marks, then all locants must be cited for the
    # parent structure or that structural unit." A ring position that carries an isotope label
    # is its own kind of position, so (:2939) gives the labelled ring no
    # locant-free prefix ('oxiranyl'): OPSIN reads '(3-13C)oxiranylmethanol' as ambiguous.
    assert_clean(smiles, must_contain=must)
    row = name_row(smiles, tier="pin")
    if row["tier"] not in (None, "abstain"):
        assert not ambiguity_messages(row["name"]), row


def test_ring_prefix_of_a_labelled_molecule_is_not_locant_free():
    # (the Blue Book, sentence in the test above): the ring prefix function
    # reads the labels of the molecule it is given and never returns the locant-free form
    m = Chem.MolFromSmiles("OCC1[13CH2]O1")
    assert get_ring_substituent_name(m, (2, 3, 4), attachment_point=2) != "oxiranyl"
    m = Chem.MolFromSmiles("OCC1CO1")
    assert get_ring_substituent_name(m, (2, 3, 4), attachment_point=2) == "oxiranyl"


_LABELLED_RING_SMILES = "OC(=O)CN1C[13CH2]1"
_LABELLED_RING_BOOK_NAME = "[(2-13C)aziridin-1-yl]acetic acid"


def test_labelled_ring_book_name_describes_the_input():
    # the book name of the labelled ring prefix, pinned in test_amb5_isotope_locants.py, is a
    # name of the input structure: OPSIN reads it without a warning and returns the input's
    # InChIKey
    assert not ambiguity_messages(_LABELLED_RING_BOOK_NAME)
    assert round_trips(_LABELLED_RING_SMILES, _LABELLED_RING_BOOK_NAME)


_STEREO_RING_SMILES = "COC(=O)c1cc2nc(CCl)n(C[C@@H]3CCO3)c2s1"
_STEREO_RING_BOOK_NAME = (
    "methyl 2-(chloromethyl)-3-{[(2S)-oxetan-2-yl]methyl}-3H-thieno[2,3-d]imidazole-5-carboxylate"
)


def test_stereo_ring_prefix_round_trips():
    # (2), the Blue Book (see above): the prefix cites its locant. OPSIN reads a
    # stereodescriptor wherever it stands, so this test pins the structure and the locant; the
    # placement of the descriptor is pinned by test_stereo_descriptor_precedes_its_ring_prefix.
    row = assert_clean(_STEREO_RING_SMILES, must_contain="oxetan-2-yl")
    assert round_trips(_STEREO_RING_SMILES, row["name"])


def test_stereo_ring_book_name_describes_the_input():
    # the book name of the stereo ring prefix (the descriptor '(2S)' at the front of the
    # oxetan-2-yl prefix, is a name of the input structure: OPSIN reads it without a
    # warning and returns the input's InChIKey
    assert not ambiguity_messages(_STEREO_RING_BOOK_NAME)
    assert round_trips(_STEREO_RING_SMILES, _STEREO_RING_BOOK_NAME)


_P91_3 = ("P-91.3 NAMING OF STEREOISOMERS (BlueBookV2.md:44639), paragraph 2 (:44643): \"In preferred "
          "IUPAC names, stereodescriptors are placed immediately at the front of the part of the name "
          "to which they relate. They are placed at the front of the complete name when related to the "
          "parent structure; ... When they relate to substituent groups, they are cited at the front of "
          "the corresponding prefix. They are preceded by a numerical or letter locant ...\"")


def test_stereo_descriptor_precedes_its_ring_prefix():
    # (see _P91_3); example under (:7489):
    # '10-{[(3S)-1-phosphabicyclo[2.2.2]octan-3-yl]methyl}-10H-phenoxazine (PIN)'.
    # The stereocentre is C-2 of the oxetane, so '(2S)' belongs inside the substituent.
    assert name_row(_STEREO_RING_SMILES)["name"] == _STEREO_RING_BOOK_NAME


def test_default_tier_stereo_descriptor_precedes_its_ring_prefix():
    # (see _P91_3): the default tier ships the name with the descriptor in front of the
    # ring prefix, or declines; it never ships '(2S)' in front of the whole name, where it is read
    # as a descriptor of the parent (C-2 of the thieno-imidazole is not a stereocentre).
    row = name_row(_STEREO_RING_SMILES, tier="pin")
    assert row["tier"] == "abstain" or row["name"] == _STEREO_RING_BOOK_NAME, row


def _descriptors_stand_in_ring_prefixes(name, ring="oxetan|oxolan|aziridin", parent_block=False):
    """Every ring prefix of the name carries its descriptor at its front, and, unless the parent
    has a stereocentre of its own (``parent_block``), no descriptor block stands in front of
    the whole name."""
    import re
    prefixes = re.findall(r"(?:\(\d[RS]\)-)?(?:%s)[a-z]*-\d-yl" % ring, name)
    return (bool(prefixes) and all(p.startswith("(") for p in prefixes)
            and (parent_block or not re.match(r"^(?:[a-z0-9,\- ]+ )?\(\d[RS]\)-", name)))


@pytest.mark.parametrize("smiles,expected", [
    # shape (b) of the census: the descriptor of the ring carbon was written without a locant
    # in front of a compound prefix ('[(S)-(oxetan-2-yl)methyl]')
    ("CS(=O)OC[C@@H]1CCO1", "[(2S)-oxetan-2-yl]methyl methanesulfinate"),
    ("CC(=O)NC[C@@H]1CCO1", "N-{[(2S)-oxetan-2-yl]methyl}acetamide"),
    ("CC(=O)NCC[C@@H]1CCO1", "N-{2-[(2R)-oxetan-2-yl]ethyl}acetamide"),
    # shape (a): the descriptor of the ring carbon stood in front of the whole name
    ("OC[C@@H]1CCO1", "[(2S)-oxetan-2-yl]methanol"),
    ("OC[C@@H]1CN1", "[(2S)-aziridin-2-yl]methanol"),
    ("OC[C@@H]1CCCO1", "[(2S)-oxolan-2-yl]methanol"),
    ("COC(=O)c1cc2nc(CCl)n(C[C@@H]3CCCO3)c2s1",
     "methyl 2-(chloromethyl)-3-{[(2S)-oxolan-2-yl]methyl}-3H-thieno[2,3-d]imidazole-5-carboxylate"),
    ("CC(=O)N(C)[C@@H]1CCO1", "N-methyl-N-[(2S)-oxetan-2-yl]acetamide"),
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_descriptor_of_a_substituent_ring_stands_in_its_prefix(smiles, expected, tier):
    # (see _P91_3). The default tier ships the name or declines.
    row = name_row(smiles, tier=tier)
    if tier == "pin" and row["tier"] == "abstain":
        return
    assert row["name"] == expected, row
    assert not ambiguity_messages(row["name"]), row
    assert round_trips(smiles, row["name"]), row


@pytest.mark.parametrize("smiles", [
    "CC(C)OC(=O)c1ccc2nc(C(C)Cl)n(C[C@@H]3CCO3)c2n1",                     # id 1323
    "CC1=NN(C(=C1)C2=CN(C(=N2)C3=C4C=NN(C4=CC(=N3)C(=O)N)C)C)C[C@H]5CCO5",  # id 3015
    "COC(=O)C1=CC2=C(C=C1)N=C(N2C[C@@H]3CCO3)CC4=CC(=C(C=C4F)C5=CC=CC(=O)N5)F",  # id 4601
])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_fused_parent_with_a_stereo_oxetane_branch_cites_the_descriptor_in_the_prefix(smiles, tier):
    # (see _P91_3): the parent of these names is a fused heteroaromatic ring system whose
    # position 2 is not a stereocentre, so the '(2S)' of a pin_verified name is the descriptor of
    # the oxetane carbon and belongs in front of the oxetane prefix.
    row = name_row(smiles, tier=tier)
    if tier == "pin" and row["tier"] == "abstain":
        return
    assert _descriptors_stand_in_ring_prefixes(row["name"]), row
    assert not ambiguity_messages(row["name"]), row
    assert round_trips(smiles, row["name"]), row


@pytest.mark.parametrize("smiles", [
    "C[C@H]1CCCN1C[C@H]1CCO1",
    "OC(=O)[C@H]1CCCN1C[C@H]1CCO1",
    "COC(=O)[C@H]1CCCN1C[C@H]1CCCO1",
])
def test_parent_and_ring_prefix_each_cite_their_own_descriptor(smiles):
    # (see _P91_3): "They are placed at the front of the complete name when related to the
    # parent structure;... When they relate to substituent groups, they are cited at the front of
    # the corresponding prefix."
    row = assert_clean(smiles, tier="pin")
    assert _descriptors_stand_in_ring_prefixes(row["name"], ring="oxetan|oxolan", parent_block=True), row
    import re
    assert len(re.findall(r"\(\d[RS]\)", row["name"])) == 2, row      # the parent's and the prefix's


def test_ring_yl_attachment_descriptor_cites_the_locant_of_the_prefix():
    # (see _P91_3): the locant of the descriptor is the locant the ring prefix cites for
    # its free valence ('oxetan-2-yl' -> '(2S)'); a ring whose free valence is not its only
    # stereogenic unit, or whose prefix is not the form the caller prints, has none.
    from orthonym.assembly.substituent_naming import ring_yl_attachment_descriptor
    m = Chem.MolFromSmiles("OC[C@@H]1CCO1")
    from orthonym.perception.stereo import assign_stereochemistry
    assign_stereochemistry(m)
    ring = [2, 3, 4, 5]
    assert ring_yl_attachment_descriptor(m, ring, 2, ring_name="oxetan-2-yl") == "(2S)"
    assert ring_yl_attachment_descriptor(m, ring, 2, ring_name="oxetanyl") is None
    assert ring_yl_attachment_descriptor(m, ring, 2) is None
    two = Chem.MolFromSmiles("OC[C@@H]1C[C@H](C)CO1")                  # a second stereocentre in the ring
    assign_stereochemistry(two)
    assert ring_yl_attachment_descriptor(two, [2, 3, 4, 6, 7], 2, ring_name="oxolan-2-yl") is None
    m2 = Chem.MolFromSmiles("CC(=O)NC[C@H]1C=CCC1")
    assign_stereochemistry(m2)
    assert ring_yl_attachment_descriptor(m2, [5, 6, 7, 8, 9], 5) == "(1R)"
