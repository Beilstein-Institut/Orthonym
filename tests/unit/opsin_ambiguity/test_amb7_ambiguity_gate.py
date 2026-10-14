"""The final validity check withdraws a name that OPSIN reads as ambiguous, after one retry
with every locant cited.

 Citation of locants (the Blue Book): "In preferred IUPAC names, if any locants
are essential for defining the structure of the parent structure or of a unit of structure as
defined by its appropriate enclosing marks, then all locants must be cited for the parent
structure or that structural unit." OPSIN raises ``APPEARS_AMBIGUOUS`` when it had to choose
among several structures for a part of a name, which is what a name does that omits an
essential locant.
"""
import pytest

from orthonym import Orthonym, namer
from orthonym.errors import is_failure_name
from orthonym.jvm_bridge import OpsinUnavailable
from orthonym.metrics import provenance as pv
from orthonym.validation import opsin_ambiguity

from ._helpers import ambiguity_messages, assert_clean, name_row, round_trips

# the names are the shipped ones: the OPSIN validity gate is on (tests/conftest.py)
pytestmark = pytest.mark.opsin_gate

AZIRIDINE = "COC(=O)CN1CC1"
AZIRIDINE_AMBIGUOUS = "methyl aziridinylacetate"
AZIRIDINE_BOOK = "methyl (aziridin-1-yl)acetate"


def _general_engine() -> Orthonym:
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True)


def _gate(name: str) -> str:
    """The best-effort validity gate on ``name`` for the aziridine input."""
    return namer._final_opsin_validity_gate(name, AZIRIDINE, besteffort_unverified=True,
                                            general_fallback_tier=True)


def test_ambiguity_messages_read_the_warning():
    #: 'aziridinyl' names no position of the ring
    assert opsin_ambiguity.ambiguity_messages(AZIRIDINE_AMBIGUOUS)
    assert opsin_ambiguity.ambiguity_messages(AZIRIDINE_BOOK) == ()


def test_ambiguity_messages_are_empty_when_in_process_opsin_cannot_answer(monkeypatch):
    # an unavailable OPSIN is no evidence of ambiguity; the round trip fails closed on it
    def unavailable(name, allow_radicals=True):
        raise OpsinUnavailable("in-process OPSIN not available")

    monkeypatch.setattr("orthonym.jvm_bridge.opsin_warnings", unavailable)
    assert opsin_ambiguity.ambiguity_messages(AZIRIDINE_AMBIGUOUS) == ()


def test_gate_suppresses_an_ambiguous_name():
    #: the ring prefix lacks the locant of its free valence
    out = _gate(AZIRIDINE_AMBIGUOUS)
    assert is_failure_name(out), out
    row = pv.get_provenance()
    assert pv.resolve_gate_outcome(row["gate_outcome"], row["gate_outcome_name"], out) == \
        pv.GATE_OUTCOME_AMBIGUOUS_SUPPRESSED


def test_gate_suppresses_an_unbracketed_compound_prefix():
    #: '1-iodoethynyl' can be read as 1-iodo plus 'ethynyl' with no locant
    name = "1-iodoethynyl-4-methylbenzene"
    assert is_failure_name(namer._final_opsin_validity_gate(name, "Cc1ccc(C#CI)cc1"))


@pytest.mark.parametrize("name,smiles", [(AZIRIDINE_BOOK, AZIRIDINE), ("ethanol", "CCO")])
def test_gate_ships_a_name_that_opsin_reads_once(name, smiles):
    #: a name that cites its locants is read once by OPSIN
    assert namer._final_opsin_validity_gate(name, smiles) == name
    row = pv.get_provenance()
    assert pv.resolve_gate_outcome(row["gate_outcome"], row["gate_outcome_name"], name) == \
        pv.GATE_OUTCOME_SELF01


@pytest.mark.parametrize("smiles", [
    "O=C(C1CC1)C1CCO1", "CPC1=NN=N[NH2+]1", "C1C(=O)NC(=O)[15NH]1",
    "NP(=O)(c1ccc(C(F)(F)F)cc1)c1ccc(C(F)(F)F)cc1",                         # 'P,P-bis[...]phosphinic amide'
    "C1CCCC(CC1)(C(=O)[O-])[P+](c2ccccc2)(c2ccccc2)c2ccccc2",                # '1-(...)cycloheptane...'
    "C1CCC(CC1)S(=O)(=O)c1ccc(cc1)[S+](c1ccccc1)c1ccccc1[O-]",              # '2-{...}phenolate'
    "CC1=CC=C(C=C1)[13CH](C)OC(=[18O])C2=C(C(=C(C(=C2F)F)F)F)F",            # the label on one oxygen
])
def test_default_tier_never_ships_ambiguous(smiles):
    #: the default tier ships a preferred name or declines
    row = name_row(smiles, tier="pin")
    if row["tier"] not in (None, "abstain"):
        assert not ambiguity_messages(row["name"]), row


def test_gate_retries_with_forced_locants(monkeypatch):
    # A licence that omits a locant without asking the scope: the retry switches every licence
    # off (``forced_locant_scope``), so the name cites its locants.
    from orthonym.assembly import locant_omission as lo
    real = lo.l4_no_isomer_by_relocation
    monkeypatch.setattr(lo, "l4_no_isomer_by_relocation",
                        lambda *a, **k: (not lo.locants_are_forced()) or real(*a, **k))
    row = name_row("C1CC(=C(C2=CC=CC=C2)C(=O)C3=CC=CC=C3)C1")
    assert row["tier"] not in (None, "abstain")
    assert not ambiguity_messages(row["name"]), row


def test_the_retry_runs_once_with_locants_forced(monkeypatch):
    #: one retry with every locant cited, then the name is withdrawn
    from orthonym.assembly import locant_omission as lo
    calls = []
    eng = _general_engine()

    def always_ambiguous(smiles):
        calls.append(lo.locants_are_forced())
        return AZIRIDINE_AMBIGUOUS

    monkeypatch.setattr(eng, "_name_impl", always_ambiguous)
    gated = _gate(AZIRIDINE_AMBIGUOUS)
    out = eng._retry_cascade_on_gate_rejection(AZIRIDINE, pre_gate=AZIRIDINE_AMBIGUOUS, gated=gated)
    assert is_failure_name(out)
    forced = [c for c in calls if c]
    assert len(forced) == 1, calls
    assert not lo.locants_are_forced()


def test_the_retry_does_not_run_for_another_rejection(monkeypatch):
    #: only a name OPSIN reads as ambiguous is named again with every locant cited
    from orthonym.assembly import locant_omission as lo
    calls = []
    eng = _general_engine()
    monkeypatch.setattr(eng, "_name_impl", lambda smiles: calls.append(lo.locants_are_forced()))
    gated = _gate("notaname")
    eng._retry_cascade_on_gate_rejection(AZIRIDINE, pre_gate="notaname", gated=gated)
    assert not any(calls), calls


@pytest.mark.parametrize("smiles", [
    "CC(=O)NC(CSSC(C)=O)C(C)=O",
    "C1CC(=C(C2=CC=CC=C2)C(=O)C3=CC=CC=C3)C1",
    "NP(=O)(c1ccc(C(F)(F)F)cc1)c1ccc(C(F)(F)F)cc1",
    "CC(C(=O)NC1(CCCC1)CC2=CC=CC=C2)N3C=NC=N3",
])
def test_best_effort_ships_no_ambiguous_name(smiles):
    #: a name that ships is read once by OPSIN; a molecule with no such name is not named
    row = name_row(smiles)
    assert row["tier"] == "abstain" or not ambiguity_messages(row["name"]), row


def test_exit_withdraws_an_ambiguous_name_that_never_met_the_gate():
    #: a name from a producer after the gate still ends at the final check, which reads
    # OPSIN's warnings too; the round trip of this name is exact
    eng = _general_engine()
    assert eng._exit_round_trip_check(AZIRIDINE_BOOK, AZIRIDINE) == AZIRIDINE_BOOK
    assert is_failure_name(eng._exit_round_trip_check(AZIRIDINE_AMBIGUOUS, AZIRIDINE))


def test_a_clean_molecule_is_named_as_before():
    #: the check changes no name that OPSIN reads once
    assert_clean(AZIRIDINE, must_contain="aziridin-1-yl")


# the label on the carbonyl oxygen of an ester: OPSIN reads every spelling that gives the input
# ('(18O)', '[18O]', '(18O1)', 'benzene-1-(18O)carboxylate') as ambiguous, and rejects the ones
# with a locant
LABELLED_ESTER = "CC1=CC=C(C=C1)[13CH](C)OC(=[18O])C2=C(C(=C(C(=C2F)F)F)F)F"


def test_best_effort_keeps_the_name_that_round_trips_and_labels_it_below_the_pin():
    # (the Blue Book) writes '[18O]phosphonate' for the same reason: the label on
    # one oxygen of a group has no locant OPSIN reads. At the best-effort tier the name is better
    # than none; it is not certified as the preferred name.
    row = name_row(LABELLED_ESTER, "best-effort")
    assert row["tier"] not in (None, "abstain", "pin_verified", "pin_unverified"), row
    assert round_trips(LABELLED_ESTER, row["name"]), row
    assert ambiguity_messages(row["name"]), row


def test_default_tier_declines_the_name_that_opsin_reads_as_ambiguous():
    # The default tier ships a preferred name or declines (user ruling 2026-09-30). The name of
    # the labelled ester is valid, OPSIN reads it as ambiguous, and no clean name exists, so it
    # is not certified as the preferred name: the engine built a name that is not a verified PIN
    # (NO_VERIFIED_PIN).
    row = name_row(LABELLED_ESTER, "pin")
    assert row["tier"] == "abstain", row
    assert row["limit_code"] == "NO_VERIFIED_PIN", row


def test_a_name_kept_as_the_last_resort_is_checked_like_any_name():
    #: the name kept for the molecule is not withdrawn again by the exit of the call
    eng = _general_engine()
    assert eng.name(LABELLED_ESTER, on_failure="label") == eng.name_with_confidence(LABELLED_ESTER)["name"]


def test_a_withdrawn_name_is_restored_only_for_a_molecule_without_a_name():
    #: a name OPSIN reads once stands over the withdrawn one
    from orthonym.assembly import memo
    token = memo.push_scope()
    try:
        assert is_failure_name(_gate(AZIRIDINE_AMBIGUOUS))
        assert opsin_ambiguity.restore_withdrawn(AZIRIDINE_BOOK, AZIRIDINE) == AZIRIDINE_BOOK
        assert opsin_ambiguity.restore_withdrawn("unknown organic compound", AZIRIDINE) == \
            AZIRIDINE_AMBIGUOUS
    finally:
        memo.pop_scope(token)


def test_a_withdrawn_name_is_restored_only_when_it_is_verified(monkeypatch):
    #: the last resort is a name that round-trips; a gate outcome that verified nothing
    # needs the full round trip
    from orthonym.assembly import memo
    token = memo.push_scope()
    try:
        assert is_failure_name(_gate(AZIRIDINE_AMBIGUOUS))
        monkeypatch.setattr(pv, "opsin_label_for_gate_outcome", lambda outcome: "unverified")
        monkeypatch.setattr(namer, "_shipped_name_round_trip", lambda name, smiles: "failed")
        failure = "unknown organic compound"
        assert opsin_ambiguity.restore_withdrawn(failure, AZIRIDINE) == failure
        monkeypatch.setattr(namer, "_shipped_name_round_trip", lambda name, smiles: "verified")
        assert opsin_ambiguity.restore_withdrawn(failure, AZIRIDINE) == AZIRIDINE_AMBIGUOUS
    finally:
        memo.pop_scope(token)


@pytest.mark.parametrize("smiles,name", [
    ("[CH2+]CC", "propylium"),
    ("CC[CH2+]", "propylium"),
    ("[CH2+]CCC", "butylium"),
    ("[CH2+]CCCC", "pentylium"),
])
def test_the_default_tier_keeps_a_preferred_name_that_opsin_reads_as_ambiguous(smiles, name):
    # 'propylium (PIN)' is printed under Specific method (the Blue Book):
    # "Cations formed formally by the removal of a hydride ion, H-, from a terminal atom of a
    # saturated unbranched acyclic hydrocarbon... are named by replacing the 'ane' ending in
    # the name of the parent hydride by the suffix 'ylium'" (example:41557). OPSIN reads the
    # name as ambiguous and gives the right structure; the method names the terminal atom by
    # definition, so the name has one position and is not withdrawn.
    assert ambiguity_messages(name)
    row = name_row(smiles, tier="pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert opsin_ambiguity.licensed_ambiguity(name, smiles) == "P-73.2.2.1.1"


@pytest.mark.parametrize("smiles", ["C[CH+]C", "CC[CH+]CC", "[CH2+]C(C)C"])
def test_a_name_of_another_class_is_not_licensed(smiles):
    # names the terminal atom of an unbranched alkane; the cation of any other
    # structure has no licence for a locant-free name
    assert opsin_ambiguity.licensed_ambiguity("propylium", smiles) is None


@pytest.mark.parametrize("smiles,name", [
    ("[18O-][N+](=O)c1ccccc1", "(18O)nitrobenzene"),
    ("[18O]=[N+]([O-])c1ccccc1", "(18O)nitrobenzene"),
    ("CC(=O)[18O-].[Na+]", "sodium (18O)acetate"),
    ("NC(N)=[15NH2+].[Cl-]", "(15N)guanidinium chloride"),
])
def test_a_label_on_a_delocalized_group_is_licensed(smiles, name):
    # DELOCALIZED RADICALS AND IONS (the Blue Book): "Delocalization in names involving
    # one radical or ionic center in an otherwise conjugated double bonds structure is denoted by
    # the appropriate suffix without locants." The compound is one species whatever atom a drawing
    # puts the charge on, so the descriptor has no position to cite; OPSIN reads it as ambiguous.
    assert ambiguity_messages(name)
    assert opsin_ambiguity.licensed_ambiguity(name, smiles) == "P-76"
    row = name_row(smiles, tier="pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_label_on_one_oxygen_of_an_ester_is_not_licensed():
    # the carbonyl oxygen and the oxygen of the ester group are two atoms of two kinds: the label
    # moved to the other gives another compound, so the descriptor has a position to cite
    name = "1-(4-methylphenyl)(1-13C)ethyl 2,3,4,5,6-pentafluoro(18O)benzoate"
    assert ambiguity_messages(name)
    assert opsin_ambiguity.licensed_ambiguity(name, LABELLED_ESTER) is None


@pytest.mark.parametrize("smiles,name", [
    ("CCCC(=O)SOOC", "[1-(methylperoxy)sulfanyl]butan-1-one"),
    ("CCCC(=O)[Se]OOC", "[1-(methylperoxy)selanyl]butan-1-one"),
    ("CCCC(=O)[Te]OOC", "[1-(methylperoxy)tellanyl]butan-1-one"),
    ("CCCC(=O)SOOC1CC1", "[1-(cyclopropylperoxy)sulfanyl]butan-1-one"),
])
def test_the_pseudoester_of_a_peroxy_chain_is_licensed(smiles, name):
    # Pseudoesters (the Blue Book): "When 'E' is an element belonging to Group
    # 16, the pseudoester is named as a pseudoketone", example '[1-(methylperoxy)sulfanyl]butan-1-one
    # (PIN)' (:32078). OPSIN reads the name as ambiguous and gives the structure of the input; the
    # book prints it, so it is not withdrawn.
    assert ambiguity_messages(name)
    assert opsin_ambiguity.licensed_ambiguity(name, smiles) == "P-65.6.3.4.2"
    assert namer._final_opsin_validity_gate(name, smiles) == name


@pytest.mark.parametrize("name,smiles", [
    # (the Blue Book): the diacyl analogue is spelled with the locant in front
    ("[1-(acetylperoxy)sulfanyl]ethan-1-one", "CC(=O)OOSC(C)=O"),
    # (:39473): a chalcogen chain that does not start with a peroxy group, example
    # '1-{[(methylsulfanyl)oxy]selanyl}propan-1-one (PIN)' (:39497)
    ("{1-[(methylsulfanyl)oxy]selanyl}propan-1-one", "CCC(=O)[Se]OSC"),
    # the name of the class, for another structure of the class
    ("[1-(methylperoxy)sulfanyl]butan-1-one", "CCCC(=O)SOOCC"),
])
def test_a_pseudoketone_of_another_class_is_not_licensed(name, smiles):
    # (the Blue Book) licenses the pseudoester of a peroxy chain that ends in
    # an organyl group, and the name its producer builds for that structure
    assert opsin_ambiguity.licensed_ambiguity(name, smiles) is None
    assert is_failure_name(namer._final_opsin_validity_gate(name, smiles))


@pytest.mark.parametrize("smiles,is_class", [
    ("CCCC(=O)SOOC", True),
    ("CCCC(=O)[Se]OOC", True),
    ("CC(=O)OOSC(C)=O", False),               # R' is an acyl group:
    ("CCC(=O)[Se]OSC", False),                # the chain does not start with -O-O-:
    ("CCCC(=O)SSOC", False),
    ("CCCC(=O)SOOS(=O)(=O)C", False),
    ("CCCCCC(=O)SCC", False),                 # a thioester
    ("CCCC(=O)OOC", False),                   # the acyl group is on oxygen
])
def test_the_class_of_the_pseudoester_of_a_peroxy_chain(smiles, is_class):
    # Pseudoesters (the Blue Book)
    from rdkit import Chem

    from orthonym.rules.pseudoketones import pseudoester_book_name
    assert (pseudoester_book_name(Chem.MolFromSmiles(smiles)) is not None) is is_class


@pytest.mark.parametrize("smiles", [
    "c1ccccc1C(=O)SOOC",                       # benzoyl
    "C1CCCCC1C(=O)SOOC",                       # cyclohexanecarbonyl
    "O=C(SOOC)C1CC1",                          # cyclopropanecarbonyl
    "c1ccccc1C(=O)[Te]OOC",
    "c1ccccc1CC(=O)SOOC",                      # the ring is the end of the chain of the acyl group
    "c1ccccc1C(=O)[SiH3]",
    "C1CCCCC1C(=O)[PH2]",
])
def test_the_acyl_group_of_a_ring_is_not_the_acyl_chain_of_a_pseudoketone(smiles):
    # Acyl groups derived from systematically named carboxylic acids
    # (the Blue Book): the acyl group of a ring is named from the ring and the suffix
    # 'carbonyl', example 'cyclohexanecarbonyl (preferred prefix)' (:30628); no chain of the
    # acyl group runs through a ring, so the chain pseudoketone producers decline
    from rdkit import Chem

    from orthonym.rules.pseudoketones import name_acyl_hetero_pseudoketone, pseudoester_book_name
    mol = Chem.MolFromSmiles(smiles)
    assert pseudoester_book_name(mol) is None
    assert name_acyl_hetero_pseudoketone(mol) is None


@pytest.mark.parametrize("smiles,name", [
    ("c1ccccc1C(=O)SOOC", "[(methylperoxysulfanyl)(oxo)methyl]benzene"),
    ("C1CCCCC1C(=O)SOOC", "[(methylperoxysulfanyl)(oxo)methyl]cyclohexane"),
    ("O=C(SOOC)C1CC1", "[(methylperoxysulfanyl)(oxo)methyl]cyclopropane"),
])
def test_the_pseudoester_of_a_ring_acyl_group_keeps_its_verified_best_effort_name(smiles, name):
    # (the Blue Book): the acyl group of a ring is not a chain, so the
    # molecule is outside the class of the licence of and keeps the name of
    # another producer, which OPSIN reads as the input
    row = name_row(smiles)
    assert row["name"] == name, row
    assert round_trips(smiles, row["name"]), row
    assert not ambiguity_messages(row["name"]), row


@pytest.mark.parametrize("smiles,delocalized", [
    ("[18O-][N+](=O)c1ccccc1", True),
    ("CC(=O)[18O-]", True),
    ("CC(=[18O])OC", False),                  # an ester: the two oxygens differ
    ("c1ccccc1[13CH3]", False),               # one carbon of a kind
    ("[13CH3]c1ccccc1", False),
])
def test_heavy_label_on_delocalized_atoms(smiles, delocalized):
    from rdkit import Chem

    from orthonym.rules.isotopes import heavy_label_on_delocalized_atoms
    assert heavy_label_on_delocalized_atoms(Chem.MolFromSmiles(smiles)) is delocalized
