# tests/unit/test_coverage_veto.py
""" a phase Task L1: the coverage AUDIT becomes a real VETO under
`ORTHONYM_COVERAGE_AUDIT=veto`.

L0 built the SHADOW audit at `namer.py::_finish` (telemetry only, never
changed the returned name). L1 flips `veto` mode from a no-op into a real
rejecter: when the computed `CoverageVerdict.complete is False` AND
`.method != "unavailable"`, `_finish` must route to the existing honest
abstain (`_descriptive_fallback`) instead of shipping the incomplete winner.

CARRIED CONSTRAINTS (bind this file too):
  * NEVER veto when `verdict.method == "unavailable"` -- that is the
    fail-OPEN signal (OPSIN absent / a skip-reanchor carve-out / gate
    disabled), and vetoing there would abstain names that are actually fine.
  * `shadow` mode must stay a pure no-op (L0 behaviour untouched).
  * A certificate exception must never veto (fail-open on error).
"""
import pytest

from orthonym.namer import Orthonym, is_failure_name

pytestmark = pytest.mark.unit

# A general-engine best-effort winner whose `certify_general_result` (E1 + binding spine) FAILS --
# `e1_spine` incomplete -- even though a name that OPSIN reads back to the canary ships under SHADOW.
#
# The canary that stood here, a stigmastane-type tetrol
# (C=C1[C@@H](O)CC[C@]2(C)C3=C(CC[C@@H]12)[C@]1(O)[C@@H](O)C[C@H]([C@H](C)CC[C@H](CC)C(C)C)[C@@]1(C)C[C@H]3O),
# is no longer a general-engine winner: the fused PIN offer names it by fusion nomenclature
# ('...-tetradecahydro-14H-cyclopenta[a]phenanthrene-3,11,14,15-tetrol', OPSIN verified), so its
# verdict is `self01`/complete and neither test below could reach the e1_spine path with it.
#
# The replacement still takes the general engine, and behaves name for name the same at 0c529ca8d
# and after the LO fix round, in both modes: a bridged hexacyclic ketal cage
# (hexacyclo[9.9.1.1^1,13.1^3,7.0^11,23.0^17,22], three ring oxygens) that none of the fusion
# builders names. It came from a scan of a dev split for general-engine winners with an incomplete
# e1_spine verdict (5 of the 53 e1_spine winners; the other four are
# COc1cc2c(c3oc(=O)c4c(c13)[C@@H](O)CC4=O)[C@@H]1C=CO[C@@H]1O2, the pyrene-5,6-dione
# O=c1ccc2cc1=O and two polycyclic quinone natural products). Both names
# below were read back to the canary's full InChIKey by OPSIN (asserted in the first test).
_CAGE_CANARY_SMILES = (
    "CO[C@@H]1CC(=O)c2cccc3c2[C@@]12Oc1ccc(O)c4c1[C@](O3)(O2)[C@H](O)[C@H](Cl)C4=O"
)
# The SHADOW-mode winner: the general-engine best-effort name whose E1 spine is incomplete (the two
# ketone groups cited as '-8,18-dione' suffixes).
_CAGE_MALFORMED_NAME = (
    "(1S,9S,10S,11S,20R)-9-chloro-6,10-dihydroxy-20-methoxy-2,12,21-trioxahexacyclo"
    "[9.9.1.1^1,13.1^3,7.0^11,23.0^17,22]tricosa-3(23),4,6,13(22),14,16-hexaene-8,18-dione"
)

# The VETO-mode rescue: when the veto rejects the incomplete-spine winner above,
# the L2-L4 offer layer (select_rt_passing, landed AFTER this test was authored)
# substitutes a DIFFERENT, complete, full-InChIKey-RT-verified floor candidate
# instead of abstaining -- so a name still ships and 0-wrong is held. It uses the
# dioxo PREFIX + '-ene' form rather than the '-dione' suffix.
_CAGE_VETO_RESCUE_NAME = (
    "(1S,9S,10S,11S,20R)-9-chloro-6,10-dihydroxy-20-methoxy-8,18-dioxo-2,12,21-trioxahexacyclo"
    "[9.9.1.1^1,13.1^3,7.0^11,23.0^17,22]tricosa-3(23),4,6,13(22),14,16-hexaene"
)


def _inchikey_of_smiles(smiles):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _inchikey_of_name(name):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = opsin_parse(name)
    return _inchikey_of_smiles(smiles) if smiles else None


def _canary_namer():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


class TestVetoRejectsIncompleteEGeneralEngineWinner:
    def test_e1_spine_incomplete_veto_mode_rescues_with_verified_complete(
            self, monkeypatch, opsin_gate):
        # Needs the REAL gate on (like L0's own leak-across-molecules test):
        # the suite's autouse gate-off default lets the legacy PIN pipeline
        # ship an un-suppressed WRONG candidate directly for this molecule,
        # so `general_fallback` (and thus the GeneralEngineResult / e1_spine
        # path this test targets) never engages.
        #
        # change-asserted-value: this test originally asserted the veto
        # ABSTAINS. Since then the L2-L4 offer layer (select_rt_passing over
        # the offers, incl. the floor) landed inside the same `_finish`
        # block, running unconditionally under general_fallback_unverified. So
        # the veto still REJECTS the incomplete-spine winner, but instead of
        # abstaining the offer layer now substitutes a DIFFERENT, complete,
        # full-InChIKey-RT-verified candidate -- a strict coverage improvement
        # with 0-wrong held (verify_or_none True). The veto's real invariant
        # (never SHIP the incomplete winner) is unchanged and still asserted.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")
        nm = _canary_namer()
        name = nm.name(_CAGE_CANARY_SMILES)
        verdict = nm._last_coverage_verdict
        # Sanity: the certificate really did flag this winner incomplete.
        assert verdict is not None
        assert verdict.method == "e1_spine"
        assert verdict.complete is False
        # The veto must have fired: the incomplete-spine winner must NOT ship.
        assert name != _CAGE_MALFORMED_NAME
        #... and the rescue is a real, RT-verified complete name, NOT a
        # sentinel abstain.
        assert not is_failure_name(name), (
            f"expected an RT-verified rescue, got abstain sentinel {name!r}")
        assert name == _CAGE_VETO_RESCUE_NAME
        # Both names describe the canary: OPSIN reads each back to its full InChIKey.
        assert _inchikey_of_name(name) == _inchikey_of_smiles(_CAGE_CANARY_SMILES)
        assert (_inchikey_of_name(_CAGE_MALFORMED_NAME)
                == _inchikey_of_smiles(_CAGE_CANARY_SMILES))

    def test_e1_spine_incomplete_shadow_mode_still_ships_it(self, monkeypatch,
                                                             opsin_gate):
        # L0 behaviour must be untouched: shadow NEVER vetoes.
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "shadow")
        nm = _canary_namer()
        name = nm.name(_CAGE_CANARY_SMILES)
        verdict = nm._last_coverage_verdict
        assert verdict is not None and verdict.complete is False
        assert name == _CAGE_MALFORMED_NAME


class TestVetoLeavesCompleteWinnersUnchanged:
    def test_complete_winner_veto_mode_ships_unchanged(self, monkeypatch):
        monkeypatch.setenv("ORTHONYM_COVERAGE_AUDIT", "veto")
        nm = Orthonym()
        name = nm.name("CCO")
        assert name == "ethanol"
        verdict = nm._last_coverage_verdict
        assert verdict is not None and verdict.complete is True


