"""Unit tests for CandidatePool, HandlerPolicy, HANDLER_POLICIES, and the
thread-local pool store (a phase).

Tests behavior-preserving extraction of composer.py handler cascade.
Critical regression tests:
- test_pool_add_does_NOT_pass_parent_atom_indices_* proves Risk 1 mitigation
- test_module_level_parent_correctness_binding proves remediation
"""

import threading

import pytest
from rdkit import Chem

from orthonym.assembly.candidate_pool import (
    CandidatePool,
    HandlerPolicy,
    HANDLER_POLICIES,
    clear_pool,
    get_current_pool,
    _pool_store,
)
import orthonym.assembly.candidate_pool as cp_module
from orthonym.assembly.coverage_scoring import (
    CONFIDENCE_GATE_THRESHOLD,
    CandidateName,
    HANDLER_PRIORITY,
    compute_confidence,
)
from orthonym.namer import MolecularFeatures


# ---------------------------------------------------------------------------
# Test helper (mirrors tests/unit/rules/test_coverage_scoring.py pattern)
# ---------------------------------------------------------------------------

def _make_features(smiles, functional_groups=None, principal_group=None, **extra):
    """Create a minimal MolecularFeatures for testing pool.add.

    ``extra`` sets further attributes by name, as
    tests/unit/rules/test_coverage_gating_expansion.py does.
    """
    mol = Chem.MolFromSmiles(smiles)
    f = MolecularFeatures(mol=mol, smiles=smiles)
    if functional_groups is not None:
        f.functional_groups = functional_groups
    if principal_group is not None:
        f.principal_group = principal_group
    for key, val in extra.items():
        setattr(f, key, val)
    return f


# ---------------------------------------------------------------------------
# HandlerPolicy + HANDLER_POLICIES tests
# ---------------------------------------------------------------------------

class TestHandlerPolicy:
    def test_handler_policy_creation(self):
        """Test 1: HandlerPolicy dataclass instantiation with defaults.

        a phase: cascade_ratio_min, min_ratio_accept, gate_threshold
        fields REMOVED from the dataclass. Only handler_id / tier /
        priority / direct_return remain.
        """
        p = HandlerPolicy(handler_id='x', tier='ring_a', priority=4)
        assert p.handler_id == 'x'
        assert p.tier == 'ring_a'
        assert p.priority == 4
        assert p.direct_return is False
        # a phase: the three gate fields must no longer exist on
        # the dataclass (removed from HandlerPolicy definition).
        from dataclasses import fields
        field_names = {f.name for f in fields(HandlerPolicy)}
        assert 'cascade_ratio_min' not in field_names, "SC-8: cascade_ratio_min must be removed"
        assert 'min_ratio_accept' not in field_names, "SC-8: min_ratio_accept must be removed"
        assert 'gate_threshold' not in field_names, "SC-8: gate_threshold must be removed"

    def test_handler_policies_dict_populated(self):
        """Test 2: HANDLER_POLICIES contains all expected handler IDs."""
        expected_handlers = {
            'complex_ring', 'heterocycle', 'benzene', 'chain',
            'oxime', 'hydrazone', 'isocyanate', 'isothiocyanate',
            'carbamic_acid', 'carbamate', 'urea', 'guanidine',
            'sulfoxide', 'sulfone', 'thioether', 'boronic_acid',
            'partial_sat', 'polycyclic', 'acid_halide', 'anhydride',
            'lactone', 'lactam', 'ring_ester', 'polyfunctional',
            'multi_ester', 'ester', 'phosphine_oxide', 'phosphate_ester',
            'phosphine', 'phosphinic_acid', 'ring_assembly',
            'ring_nitrile', 'amide', 'amine', 'n_oxide',
        }
        missing = expected_handlers - set(HANDLER_POLICIES.keys())
        assert not missing, f"Missing handlers: {missing}"

    def test_handler_policies_complex_ring_is_tier_a(self):
        """Test 3: complex_ring is Tier A with correct tier/priority/direct_return.

        HandlerPolicy carries no ratio or threshold fields: the 0.40 complex_ring
        acceptance is tested inline in handlers/tier_a_ring.py.
        """
        p = HANDLER_POLICIES['complex_ring']
        assert p.direct_return is False, "Tier A — pool decides"
        assert p.tier == 'ring_a'
        assert p.priority == HANDLER_PRIORITY['complex_ring']

    def test_handler_policies_chain_priority_raised_to_ring_a_equal(self):
        """Test 5: the chain policy priority is 4 (that of the ring_a handlers).

        HANDLER_PRIORITY['chain'] is 1: select_best_candidate ranks by
        HANDLER_PRIORITY, not by the HANDLER_POLICIES-level priority.
        """
        p = HANDLER_POLICIES['chain']
        assert p.priority == 4, "chain policy priority is 4 (the ring_a handlers' value)"
        assert p.tier == 'chain'
        assert p.direct_return is False
        # HANDLER_PRIORITY['chain'] remains 1.
        assert HANDLER_PRIORITY['chain'] == 1, (
            "HANDLER_PRIORITY['chain'] must stay 1: the select_best_candidate "
            "tiebreak ranks by it"
        )

    @pytest.mark.parametrize("handler_id", [
        # 'chain' intentionally EXCLUDED per a phase: its policy-level
        # priority is promoted to 4 while HANDLER_PRIORITY['chain'] remains 1.
        'complex_ring', 'heterocycle', 'benzene', 'oxime',
        'polycyclic', 'n_oxide', 'amine', 'polyfunctional',
    ])
    def test_handler_policies_priority_pulled_from_handler_priority(self, handler_id):
        """Test 18 (REGRESSION): every priority pulled from HANDLER_PRIORITY
        EXCEPT chain (intentional divergence).

        Includes n_oxide, amine, polyfunctional which Plan 02 Task 2 added to
        HANDLER_PRIORITY. Proves single-source-of-truth invariant — no
        hardcoded priority literals in HANDLER_POLICIES except the intentional
        chain=4 promotion documented in a phase.
        """
        assert HANDLER_POLICIES[handler_id].priority == HANDLER_PRIORITY[handler_id]

    def test_handler_policies_excludes_ion_classes(self):
        """Test 19: salt/ion/radical/simple_molecule excluded per RESEARCH §9.2 + ISS-001."""
        for excluded in ('salt', 'ion', 'radical', 'simple_molecule'):
            assert excluded not in HANDLER_POLICIES, \
                f"{excluded!r} should NOT be in HANDLER_POLICIES (different naming semantics)"


# ---------------------------------------------------------------------------
# CandidatePool init tests
# ---------------------------------------------------------------------------

class TestCandidatePoolInit:
    def test_pool_init_first_applicable(self):
        """Test 6: a new pool holds no candidate and no direct-return winner."""
        pool = CandidatePool()
        assert pool._candidates == []
        assert pool._direct_return_winner is None


# ---------------------------------------------------------------------------
# CandidatePool selection tests
# ---------------------------------------------------------------------------

class TestCandidatePoolSelection:
    def test_pool_first_applicable_returns_first_added(self):
        """Test 9: best returns the first-added candidate regardless of confidence."""
        pool = CandidatePool()
        # Add three candidates manually (bypassing add to control confidence)
        c1 = CandidateName(name="ethanol", handler='chain', confidence=0.4)
        c2 = CandidateName(name="ethyl alcohol", handler='chain', confidence=0.9)
        c3 = CandidateName(name="hydroxyethane", handler='chain', confidence=0.7)
        pool._candidates = [c1, c2, c3]
        best = pool.best()
        assert best is c1, "best() returns the first-added candidate (not the highest confidence)"

    def test_pool_best_empty_returns_none(self):
        """pool.best with no candidates returns None."""
        pool = CandidatePool()
        assert pool.best() is None

    def test_pool_direct_return_winner_short_circuits(self):
        """Test 17: the first-added direct_return candidate wins."""
        pool = CandidatePool()
        c1 = CandidateName(name="polycyclic_name", handler='polycyclic', confidence=0.6)
        c2 = CandidateName(name="complex_ring_name", handler='complex_ring', confidence=0.9)
        pool._candidates = [c1, c2]
        assert pool.best() is c1

    def test_best_returns_direct_return_winner_else_first(self):
        """best returns the candidate of the first direct_return handler when
        one was added, otherwise the first candidate added.

        Built through add on the real molecules, so the ratio floor does not
        void the names. Hexahelicene is named twice: by complex_ring
        (direct_return False) and then by polycyclic (direct_return True); the
        pool holds both and best is the polycyclic candidate. Ethanol is named
        by chain and then by complex_ring (neither direct_return): best is the
        first.
        """
        assert HANDLER_POLICIES['complex_ring'].direct_return is False
        assert HANDLER_POLICIES['polycyclic'].direct_return is True
        assert HANDLER_POLICIES['chain'].direct_return is False

        features = _make_features("c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12")
        pool = CandidatePool()
        first = pool.add("hexahelicene", "complex_ring", features)
        second = pool.add("hexahelicene", "polycyclic", features)
        assert first is not None and second is not None
        assert len(pool.all_candidates()) == 2
        assert pool.best() is second
        assert pool.best() is not first

        features = _make_features("CCO")
        pool = CandidatePool()
        first = pool.add("ethanol", "chain", features)
        second = pool.add("ethanol", "complex_ring", features)
        assert first is not None and second is not None
        assert len(pool.all_candidates()) == 2
        assert pool.best() is first


# ---------------------------------------------------------------------------
# CandidatePool.add — gate behavior + post-hoc parent_atom_indices
# ---------------------------------------------------------------------------

class TestCandidatePoolAdd:
    def test_pool_add_returns_candidate_for_non_gated_handler(self):
        """pool.add with non-Tier-B handler stores and returns candidate."""
        features = _make_features("CCO")
        pool = CandidatePool()
        cand = pool.add("ethanol", "chain", features)
        assert cand is not None
        assert cand.name == "ethanol"
        assert cand.handler == "chain"
        assert len(pool.all_candidates()) == 1

    def test_pool_parent_atom_indices_propagated_post_hoc(self):
        """Test 15: parent_atom_indices set POST-HOC on returned candidate."""
        features = _make_features("CCO")
        pool = CandidatePool()
        parent_atoms = {0, 1, 2}
        cand = pool.add("ethanol", "chain", features, parent_atom_indices=parent_atoms)
        assert cand is not None
        assert cand.parent_atom_indices == parent_atoms

    def test_pool_unknown_handler_id_no_gate(self):
        """Test 20: unknown handler_id accepted with no gate enforcement."""
        features = _make_features("CCO")
        pool = CandidatePool()
        cand = pool.add("ethanol", "totally_unknown_handler_xyz", features)
        # No policy → no gate → candidate stored regardless of confidence
        assert cand is not None
        assert len(pool.all_candidates()) == 1

    def test_ratio_floor_and_tier_b_gate_void_a_candidate(self):
        """add returns None, and stores nothing, when the ratio floor or the
        confidence gate of a ring_b handler rejects the candidate.

        A 30-carbon chain with a principal alcohol group and a very short name:
          * "ol" has ratio 0.044, below RATIO_REJECT_FLOOR, so any handler's
            candidate is voided, unless the producer measured a complete atom
            partition (coverage_measured=True), which supersedes the floor;
          * "hexane" has ratio 0.133, above the floor, so only the confidence
            gate that applies to ring_b handlers (thioether) voids it, and the
            same name from a chain handler is kept.
        """
        features = _make_features(
            "C" * 30,
            functional_groups={'alcohol': [(1,)]},
            principal_group='alcohol',
            principal_group_atoms=[(1,)],
            substituents={0: [[2, 3]]},
            ring_substituents={},
            heterocycle_substituents={},
            benzene_substituents={},
        )
        assert compute_confidence("ol", "chain", features).factors['ratio'] < cp_module.RATIO_REJECT_FLOOR
        hexane_chain = compute_confidence("hexane", "chain", features)
        assert hexane_chain.factors['ratio'] >= cp_module.RATIO_REJECT_FLOOR
        assert HANDLER_POLICIES['thioether'].tier == 'ring_b'
        assert HANDLER_POLICIES['chain'].tier == 'chain'
        assert compute_confidence("hexane", "thioether", features).confidence < CONFIDENCE_GATE_THRESHOLD

        pool = CandidatePool()
        assert pool.add("ol", "chain", features) is None
        assert pool.add("hexane", "thioether", features) is None
        assert pool.all_candidates() == []

        measured = pool.add("ol", "chain", features, coverage_measured=True)
        assert measured is not None
        kept = pool.add("hexane", "chain", features)
        assert kept is not None
        assert pool.all_candidates() == [measured, kept]


# ---------------------------------------------------------------------------
# Thread-local pool lifecycle tests
# ---------------------------------------------------------------------------

class TestPoolLifecycle:
    def test_clear_pool_resets(self):
        """Test 14: clear_pool resets thread-local state."""
        features = _make_features("CCO")
        clear_pool()
        pool = get_current_pool()
        pool.add("ethanol", "chain", features)
        assert len(pool.all_candidates()) == 1
        clear_pool()
        new_pool = get_current_pool()
        assert len(new_pool.all_candidates()) == 0

    def test_get_current_pool_lazy_init(self):
        """get_current_pool creates pool on first access in a fresh thread."""
        # Reset by deleting the attribute (simulates fresh thread)
        if hasattr(_pool_store, 'pool'):
            delattr(_pool_store, 'pool')
        pool = get_current_pool()
        assert isinstance(pool, CandidatePool)

    def test_pool_thread_local_isolated(self):
        """Test 13: pools in different threads do not share candidates."""
        features = _make_features("CCO")
        results = {}
        results_lock = threading.Lock()

        def worker(name, handler, key):
            clear_pool()
            pool = get_current_pool()
            pool.add(name, handler, features)
            with results_lock:
                results[key] = len(pool.all_candidates())

        t1 = threading.Thread(target=worker, args=("ethanol", "chain", "t1"))
        t2 = threading.Thread(target=worker, args=("methanol", "chain", "t2"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread sees its own pool with exactly 1 candidate
        assert results == {"t1": 1, "t2": 1}, \
            f"Thread isolation broken or test races; got {results!r}"


# ---------------------------------------------------------------------------
# module-level binding regression test
# ---------------------------------------------------------------------------

class TestModuleLevelImportBinding:
    def test_module_level_parent_correctness_binding_exists(self):
        """Test 21 (ISS-005 REGRESSION): ParentCorrectnessScorer is bound at
        MODULE LEVEL via try/except ImportError. Either the class or None
        — both indicate the binding lives at module scope (not per-call)."""
        assert hasattr(cp_module, 'ParentCorrectnessScorer'), \
            "Module must expose ParentCorrectnessScorer binding (class or None)"
        # Binding must be either the class or None (not undefined / missing).
        assert cp_module.ParentCorrectnessScorer is None \
            or callable(cp_module.ParentCorrectnessScorer), \
            f"Binding must be class or None; got {cp_module.ParentCorrectnessScorer!r}"
