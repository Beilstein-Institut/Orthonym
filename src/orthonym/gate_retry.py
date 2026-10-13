"""Candidates that follow the first one of a molecule, after the validity gate withdrew it.

``namer.Orthonym._retry_cascade_on_gate_rejection`` names the molecule again, either with the
dispatch classes that have not been tried or, for a name OPSIN reads as ambiguous, with every
locant cited. Each further candidate gets the checks the first one had (:func:`gate_candidate`).
The functions read the checks from the ``namer`` module when they run, so a test that replaces
one of them replaces it here too.
"""
from typing import Optional

from .validation.opsin_ambiguity import FORCED_LOCANT_REASON


def gate_candidate(namer_obj, smiles: str, cand: str) -> Optional[str]:
    """The stereo and grammar backstops and the validity gate for one more candidate of the
    molecule, as the first one had them; None when RDKit cannot read ``smiles``.

     (a review F1): the result is the gate's OUTPUT, not the pre-gate ``cand``. The gate can
    return a name DIFFERENT from its input (the stereo-omission reclaim composes the input's
    dropped stereo back on), so the pre-gate name would ship the flat, stereo-INCOMPLETE name at
    the RT-verified tier, which the BE-STRICT contract forbids."""
    from rdkit import Chem

    from . import namer
    from .assembly.coverage_scoring import retrieve_confidence
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    conf = retrieve_confidence()
    cand = namer._final_stereo_check(
        mol, cand, handler=conf.get('handler', 'unknown'),
        atom_to_locant=conf.get('atom_to_locant'),
        is_phenol_benzene=conf.get('is_phenol_benzene'),
    )
    cand = namer._final_grammar_check(
        cand, smiles, conf.get('handler', 'unknown'),
        namer_obj._grammar, namer_obj._grammar_stats,
    )
    return namer._final_opsin_validity_gate(
        cand, smiles, namer_obj._grammar_stats,
        besteffort_unverified=namer_obj._general_fallback_unverified,
        general_fallback_tier=namer_obj._general_fallback)


def name_with_forced_locants(namer_obj, smiles: str) -> Optional[str]:
    """The molecule named once more with every omission licence off
    (``locant_omission.forced_locant_scope``), in a fresh fragment cache (the cache of the
    first run holds names made with the licences on), and gated as any candidate; None when no
    name came. One run, never a loop."""
    from .assembly.fragment_naming import isolated_naming_session
    from .assembly.locant_omission import forced_locant_scope
    from .errors import _DESCRIPTIVE_FALLBACK_NAMES
    try:
        with forced_locant_scope(FORCED_LOCANT_REASON), isolated_naming_session(reset_cache=True):
            cand = namer_obj._name_impl(smiles)
            if not cand or cand in _DESCRIPTIVE_FALLBACK_NAMES:
                return None
            return gate_candidate(namer_obj, smiles, cand)
    except Exception:                                           # noqa: BLE001
        return None
