"""a phase guanidine handler — Tier B retained-name (gate 0.40).

Verbatim lift of composer.py:892-901 (inline branch) +
composer.py:2760-2906 (_try_name_guanidine body). Per internal notes,
body stays in composer.py until Plan-03 commit 03-10.

IUPAC cite: (guanidine and its derivatives; retained name with
N-substitution).

References:
- composer.py:892-901 (inline dispatch branch; REMOVED at this commit).
- composer.py:2760-2906 (_try_name_guanidine body).
- internal notes-DECOMP.md row 'guanidine' + purity proof.
"""
from __future__ import annotations

from typing import Any, Optional

from rdkit import Chem

from ..name_tree import NameTreeNode, NamingResult


def _guanidine_principal_rank(features: Any) -> Optional[str]:
    """'senior' when an acyclic guanidine is the parent class against the principal group the
    molecule's other groups give it, its unit is a plain guanidine and the builder can spell the
    arm of that group; 'senior_no_arm' when the guanidine is the parent but the builder cannot
    (a hydrazide principal group, see below); else None.

     "Guanidine and its derivatives" (the Blue Book): "In the presence of a
    characteristic group having seniority over guanidine (see item 11 in, the following
    prefixes are used", so guanidine itself ranks at class 11 of (:18162,
    ':18184' "11 Amides [in the order of the corresponding acids...]"): the amidine of carbonic
    acid takes the place of urea, the amide of carbonic acid. The groups it outranks are those
    ``urea_is_senior_to`` lists, the hydrazides and imides (12, 13), nitriles (14) down to amines
    (19) and imines (20); the esters, acid halides, the amides and amidines of the carboxylic and
    other 7a acids stay senior ('N-carbamimidoylformamide (PIN)',:34290), and 'N-carbamimidoylurea
    (PIN)' (:34292) puts urea above it.

    An acyclic imide is not the class 13 group of the table: IMIDES, (:33580)
    "Acyclic imides are N-acyl derivatives of primary amides and are named as such (see.
    Cyclic imides are preferably named as heterocyclic pseudoketones", so a -CO-NH-CO- outside a ring
    (an N-acylurea too) is an amide, kept senior to guanidine (``urea.acyclic_imide_present``).

    A hydrazide principal group is junior to guanidine as well (class 12), but the substituent
    namer spells its -CO-NH-NH2 arm '(aminoamino)...oxo', not the 'hydrazinyl'... 'oxo' of
     HYDRAZIDES,:34014 "When the -CO-NHNH2 group is at the end of a chain the use of the
    prefixes 'hydrazinyl' and 'oxo' generates preferred IUPAC names";:34018 "3-hydrazinyl-3-oxo
    propanoic acid (PIN)"), and the acylhydrazinyl and acetohydrazido arms of:34020 are not built
    at all. Such a molecule is 'senior_no_arm': the guanidine is its parent, the builder declines
    and the name any other handler gives is labelled below the PIN, as for urea (``urea.
    HYDRAZIDE_PRINCIPALS``).

    The principal group is read the way the ring systems see it: an amidine or guanidine whose
    carbon is a ring atom is the ring itself and its one exocyclic -NH2 an amine "Primary
    amines, R-NH2, are systematically named in the following ways",:26168; 'quinolin-4-amine (PIN)',
    :26199; ``seniority.ring_skeleton_units_resolved``), so the 'amidine' the perception
    reports for the 2-amino-3,4-dihydroquinoline ring is the amine class 19, junior to guanidine.

    A plain guanidine: no nitrogen of the unit bears another nitrogen (an aminoguanidine, a
    guanylhydrazone), a sulfur or phosphorus (the sulfonyl and phosphoryl amides are classes of
    their own), a carbon doubly bonded to a heteroatom (an acylguanidine is named on the acyl
    group's class) or a hydroxy group (N-hydroxyguanidine is named by the hydroxylamine handler,
    which already writes the guanidine parent): the builder leaves those to the producers that
    name them today."""
    from ...rules.seniority import (
        get_principal_group,
        guanidine_parent_matches,
        ring_skeleton_units_resolved,
    )
    from .urea import HYDRAZIDE_PRINCIPALS, acyclic_imide_present, urea_is_senior_to
    fg = getattr(features, 'functional_groups', None) or {}
    mol = getattr(features, 'mol', None)
    if mol is None or getattr(features, 'principal_group', None) is None:
        return None
    matches = guanidine_parent_matches(mol, fg.get('guanidine') or ())
    if not matches:
        return None
    for match in matches:
        c_idx = match[1]
        for n_idx in (match[0], match[2], match[3]):
            for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
                if nb.GetIdx() == c_idx or nb.GetAtomicNum() == 1:
                    continue
                sym = nb.GetSymbol()
                if sym in ('N', 'S', 'P') or (sym == 'O' and nb.GetTotalNumHs()):
                    return None
                if sym == 'C' and any(
                        b.GetBondType() == Chem.BondType.DOUBLE
                        and b.GetOtherAtom(nb).GetSymbol() in ('O', 'S', 'Se', 'Te', 'N')
                        for b in nb.GetBonds()):
                    return None
    ring_atoms = {a for ring in mol.GetRingInfo().AtomRings() for a in ring}
    principal, p_matches = get_principal_group(
        mol, ring_skeleton_units_resolved(mol, fg, ring_atoms))
    if principal is None:
        return 'senior'
    if not urea_is_senior_to(principal):
        return None
    if principal == 'imide' and acyclic_imide_present(mol, p_matches):
        return None
    return 'senior_no_arm' if principal in HYDRAZIDE_PRINCIPALS else 'senior'


def guanidine_parent_expected(features: Any) -> bool:
    """True when the PIN of this molecule is a guanidine name while another group was perceived
    as principal (``_guanidine_principal_rank``); used to label any other handler's name below
    the PIN, as ``urea_parent_expected`` does for urea."""
    try:
        return _guanidine_principal_rank(features) is not None
    except Exception:  # noqa: BLE001 -- no answer: no label change
        return False


def _is_guanidine(features: Any) -> bool:
    """Guanidine FG present AND no principal group senior to guanidine.

    Mirrors composer.py:892 (principal_group None), widened as ``handlers.urea._is_urea`` is:
    a molecule whose other groups all rank below guanidine (``_guanidine_principal_rank``) has a
    guanidine name as its PIN, 'N-(2-hydroxyethyl)guanidine' not '2-[(diaminomethylidene)amino]
    ethan-1-ol', the Blue Book)."""
    fg = getattr(features, 'functional_groups', None) or {}
    if not fg.get('guanidine'):
        return False
    if getattr(features, 'principal_group', None) is None:
        return True
    try:
        return _guanidine_principal_rank(features) == 'senior'
    except Exception:  # noqa: BLE001 -- no answer: the producers that name it today keep it
        return False


def name_guanidine(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """a phase Tier-B guanidine handler."""
    from ..candidate_pool import get_current_pool
    from ..composer import (
        _enrich_handler_name,
        _inject_stereo_if_missing,
        _try_name_guanidine,
    )

    guanidine_name = _try_name_guanidine(features)
    if not guanidine_name:
        return None

    # ``_try_name_guanidine`` builds a COMPLETE name: the parent plus every N-substituent it
    # walked itself (it refuses when one is unnameable). When the molecule's principal group is
    # another one (``_is_guanidine``, the guanidine outranking it) the features carry the ring or
    # chain of that group as the parent, and enrichment would spell the arm a second time
    # ('ethylbenzenylN-{2-[4-(hydroxymethyl)phenyl]ethyl}guanidine'); as in ``name_urea`` the
    # complete name is not enriched.
    if getattr(features, 'principal_group', None) is None:
        guanidine_name = _enrich_handler_name(features, guanidine_name, "guanidine")

    pool = get_current_pool()
    cand = pool.add(guanidine_name, "guanidine", features)
    if cand is None:
        return None

    # Guanidine, like urea, has no numbered skeleton: its only locants are the
    # italic letters N, N', N'', the Blue Book "Guanidine and its
    # derivatives": "the locants N, N' and N'' are used in preferred IUPAC names.
    # The locants 1, 2, and 3 have been used but are no longer recommended").
    # A front-of-name block with NUMERIC locants therefore resolves to nothing in
    # the parent "Citation of locants", the Blue Book); the stereo of the
    # N-substituents is cited inside their own prefixes "NAMING OF
    # STEREOISOMERS", the Blue Book), which the substituent namer does. Without the
    # declared scope the ring and chain readings both agreed and the block was
    # prepended: '(1E)-N,N'-di[(1E)-prop-1-en-1-yl]guanidine', which OPSIN rejects.
    final_name = _inject_stereo_if_missing(features, cand.name,
                                           atom_to_locant=None,
                                           parent_scope='retained_no_locants')
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(parent_stem=final_name, class_id="guanidine", iupac_section_cite="P-66.4.1.2.1", fragment_legacy=final_name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_guanidine", "_is_guanidine"]
