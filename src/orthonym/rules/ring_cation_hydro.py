"""One numbering for a ring cation whose neutral ring is a hydro derivative of a mancude
heteromonocycle ('2-methyl-3,4,5,6-tetrahydropyrimidin-1-ium').

``rules.ions._emit_ring_cumulative_suffix`` names the neutral ring (its own numbering: hydro
locants lowest, '2-methyl-1,4,5,6-tetrahydropyrimidine') and cites the '-ium' from the numbering
of the charged ring ('-1-ium', or '-3-ium' once the read-back has moved it): two numberings in
one name. For a ring whose parent has no indicated hydrogen they disagree, and the name is
either another species or withdrawn by the check of the default tier ("suffix/free
valence [3] > [1]").

The numbering is one authority here, the one that names the neutral hydro ring
(``heterocycles._mancude_hydro_select``) with the ring atom of the '-ium' as the suffix position.
 "General rule for systematically naming cationic centers in parent hydrides"
(the Blue Book),:41368: a cation "derived formally by adding one or more hydrons to any
position of a neutral parent hydride... or whose degree of hydrogenation has been modified (see
 is named by replacing the final letter 'e' of the parent hydride name, if any, by the suffix
'ium'". The '-ium' is a suffix of the name, so "NUMBERING" (:3219) ranks its locant by
(c) "principal characteristic groups and free valences (suffixes)" (:3256), after (b) "indicated
hydrogen for unsubstituted compounds" (:3246) and before (e) "(i) low locants are given to
hydro/dehydro prefixes" (:3288-:3289) and (f) "detachable alphabetized prefixes, all considered
together" (:3301). A hydro prefix is detachable but not alphabetized and stands directly before
the parent hydride (e),:3288): '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium (PIN)'
,:41447).
"""
import re


def hydro_ring_cation_name(mol, ring_mol, ring_system, center_idx: int) -> str:
    """The name of the ring cation of ``mol`` centred on ``center_idx``, in one numbering, or ''
    when the ring is not a partially saturated mancude heteromonocycle of this kind.

    ``ring_mol`` is ``mol`` with the centre neutralized, index-preserving (the neutral ring the
    '-ium' is added to). The numbering is that of the hydro name of ``ring_mol`` with the centre as
    the suffix atom: heteroatoms, indicated hydrogen (b), the '-ium' (c), the hydro prefixes (e),
    the substituent prefixes (f). Declines ('') for a ring that is aromatic, saturated, fused, of a
    size the hydro namer does not number, a centre that is a carbon or is itself the indicated-
    hydrogen position (the parent's own ``1H`` is then consumed by the '-ium', which the caller's
    rule treats), a second charge, a guanidine carbon (three nitrogen neighbours) or a substituent
    that cannot be named; the caller's producer then stands, as it did before this module."""
    try:
        return _hydro_ring_cation_name(mol, ring_mol, ring_system, center_idx)
    except (RuntimeError, ValueError, KeyError, IndexError, TypeError, RecursionError):
        return ''


def _hydro_ring_cation_name(mol, ring_mol, ring_system, center_idx: int) -> str:
    from .heterocycles import _mancude_hydro_select, name_partially_saturated_monocyclic_heterocycle
    from .ions import _elide_terminal_e, _join_prefix_stem, _p74_ring_substituent_prefix
    from .stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
    ring_set = set(ring_system)
    rings = [set(r) for r in ring_mol.GetRingInfo().AtomRings() if ring_set & set(r)]
    if len(rings) != 1 or rings[0] != ring_set:
        return ''                       # fused, bridged or spiro: the fusion numbering is fixed
    if ring_mol.GetAtomWithIdx(center_idx).GetAtomicNum() == 6:
        return ''
    if any(a.GetFormalCharge() for a in ring_mol.GetAtoms()):
        return ''
    # A ring carbon with three nitrogen neighbours is a guanidine group. Whether its exocyclic
    # nitrogen is an 'amino' prefix of the cation or the guanidine is a parent of its own is not
    # settled by a Blue Book example "In the presence of a characteristic group
    # having seniority over guanidine...",:34266, speaks of the prefix 'carbamimidoylamino'), and
    # the caller's producers keep the cyclic guanidinium as they did before this module.
    if any(ring_mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and sum(
            1 for n in ring_mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 7) == 3
            for i in ring_set):
        return ''
    sel = _mancude_hydro_select(ring_mol, ring_set, {center_idx})
    if sel is None:
        return ''
    ordered, loc, ih = sel['ordered'], sel['loc'], sel['ih']
    if center_idx not in ordered or any(ordered[p] == center_idx for p in ih):
        return ''
    a2l = {ordered[p]: loc[p] for p in range(sel['n'])}
    stem = name_partially_saturated_monocyclic_heterocycle(ring_mol, ring_system, {center_idx})
    if not stem or 'unknown' in stem.lower() or not _stem_is_in_numbering(stem, sel):
        return ''
    prefix = _p74_ring_substituent_prefix(ring_mol, ring_system, a2l, set())
    if prefix is None:
        return ''
    # "NAMING OF STEREOISOMERS" (the Blue Book),:44643: "They are placed at the front
    # of the complete name when related to the parent structure": the ring's descriptors, in the
    # locants of the numbering this name carries.
    stereo = format_stereodescriptor_string(collect_stereodescriptors(mol, a2l))
    return f"{stereo}{_join_prefix_stem(prefix, _elide_terminal_e(stem))}-{a2l[center_idx]}-ium"


def _stem_is_in_numbering(stem: str, sel) -> bool:
    """True when the hydro prefixes and the indicated hydrogen that ``stem`` cites are those of the
    selected numbering ``sel`` (``heterocycles._mancude_hydro_select``): the stem and the locant
    of the '-ium' are two readings of one numbering or the name is not built. The stem is written
    by the namer of the neutral hydro ring (its retained parent 'pyrimidine' or 'imidazole',
    , the numbering by the selection that namer's hydro locants follow."""
    loc, ih = sel['loc'], sel['ih']
    hydro = ','.join(str(loc[p]) for p in sorted(sel['sp3_pos'] - ih, key=lambda q: loc[q]))
    lead = f"{hydro}-{sel['prefix']}"
    if not stem.startswith(lead):
        return False
    rest = stem[len(lead):]
    if ih:
        cited = ','.join(f"{loc[p]}H" for p in sorted(ih, key=lambda q: loc[q]))
        return rest.startswith(f"-{cited}-")
    return not re.match(r'-?\d+H', rest)
