"""The acyl prefix of a saturated chain substituent that is a carbonyl group and carries another
detachable group: '2-hydroxypropanoyl', not '2-hydroxy-1-oxopropyl'.

``substituent_naming._name_polyfunctional_acyclic_substituent_impl`` numbers a chain from its free
valence and cites every group on it as a detachable prefix, the oxo on the free-valence carbon
included ('1-oxo'). That spelling is the name of the group by substitution on a ``-yl`` prefix; the
Blue Book gives the acyl group the name of the acid and puts the other groups on it.

 Acyl groups derived from carboxylic acids with names retained only for general
nomenclature, (the Blue Book): "substitution on acyl groups is identical to that of
acids... Preferred IUPAC names are systematic substitutive names", with 'propanoyl (preferred prefix)',
'1-oxopropyl' the alternative that is not preferred (:30562). Substituted acyl prefixes the book prints as
preferred: '3-chloropropanoyl' (N-acetyl-N-(3-chloropropanoyl)benzamide (PIN),:33117),
'2-methylidenebutanoyl' ([2,3-dichloro-4-(2-methylidenebutanoyl)phenoxy]acetic acid (PIN),:31400),
'oxoacetyl' (:30516) and, for the two-carbon acyl, 'chloroacetyl' ((chloroacetyl)oxyl (PIN),:40694); the
locant of the one position a substituent can take on an acetyl group is omitted, (:3031) "All
locants are omitted for parent compounds when all substitutable hydrogen atoms have the same locant".
"""
from typing import Dict, List, Optional


def is_acyl_prefix(name: str) -> bool:
    """True when ``name`` is an acyl name ``acyl_prefix_text`` spells ('2-oxopropanoyl', 'chloroacetyl'),
    not the '1-oxopropyl' form that cites the oxo of the free-valence carbon as a prefix."""
    return name.endswith(("anoyl", "acetyl"))


def acyl_prefix_text(mol, attach_idx: int, sub_atoms, prefix_on: Dict[int, List[str]],
                     groups: Dict[str, List[int]], backbone) -> Optional[str]:
    """The acyl name of a linear saturated chain (``backbone``, numbered from ``attach_idx``) whose
    free-valence carbon carries an oxo and nothing else, with the other groups of ``groups`` (prefix ->
    locants counted from the free valence) on it; None when the chain is not such a group. A two-carbon
    acyl takes one kind of prefix (acetyl has one position and the locant is omitted); two different
    prefixes on the one carbon need the enclosing marks of, and the caller keeps its
    spelling. An acyl group on a nitrogen is the acylamino prefix of an amide: Substituents of the
    types -NH-CO-R and -NH-SO2-R (the Blue Book) names it by the 'amido' prefix (method (1),
    '(4-acetamido-3-methylphenyl)arsonic acid (PIN)',:33010) or by 'acylamino' (method (2)); that
    choice is not made here and the caller keeps its spelling."""
    n_carbons = len(backbone)
    if n_carbons < 2 or prefix_on.get(attach_idx) != ["oxo"]:
        return None
    if any(nb.GetAtomicNum() == 7 for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
           if nb.GetIdx() not in sub_atoms):
        return None
    rest = {p: [loc for loc in locs if not (p == "oxo" and loc == 1)] for p, locs in groups.items()}
    rest = {p: locs for p, locs in rest.items() if locs}
    if not rest or (n_carbons == 2 and len(rest) != 1):
        return None
    from ..data.chain_names import get_chain_prefix
    from .naming_utils import alpha_sort_key
    from .substituent_naming import _joined_prefix_parts, _located_prefix_part
    parts = [_located_prefix_part(p, rest[p], cite_locants=n_carbons > 2)
             for p in sorted(rest, key=alpha_sort_key)]
    return "".join(_joined_prefix_parts(parts)) + (
        "acetyl" if n_carbons == 2 else f"{get_chain_prefix(n_carbons)}anoyl")
