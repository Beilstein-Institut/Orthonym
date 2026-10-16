"""Numbering of a more-than-ten-membered heteromonocycle for replacement ("a") nomenclature.

Moved out of ``rules/heterocycles.py`` (a FILES entry of the size baseline) with its one helper; the
numbering rule is unchanged except for the suffix set, below.

 NUMBERING (the Blue Book): when there is a choice, low locants go to, in this order, the
heteroatoms (a-b), "(c) principal characteristic groups and free valences (suffixes)" (:3256), then the
hydro/ene endings. Lactones (the Blue Book): "(1) as heterocyclic pseudoketones by adding
the suffix 'one', 'dione'... Method (1) gives preferred IUPAC names", so the C=O of a ring lactone, lactam
and ketone are all '-one' suffix carbons of ONE set.

The defect this fixes: the suffix set handed in held the atoms of the perceived principal group only (the
lactone: its carbonyl carbon, the ring oxygen and the alkyl carbon, which are symmetric between the two
directions), so a second ring C=O (a ketone beside the lactone) never entered the comparison, the ties fell
to the ene locant, and the ring was numbered the other way round: '1-oxacyclotetradec-4-ene-2,4-dione' for
a 14-membered lactone whose double bond is C11=C12 (the 4-ene collides with the 4-one; the oxo/ene guard
held the name back). The suffix set is now exactly the ring carbonyl carbons whenever the principal group sits
on the ring (``rules.partial_saturation._ring_carbonyl_carbons``: ketone, lactam and lactone carbonyls, "all
named with the -one suffix"); the lactone's ring oxygen and alkyl carbon are not suffix atoms.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from rdkit import Chem

from ..data.hw_heteroatoms import HETEROATOM_PRIORITY


def _macrocycle_db_locant(p: int, q: int, ring_size: int) -> int:
    """Locant cited for a ring double bond between adjacent positions p, q:
    the lower locant, except the ring-closure (1, N) bond which is cited as N."""
    if {p, q} == {1, ring_size}:
        return ring_size
    return min(p, q)


def orient_macrocycle_for_replacement(
    mol, ring_atoms, principal_group_atoms=None
) -> Tuple[Optional[List[Tuple[int, str]]], Optional[List[int]]]:
    """Number a >10-membered heteromonocycle for replacement ("a") nomenclature.

    Returns ``(heteroatom_locants, double_bond_locants)`` where the numbering
    minimises, in order (IUPAC 2013 /: the heteroatom locant
    set, then the heteroatoms by element seniority, then the PRINCIPAL
    CHARACTERISTIC GROUP locant set (c) — the suffix, e.g. the ``-2,6-dione``
    carbons), then the ring double-bond locant set.

     M4: threading ``principal_group_atoms`` (the in-ring suffix-bearing atoms)
    into the key is the fix for the macrolactam/macrolactone class whose in-ring
    ``ene`` was numbered on a DIFFERENT direction than the composer's suffix
    locants — the two disagreed and the spliced name was internally inconsistent
    (OPSIN-unparseable → abstain). With the suffix ranked ABOVE unsaturation the
    numberer picks the same direction the composer uses, so the name round-trips.
    Empty ``principal_group_atoms`` (a bare ring / crown ether) leaves the key
    unchanged, so the previously-passing cases are byte-identical (no regression),
    and any residual mismatch still fails the OPSIN validity gate → abstain.

    The ring is *kekulised* first so a fully-conjugated macrocycle that RDKit
    perceives as aromatic (e.g. the V-6 azacyclotrideca-hexaene) yields its
    localised double bonds instead of zero (the bug that made the old path emit
    a single bare ``ene``). Returns ``(None, None)`` if the ring cannot be
    ordered/kekulised, so the caller can fall back safely.
    """
    from .heterocycles import _macrocycle_ordered_ring
    from .partial_saturation import _ring_carbonyl_carbons

    pg_atoms = set(principal_group_atoms or ())
    ring_set = set(ring_atoms)
    ring_size = len(ring_set)
    carbonyls = set(_ring_carbonyl_carbons(mol, ring_set))
    if (pg_atoms & ring_set) and carbonyls:
        # the principal group is an in-ring '-one' (ketone, lactone, lactam): the suffix set is every ring
        # C=O and nothing else (the lactone's ring oxygen and alkyl carbon are not suffix atoms; counted, they
        # outweigh a ketone that sits farther round the ring)
        pg_atoms = carbonyls
    het = [a for a in ring_set if mol.GetAtomWithIdx(a).GetSymbol() != 'C']
    if not het:
        return None, None

    kmol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kmol, clearAromaticFlags=True)
    except Exception:
        return None, None

    order = _macrocycle_ordered_ring(kmol, ring_set)
    if order is None or len(order) != ring_size:
        return None, None

    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in kmol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            ring_double_bonds.append((i, j))

    best = None
    for start in range(ring_size):
        for direction in (1, -1):
            seq = [order[(start + k * direction) % ring_size] for k in range(ring_size)]
            loc = {a: idx + 1 for idx, a in enumerate(seq)}
            het_set = sorted(loc[a] for a in het)
            seniority = sorted(
                (HETEROATOM_PRIORITY.get(kmol.GetAtomWithIdx(a).GetSymbol(), 999), loc[a])
                for a in het
            )
            dbl = sorted(
                _macrocycle_db_locant(loc[i], loc[j], ring_size)
                for i, j in ring_double_bonds
            )
            # (c): the principal characteristic group takes low locants
            # BEFORE unsaturation. Only in-ring suffix atoms carry a ring locant
            # (an exocyclic -carboxylic-acid carbon is filtered by ``a in loc``);
            # empty -> ```` -> no effect on the key (no regression).
            pg_key = tuple(sorted(loc[a] for a in pg_atoms if a in loc))
            key = (het_set, seniority, pg_key, dbl)
            if best is None or key < best[0]:
                best = (key, loc, dbl)

    loc = best[1]
    dbl = best[2]
    heteroatom_locants = sorted(
        ((loc[a], kmol.GetAtomWithIdx(a).GetSymbol()) for a in het),
        key=lambda t: t[0],
    )
    return heteroatom_locants, dbl


__all__ = ["orient_macrocycle_for_replacement"]
