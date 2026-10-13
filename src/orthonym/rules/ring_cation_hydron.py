"""Where the added hydron of a ring cation can sit: the resonance structures that
``rules.ions._ring_cation_hydron_site`` reads, from RDKit below a work count and
from the matching of the pi bonds above it (``perception.charge_shift``).

Moved out of ``rules/ions.py`` so that file stays within its size budget; the
names are imported back into ``rules.ions``.
"""
from typing import Any, Dict

from rdkit import Chem

from ..perception import charge_shift


def _hydron_site_candidates(mol, ring_system, center_idx) -> Dict[int, Any]:
    """{ring atom: a resonance structure of ``mol`` with the drawn centre neutral and +1 on that atom}.

    RDKit builds every bond-order arrangement of each group that carries charge or
    N=X bonds and its C++ call cannot be interrupted, so it is asked only when
    ``charge_shift.rdkit_flags0_affordable(mol)``; above that count the structures
    come from the matching of the pi bonds (``charge_shift.hydron_sites``). An
    undecided matching, and an error in either route, give {}: the molecule has no
    site here and RDKit is never asked in the matching's place."""
    if charge_shift.rdkit_flags0_affordable(mol):
        structs = _rdkit_flags0_structures(mol)
        return {} if structs is None else _hydron_sites_in(structs, ring_system, center_idx)
    try:
        return charge_shift.hydron_sites(mol, center_idx, ring_system) or {}
    except Exception:
        return {}


def _rdkit_flags0_structures(mol):
    """The resonance structures RDKit builds for ``mol`` without a flag (at most 64), or None when it fails."""
    try:
        sup = Chem.ResonanceMolSupplier(mol, 0, 64)
        return [sup[i] for i in range(len(sup))]
    except Exception:
        return None


def _hydron_sites_in(structs, ring_system, center_idx) -> Dict[int, Any]:
    """{ring atom: the first structure of ``structs`` in which the centre is neutral and that atom, neither H nor C,
    bears +1 and a hydrogen}."""
    found = {}
    for r in structs:
        if r is None or r.GetAtomWithIdx(center_idx).GetFormalCharge() != 0:
            continue
        for idx in ring_system:
            a = r.GetAtomWithIdx(idx)
            if (idx != center_idx and a.GetAtomicNum() not in (1, 6)
                    and a.GetFormalCharge() == 1 and a.GetTotalNumHs() >= 1):
                found.setdefault(idx, r)
    return found


def _other_ring_heteroatom_with_hydrogen(mol, ring_system, center_idx) -> bool:
    """True when a ring atom other than the centre, neither H nor C, carries a
    hydrogen in the drawing.

    A resonance structure at flags 0 moves bond orders and charges and keeps the
    hydrogen count of every atom (RDKit ``Resonance.cpp``, ``fixExplicitImplicitHs``:
    only an incomplete octet, which flags 0 excludes, derives other implicit
    hydrogens). The hydron of "General rule for systematically naming
    cationic centers in parent hydrides" (the Blue Book),:41368: "A cation
    derived formally by adding one or more hydrons to any position of a neutral
    parent hydride..." can therefore sit only on a ring heteroatom that has a
    hydrogen in the drawing; without one no structure has a hydron-carrying
    heteroatom and the enumeration cannot find a site."""
    for idx in ring_system:
        atom = mol.GetAtomWithIdx(idx)
        if idx != center_idx and atom.GetAtomicNum() not in (1, 6) and atom.GetTotalNumHs() >= 1:
            return True
    return False
