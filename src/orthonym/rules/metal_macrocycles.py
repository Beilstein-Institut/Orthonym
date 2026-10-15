"""The macrocyclic ligand of a complex of a metal that the scope table admits to ``MACROCYCLE``.

A metal complex drawn as ions (the porphyrin dianion beside the metal cation) is named as a salt or an
adduct of its ions, and the ligand is an ion of its own. The general engine names the dianion of a
porphyrin as a von Baeyer cage; the porphyrin parent names it as the anion of that parent hydride:
``5,10,15,20-tetraphenylporphyrin-21,23-diide`` Anions derived from parent hydrides and
their derivatives, the Blue Book-40902; ``rules/porphyrin.name_porphyrin_dianion``).

The producer is metal-blind. This module is the scope: the name is cited only in an assembly whose
every true metal is admitted to ``MACROCYCLE`` in ``data/metal_scope.py`` (zinc), so the complexes of
the other metals keep the names they have. The Blue Book gives the name no PIN INTRODUCTION:
"Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter",:50943), and the callers
read it at the tiers that name ions only, so the default tier is unchanged.
"""
from typing import Dict, List, Optional

from rdkit import Chem

from ..data.metal_scope import MACROCYCLE, has_mechanism


def assembly_admits_macrocycle(mol) -> bool:
    """True when ``mol`` holds a true metal and every true metal of it is admitted to ``MACROCYCLE``.

    The guard reads the whole assembly: one metal outside the table keeps the rest of the assembly
    outside it, as for ``metal_atoms.ammine_component_names``."""
    from ..perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    metals = {a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO}
    return bool(metals) and all(has_mechanism(symbol, MACROCYCLE) for symbol in metals)


def macrocycle_ion_fragments(mol) -> List:
    """The fragments of ``mol`` that are porphyrin ions of an assembly admitted to ``MACROCYCLE``: a
    fragment with the 24-atom core of the porphyrin parent (``rules.porphyrin``) and a net charge below
    zero. Empty for an assembly outside the table.

    The test reads the structure alone. ``macrocycle_anion_name`` decides which of these ions have a
    porphyrin name, and its answer depends on the naming call it runs in (the name of a fused-ring
    substituent of the ion is built at the best-effort tier only), so the label of a name cannot read
    it."""
    if not assembly_admits_macrocycle(mol):
        return []
    from .porphyrin import _core_matches
    return [fragment for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if Chem.GetFormalCharge(fragment) < 0 and _core_matches(fragment)]


def macrocycle_anion_name(ion_mol, assembly_mol) -> Optional[str]:
    """The name of the macrocyclic anion ``ion_mol`` of the assembly ``assembly_mol``, or None: the
    porphyrin parent's name of the dianion (``rules.porphyrin.name_porphyrin_dianion``) when the
    assembly is admitted to ``MACROCYCLE``."""
    if ion_mol is None or not assembly_admits_macrocycle(assembly_mol):
        return None
    from .porphyrin import name_porphyrin_dianion
    return name_porphyrin_dianion(ion_mol)


def macrocycle_component_names(mol) -> Dict[str, str]:
    """``{canonical SMILES of a fragment: its name}`` for every macrocyclic anion of ``mol`` that
    ``macrocycle_anion_name`` names; ``{}`` for an assembly outside ``MACROCYCLE``."""
    names: Dict[str, str] = {}
    for fragment in macrocycle_ion_fragments(mol):
        name = macrocycle_anion_name(fragment, mol)
        if name:
            names[Chem.MolToSmiles(fragment, canonical=True)] = name
    return names
