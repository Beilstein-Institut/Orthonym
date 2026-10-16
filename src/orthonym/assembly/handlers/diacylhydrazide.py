"""1,2-Diacylhydrazines R-CO-NH-NH-CO-R', the Blue Book).

A thin dispatch shim over ``rules.diacylhydrazides``, which builds the COMPLETE name and returns None for
anything outside its documented scope. Registered at priority 1480, ahead of ``chain_diamide`` (1490),
``noncarbon_ester`` (1495) and ``ester_family`` (1500), which wrote '1-amino-2-(methylsulfanyl)ethanehydrazide'
and its like for the class, and ahead of the general_acyclic / tier_a_ring producers that dropped an acyl
('benzohydrazide' for N'-benzoylbenzohydrazide). The predicate asks for the hydrazide principal group, so a
molecule with an acid, an ester or an anhydride (senior to a hydrazide, class 12 of is never
claimed. Not enriched: the rule module already spells the parent and every substituent.
"""
from __future__ import annotations

from typing import Any, Optional

from ..name_tree import NameTreeNode, NamingResult


def _is_diacylhydrazide(features: Any) -> bool:
    """Predicate (pure): the hydrazide is the principal group and the N-N unit carries two acyl groups."""
    if getattr(features, "principal_group", None) != "hydrazide":
        return False
    from ...rules.diacylhydrazides import find_pair
    return find_pair(getattr(features, "mol", None)) is not None


def name_diacylhydrazide_handler(features: Any, mol: Any = None, style: str = "pin") -> Optional[NamingResult]:
    from ...rules.diacylhydrazides import name_diacylhydrazide
    from ..candidate_pool import get_current_pool

    mol = mol if mol is not None else getattr(features, "mol", None)
    name = name_diacylhydrazide(mol, style=style)
    if not name:
        return None
    cand = get_current_pool().add(name, "hydrazide", features)
    if cand is None:
        return None
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(parent_stem=cand.name, class_id="hydrazide",
                          iupac_section_cite="P-66.3.3.3", fragment_legacy=cand.name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_diacylhydrazide_handler", "_is_diacylhydrazide"]
