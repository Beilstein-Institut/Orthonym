"""1,2-Diacylhydrazines R-CO-NH-NH-CO-R': the senior hydrazide substituted by the other acyl group.

 "Substituted hydrazides" (the Blue Book): "Acyl, diacyl, and triacyl derivatives of
hydrazides are named by substituting the senior hydrazide by the appropriate acyl groups (the senior
hydrazide is the one derived from the senior acid). Names based on substitution of the parent hydride
'hydrazine' are no longer recommended even in general nomenclature." (:34046) gives the italic
locants N (the nitrogen of the C=O) and N' (the terminal one); (:33900) "The method of naming
hydrazides as acyl derivatives of hydrazine is no longer recommended". (:18162): hydrazides are
class 12.

Before this module every producer wrote a wrong name for the class ('1-(acetamidoamino)ethanehydrazide',
'2-phenyl-1-oxa3,4-diazahexanehydrazide', 'benzohydrazide' with the benzoyl dropped); only the OPSIN gate
and kept them off the output.

Built like the N-substituted sulfonamides (``rules.sulfonamides``): the junior acyl and the N substituents are
EXCISED, the remaining R-CO-NH-NH2 is named by the pipeline (so the parent spelling is never written twice:
benzohydrazide, thiophene-2-carbohydrazide, propanehydrazide), and the junior acyl (the shared acyl-prefix
primitive, ``assembly.substituent_naming.acyl_prefix_from_branch``) and the N substituents join the parent's
own prefix in ONE alphanumerically ordered block (``rules.locanted_prefixes``), '4-methoxy-N'-(...acetyl)
benzohydrazide'. The parent prefix is read off the parent name against the structurally derived bare stem;
anything that cannot be split with certainty is declined (None), and the old path runs unchanged.

Scope: exactly one acyclic N-N bond that carries exactly two acyl groups R-CO- (R a carbon), further
N substituents carbon groups; a neutral molecule with no stereo element; the senior acid decided by
 (1) 'a ring or ring system has seniority over a chain' (:19340) and, between two chains, by the
longer one; a tie that is not two identical acyl groups is declined. A parent that carries more than one
prefix group is declined (the merge of several groups is not built).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

# C(=O)-N-N-C(=O): the carbonyl carbons and both nitrogens acyclic
_DIACYL = Chem.MolFromSmarts("[CX3;!R](=[OX1])[NX3;!R][NX3;!R][CX3;!R](=[OX1])")

_CHAIN_STEM = re.compile(
    r"(?:acetohydrazide|(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec|tetradec|pentadec|"
    r"hexadec|heptadec|octadec|nonadec|icos)an(?:e)?hydrazide)$")
_ONE_GROUP = re.compile(r"^(?P<loc>\d+[A-Za-z]?)-(?P<name>.+)$")


def _branch(mol: Any, start: int, blocked: "set[int]") -> List[int]:
    seen, order, stack = set(blocked), [], [start]
    while stack:
        a = stack.pop()
        if a in seen:
            continue
        seen.add(a)
        order.append(a)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors() if nb.GetIdx() not in seen)
    return order


def _chain_length(mol: Any, c: int, n: int) -> int:
    """Carbons of the longest acyclic carbon path from the carbonyl carbon ``c`` (itself included)."""
    best = [1]

    def walk(a: int, seen: "set[int]") -> None:
        best[0] = max(best[0], len(seen))
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j not in seen and j != n and nb.GetSymbol() == "C" and not nb.IsInRing() and not nb.GetIsAromatic():
                walk(j, seen | {j})

    walk(c, {c})
    return best[0]


def find_pair(mol: Any) -> Optional[Dict[str, Any]]:
    """The one N-N unit with two acyl groups, classified, or None when this module does not own the molecule."""
    if mol is None or Chem.GetFormalCharge(mol) != 0:
        return None
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
           for a in mol.GetAtoms()):
        return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        return None
    hits = mol.GetSubstructMatches(_DIACYL)
    if len({frozenset((h[2], h[3])) for h in hits}) != 1:
        return None
    c_a, _o_a, n_a, n_b, c_b, _o_b = hits[0]
    units = []
    claimed = {n_a, n_b}
    for c, n, other in ((c_a, n_a, n_b), (c_b, n_b, n_a)):
        c_at = mol.GetAtomWithIdx(c)
        r = [nb for nb in c_at.GetNeighbors() if nb.GetIdx() != n and nb.GetSymbol() != "O"]
        if len(r) != 1 or r[0].GetSymbol() != "C":
            return None                        # formyl, or carbamoyl/alkoxycarbonyl: not this class
        acyl = _branch(mol, c, {n})
        if {n_a, n_b} & set(acyl):
            return None
        extra = [nb for nb in mol.GetAtomWithIdx(n).GetNeighbors() if nb.GetIdx() not in (c, other)]
        if len(extra) > 1:
            return None
        sub = None
        if extra:
            e = extra[0]
            if e.GetSymbol() != "C" or any(
                    nb.GetSymbol() == "O" and mol.GetBondBetweenAtoms(e.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                    for nb in e.GetNeighbors()):
                return None                    # a third acyl (triacyl) or a heteroatom substituent
            sub = _branch(mol, e.GetIdx(), {n})
        units.append({"c": c, "n": n, "r": r[0].GetIdx(), "acyl": acyl, "sub": sub,
                      "sub_attach": None if sub is None else extra[0].GetIdx(),
                      "ring": r[0].IsInRing(), "len": _chain_length(mol, c, n)})
        claimed |= set(acyl) | set(sub or ())
    if len(claimed) != mol.GetNumHeavyAtoms():
        return None                            # an atom outside the two acyls, the Ns and their substituents
    return {"units": units}


def _senior(mol: Any, units: List[Dict[str, Any]], names: List[str]) -> Optional[int]:
    """Index of the unit of the senior acid, or None when the two cannot be told apart safely."""
    a, b = units
    if a["ring"] != b["ring"]:
        return 0 if a["ring"] else 1           # (1): a ring has seniority over a chain
    if not a["ring"] and a["len"] != b["len"]:
        return 0 if a["len"] > b["len"] else 1
    return 0 if names[0] == names[1] else None


def _bare_ring_hydrazide(mol: Any, unit: Dict[str, Any]) -> Optional[str]:
    """SMILES of ring-CO-NHNH2 for the unit's ring with every ring substituent removed (a monocycle only)."""
    ring = next((set(r) for r in mol.GetRingInfo().AtomRings() if unit["r"] in r), None)
    if ring is None or any(mol.GetRingInfo().NumAtomRings(i) != 1 for i in ring):
        return None
    keep = ring | {unit["c"], unit["n"]}
    keep |= {nb.GetIdx() for nb in mol.GetAtomWithIdx(unit["c"]).GetNeighbors() if nb.GetSymbol() == "O"}
    other = next(nb.GetIdx() for nb in mol.GetAtomWithIdx(unit["n"]).GetNeighbors() if nb.GetSymbol() == "N")
    keep.add(other)
    return _fragment_smiles(mol, keep, {unit["n"], other})


def _fragment_smiles(mol: Any, keep: "set[int]", hydrazine_n: "set[int]") -> Optional[str]:
    """SMILES of the atoms ``keep``; only the hydrazide nitrogens ``hydrazine_n`` get their hydrogens back."""
    try:
        new_index = {old: i for i, old in enumerate(sorted(keep))}
        rw = Chem.RWMol(mol)
        for idx in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
            rw.RemoveAtom(idx)
        out = rw.GetMol()
        for old in hydrazine_n:
            atom = out.GetAtomWithIdx(new_index[old])
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
        Chem.SanitizeMol(out)
        return Chem.MolToSmiles(out)
    except Exception:  # noqa: BLE001 - a fragment RDKit cannot close is not built
        return None


def _split_parent_name(mol: Any, unit: Dict[str, Any], parent_name: str,
                       style: str) -> Optional[Tuple[List[Tuple[str, str]], str]]:
    """(prefix pieces, stem) of the parent hydrazide's name, or None when the split is not certain."""
    if unit["ring"]:
        from ..namer import name_compound
        smiles = _bare_ring_hydrazide(mol, unit)
        stem = name_compound(smiles, style=style) if smiles else None
        if not stem or not stem.endswith("hydrazide") or not parent_name.endswith(stem):
            return None
        block = parent_name[: len(parent_name) - len(stem)]
    else:
        found = _CHAIN_STEM.search(parent_name)
        if not found:
            return None
        stem, block = found.group(0), parent_name[: found.start()]
    block = block.rstrip("-")
    if not block:
        return [], stem
    one = _ONE_GROUP.match(block)
    if one is None:
        return None
    name = one.group("name")
    depth = 0
    for i, ch in enumerate(name):
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth == 0 and ch == "-" and name[i + 1:i + 2].isdigit():
            return None                        # a second prefix group: the merge is not built
    if name[0] in "([{" and name[-1] in ")]}":
        inner, d, whole = name[1:-1], 0, True
        for ch in inner:
            d += ch in "([{"
            d -= ch in ")]}"
            whole &= d >= 0
        if whole and d == 0:
            name = inner
    return [(name, one.group("loc"))], stem


def name_diacylhydrazide(mol: Any, style: str = "pin") -> Optional[str]:
    """The PIN of a 1,2-diacylhydrazine, or None (the caller's other producers run unchanged)."""
    from ..assembly.composition_primitives import _join_prefix_to_name
    from ..assembly.substituent_enumerator import name_substituent
    from ..assembly.substituent_naming import acyl_prefix_from_branch
    from ..errors import is_failure_name
    from ..namer import name_compound
    from .locanted_prefixes import locanted_prefix_block

    pair = find_pair(mol)
    if pair is None:
        return None
    units = pair["units"]
    acyls = []
    for u in units:
        nm = acyl_prefix_from_branch(mol, u["c"], u["n"], u["acyl"])
        if not nm or nm == "substituent" or " " in nm:
            return None
        acyls.append(nm)
    s = _senior(mol, units, acyls)
    if s is None:
        return None
    senior, junior = units[s], units[1 - s]

    pieces: List[Tuple[str, str]] = []
    for unit, locant in ((senior, "N"), (junior, "N'")):
        if unit["sub"] is not None:
            nm = name_substituent(mol, set(unit["sub"]), unit["sub_attach"])
            if not nm or nm == "substituent" or " " in nm:
                return None
            pieces.append((nm, locant))
    pieces.append((acyls[1 - s], "N'"))

    gone = set(junior["acyl"]) | set(senior["sub"] or ()) | set(junior["sub"] or ())
    keep = set(range(mol.GetNumAtoms())) - gone
    parent_smiles = _fragment_smiles(mol, keep, {senior["n"], junior["n"]})
    parent_name = name_compound(parent_smiles, style=style) if parent_smiles else None
    if not parent_name or is_failure_name(parent_name) or not parent_name.endswith("hydrazide"):
        return None
    split = _split_parent_name(mol, senior, parent_name, style)
    if split is None:
        return None
    parent_pieces, stem = split
    block = locanted_prefix_block(parent_pieces + pieces)
    return _join_prefix_to_name(block, stem) if block else None


__all__ = ["find_pair", "name_diacylhydrazide"]
