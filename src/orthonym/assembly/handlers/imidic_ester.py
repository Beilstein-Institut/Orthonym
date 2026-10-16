"""Esters of imidic acids that the =NH oxygen-ester handler (``imidate``) does not own.

IUPAC cite: "Esters of acids modified by functional replacement nomenclature",
 (the Blue Book-31999): 'methyl ethanimidate (PIN)', 'ethyl propanehydrazonate
(PIN)', 'methyl benzenecarboximidothioate (PIN)', 'ethyl N-sulfanylbenzenecarboximidothioate (PIN)';
 'ethyl N,N-dimethylcarbamimidothioate (PIN)'. (the Blue Book):
imidic and hydrazonic acids follow the carboxylic acids in class 7a, so their esters are class 9, senior
to every group that is not an acid, an anhydride or an ester.

What owned these molecules before: ``ester_family`` (the sulfur ester came out as
'1-(methylsulfanyl)ethan-1-imine'), the ``imine`` handler (an N-substituted ester came out as
'1-ethoxy-N-methylethan-1-imine') and the ``hydrazone`` handler ('(1-ethoxypropylidene)hydrazine'),
each shipped as ``pin_verified``. A name for the same structure that puts the ester oxygen or sulfur on
the imine carbon as a prefix is a systematic name, not the PIN.

Scope (anything else returns None and the old path runs unchanged, so no row loses a name):
one acyclic ``R-C(=N-X)-Y-R'`` group, Y = O or S, no other principal group, a neutral molecule with no
stereo element; R = H, an acyclic carbon chain whose first carbon is saturated (its unsaturation is
cited by 'ene'/'yne'; the chain is the longest,, or an unsubstituted benzene ring
(R = N gives the S-ester of carbamimidic acid); X = H, a carbon group, NH2 (the O-ester hydrazonate), or
OH "Carbohydroximic acids": 'N-hydroxyethanimidic acid (PIN)', so 'methyl
N-hydroxyethanimidate'). The =NH O-esters and the O-carbamimidates stay with ``imidate`` (priority
2900). Registered at priority 90, ahead of ``oxime`` (100), ``hydrazone`` (200) and every handler
listed above.

PURE per: predicate and builder only read ``mol``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from ...rules.locanted_prefixes import locanted_prefix_block
from ..name_tree import NameTreeNode, NamingResult
from .imidate import _collect_subgraph, _name_alkyl_fragment, _name_carbamimidate

# (C, =N, O or S, alkyl C): the imidic ester group; every atom outside a ring
_IMIDIC_ESTER = Chem.MolFromSmarts("[CX3;!R](=[NX2;!R])[OX2,SX2;!R][#6]")
_C_DOUBLE_N = Chem.MolFromSmarts("[#6]=[#7]")

# stem endings by the acid they come from: 'imidic', 'imidothioic', 'hydrazonic' (the 'e' of the
# alkane is elided before a vowel only: ethanimidate, propanehydrazonate)
_CHAIN_SUFFIX = {"imidate": "animidate", "imidothioate": "animidothioate", "hydrazonate": "anehydrazonate"}
_RING_SUFFIX = {"imidate": "carboximidate", "imidothioate": "carboximidothioate",
                "hydrazonate": "carbohydrazonate"}


def _heavy_neighbors(atom, exclude: "set[int]") -> list:
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in exclude]


def _parts(mol: Any) -> Optional[Dict[str, Any]]:
    """The one imidic ester group of ``mol`` classified, or None when it is not this handler's."""
    if mol is None or Chem.GetFormalCharge(mol) != 0:
        return None
    if any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds()):
        return None
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        return None
    hits = mol.GetSubstructMatches(_IMIDIC_ESTER)
    if len(hits) != 1 or len(mol.GetSubstructMatches(_C_DOUBLE_N)) != 1:
        return None                       # a second C=N would be left unnamed
    c, n, y, alkyl = hits[0]
    c_at, n_at, y_sym = mol.GetAtomWithIdx(c), mol.GetAtomWithIdx(n), mol.GetAtomWithIdx(y).GetSymbol()
    others = _heavy_neighbors(c_at, {n, y})
    if len(others) > 1:
        return None
    r = others[0] if others else None
    stem = "H" if r is None else ("N" if r.GetSymbol() == "N" else ("C" if r.GetSymbol() == "C" else None))
    if stem is None:
        return None
    nx = _heavy_neighbors(n_at, {c})
    if not nx:
        x = "H"
    elif len(nx) == 1 and nx[0].GetSymbol() == "C":
        x = "C"
    elif len(nx) == 1 and nx[0].GetSymbol() == "N" and nx[0].GetTotalNumHs() == 2 and nx[0].GetDegree() == 1:
        x = "N"
    elif len(nx) == 1 and nx[0].GetSymbol() == "O" and nx[0].GetTotalNumHs() == 1 and nx[0].GetDegree() == 1:
        x = "O"                           # N-hydroxy "Carbohydroximic acids")
    else:
        return None
    if stem == "N":
        # the S-ester of carbamimidic acid; the amino N must be acyclic; no hydrazonate
        if y_sym != "S" or x in ("N", "O") or r.IsInRing() or r.GetIsAromatic():
            return None
    elif y_sym == "O" and x == "H":
        return None                       # the =NH O-ester: imidate names it
    elif y_sym == "S" and x == "N":
        return None                       # 'hydrazonothioate' has no example in the book: the old path runs
    if stem == "C" and not _stem_ok(mol, r):
        return None
    return {"c": c, "n": n, "y": y, "alkyl": alkyl, "r": None if r is None else r.GetIdx(),
            "stem": stem, "x": x, "n_sub": nx[0].GetIdx() if x == "C" else None,
            "y_sym": y_sym}


def _stem_ok(mol: Any, r) -> bool:
    """R is a saturated acyclic carbon (a chain) or an unsubstituted benzene ring carbon."""
    if r.GetIsAromatic():
        ring = [list(rg) for rg in mol.GetRingInfo().AtomRings() if r.GetIdx() in rg]
        if len(ring) != 1 or len(ring[0]) != 6:
            return False
        for idx in ring[0]:
            at = mol.GetAtomWithIdx(idx)
            if at.GetSymbol() != "C" or not at.GetIsAromatic() or at.GetDegree() != (3 if idx == r.GetIdx() else 2):
                return False
        return True
    return (not r.IsInRing()) and r.GetHybridization() == Chem.HybridizationType.SP3


# What the seniority cascade reports for a molecule whose most senior group is the C=N of the ester
# (nothing senior to it survives: an acid, anhydride, other ester, amide, alcohol or amine would be the
# principal group instead).
_PG_OF_THE_C_N = (None, "imine", "hydrazone", "oxime", "amidine", "iminoester")


def _is_imidic_ester(features: Any) -> bool:
    """Predicate (pure): one imidic ester group and no group senior to it, class 9)."""
    if getattr(features, "principal_group", None) not in _PG_OF_THE_C_N:
        return False
    return _parts(getattr(features, "mol", None)) is not None


def _chain(mol: Any, start: int) -> Tuple[List[int], List[int], List[int]]:
    """The principal chain from ``start``: (atoms, double-bond locants, triple-bond locants).

     "Greater number of skeletal atoms" first, then "Seniority of... the principal
    chain": (a) the greater number of multiple bonds, (b) the greater number of double bonds
    , (j) the lower locants for 'ene' and 'yne' endings; what remains goes to
    the lowest substituent locants. The suffix sits at C-1, so criterion (h) never
    separates two chains.
    """
    def carbon_ok(idx: int) -> bool:
        at = mol.GetAtomWithIdx(idx)
        return at.GetSymbol() == "C" and not at.IsInRing() and not at.GetIsAromatic()

    paths: List[List[int]] = []
    best = [0]

    def walk(path: List[int]) -> None:
        ext = False
        for nb in mol.GetAtomWithIdx(path[-1]).GetNeighbors():
            j = nb.GetIdx()
            if j in path or not carbon_ok(j):
                continue
            ext = True
            walk(path + [j])
        if not ext:
            if len(path) > best[0]:
                best[0], paths[:] = len(path), [path]
            elif len(path) == best[0]:
                paths.append(path)

    walk([start])

    def multiple(path: List[int]) -> Tuple[List[int], List[int]]:
        dbl: List[int] = []
        trp: List[int] = []
        for k in range(len(path) - 1):
            order = mol.GetBondBetweenAtoms(path[k], path[k + 1]).GetBondType()
            if order == Chem.BondType.DOUBLE:
                dbl.append(k + 1)
            elif order == Chem.BondType.TRIPLE:
                trp.append(k + 1)
        return dbl, trp

    def key(path: List[int]):
        dbl, trp = multiple(path)
        on = {a: i + 1 for i, a in enumerate(path)}
        locs = sorted(on[i] for i in path for a in mol.GetAtomWithIdx(i).GetNeighbors()
                      if a.GetIdx() not in on and a.GetAtomicNum() > 1)
        return (-(len(dbl) + len(trp)), -len(dbl), sorted(dbl + trp), dbl, locs, [on[i] for i in path[1:]])

    chosen = min(paths, key=key)
    dbl, trp = multiple(chosen)
    return chosen, dbl, trp


def _usable(word: Optional[str]) -> bool:
    return bool(word) and word != "substituent" and " " not in word


def build_imidic_ester_name(mol: Any) -> Optional[str]:
    """The PIN of one imidic ester group, or None when this builder does not own the molecule."""
    p = _parts(mol)
    if p is None:
        return None
    c, n, y, alkyl = p["c"], p["n"], p["y"], p["alkyl"]
    if p["stem"] == "N":
        return _name_carbamimidate(mol, (c, n, p["r"], y, alkyl), thio=True)
    alkyl_word = _name_alkyl_fragment(mol, _collect_subgraph(mol, alkyl, {y}), alkyl)
    if not _usable(alkyl_word):
        return None
    suffix = "hydrazonate" if p["x"] == "N" else ("imidothioate" if p["y_sym"] == "S" else "imidate")
    pieces: "List[Tuple[str, str]]" = []
    if p["x"] == "C":
        from ..substituent_enumerator import name_substituent
        frag = _collect_subgraph(mol, p["n_sub"], {n})
        nm = name_substituent(mol, set(frag), p["n_sub"])
        if not _usable(nm):
            return None
        pieces.append((nm, "N"))
    elif p["x"] == "O":
        pieces.append(("hydroxy", "N"))
    if p["stem"] == "H":
        from ...data.chain_names import get_chain_prefix
        stem = get_chain_prefix(1) + _CHAIN_SUFFIX[suffix]
    elif mol.GetAtomWithIdx(p["r"]).GetIsAromatic():
        stem = "benzene" + _RING_SUFFIX[suffix]
    else:
        from ...data.chain_names import get_chain_prefix
        from ..composition_primitives import _build_unsaturation_infix
        from ..naming_utils import format_suffix_with_locants
        from ..substituent_enumerator import name_substituent
        chain, dbl, trp = _chain(mol, c)
        try:
            stem = format_suffix_with_locants(get_chain_prefix(len(chain)), _build_unsaturation_infix(dbl, trp),
                                              suffix, [])
        except (ValueError, KeyError):
            return None
        on = {a: i + 1 for i, a in enumerate(chain)}
        ring_or_chain = set(chain) | {n, y}
        for i in chain:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if j in on or j in ring_or_chain or nb.GetAtomicNum() <= 1:
                    continue
                frag = _collect_subgraph(mol, j, {i})
                if set(frag) & set(chain):
                    return None
                nm = name_substituent(mol, set(frag), j)
                if not _usable(nm):
                    return None
                pieces.append((nm, str(on[i])))
    block = locanted_prefix_block(pieces) if pieces else ""
    if block is None:
        return None
    return f"{alkyl_word} {block}{stem}"


def name_imidic_ester(features: Any, mol: Any = None, style: str = "pin") -> Optional[NamingResult]:
    """Handler: the imidic ester PIN, or None (gate-fail, -04) so the next handler runs."""
    from ..candidate_pool import get_current_pool

    mol = mol if mol is not None else getattr(features, "mol", None)
    name = build_imidic_ester_name(mol)
    if not name:
        return None
    cand = get_current_pool().add(name, "imidate", features)
    if cand is None:
        return None
    return NamingResult(
        name=cand.name,
        tree=NameTreeNode(parent_stem=cand.name, class_id="imidate",
                          iupac_section_cite="P-65.6.3.3.7.1", fragment_legacy=cand.name),
        atom_to_locant_hint=None,
    )


__all__ = ["name_imidic_ester", "_is_imidic_ester", "build_imidic_ester_name"]
