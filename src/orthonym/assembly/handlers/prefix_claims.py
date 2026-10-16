"""Atom-claim reconciliation of the prefix fragments ``_generate_prefixes`` builds.

Every prefix fragment reports the heavy atoms it spells (``NameFragment.atoms``).
Two producers of the SAME list can spell one atom twice, and the second spelling
is a different molecule:

* **Double spelling (S1).** A branch prefix ('2-[(2R)-2-amino-3-sulfanylpropan-
  amido]', atoms {2..8}) already spells its thiol; the functional-group loop then
  cites the same thiol again as a bare, unlocated 'sulfanyl' (atoms {7, 8}), so the
  name carries a sulfur the molecule does not have and the residues it dropped are
  never missed. The BUG-B filter in ``_generate_prefixes`` trusts a branch only for
  nine simple group types and only up to three carbons; thiols, guanidines and
  amides are outside it. A fragment whose atoms are a STRICT subset of another
  fragment's atoms is spelled by that fragment and is dropped here. A group anchored
  on the chain contains a chain atom, which no branch does, so it is never a subset
  of a branch and is untouched.
* **A chain amide spelled 'carbamoyl' (S2).** (the Blue Book):
  "The combination of the prefixes 'amino' and 'oxo' is used for describing the
  -CO-NH2 at the end of an acyclic chain in preferred IUPAC names"
  ('4-amino-4-oxobutanoic acid (PIN)',:30384; '3-carbamoylpropanoic acid' is not).
  When the amide carbon is a chain atom the loop cited 'carbamoyl' at that locant,
  one carbon too many ('5-carbamoylpentanoic acid' for glutamine). The matches that
  sit on the chain are respelled 'amino' + 'oxo'; matches inside a branch are claimed
  by the branch.

Carbonic-acid derivatives are left alone as well. A one-carbon chain parent that holds a C=X bond and
two single-bonded N/O/S (urea, carbamate, carbonate) is named on its functional parent,
'N-(propan-2-ylidene)urea (PIN)'
,:33308; example:33332), never as '1-(propan-2-ylideneamino)formamide', which is what the
reconciled list would ship for the urea that the urea handler declines.

Pseudohalide groups are left alone too. The branch namer spells -S-C#N '(cyanosulfanyl)methyl', where the
Blue Book gives "'cyano' for -CN and 'cyanato' for -O-CN, 'thiocyanato' for -S-CN, 'selenocyanato' for -Se-CN"
as the preferred prefixes "Cyanic acid", the Blue Book; '3-(thiocyanato)propanoic acid (PIN)'
:31013); dropping the group's own prefix would ship the branch's spelling as a PIN. A dropped prefix that
is a cyanate, thiocyanate, selenocyanate, isocyanate or isothiocyanate group leaves the list unchanged, so
the double spelling keeps the name withheld.

Hydrazine branches are left alone. A branch with an N-N bond is spelled
'[(carbamoylamino)amino]' by the branch namer where the Blue Book writes
'2-carbamoylhydrazin-1-yl',:38644) or '...hydrazido',
:34548); reconciling such a list would turn a withheld name into a shipped non-PIN
name, so the list is returned unchanged and the double spelling keeps the name
withheld.

The function reads structured fragments and perception data only; it never edits a
name string.
"""
from __future__ import annotations

import logging
from typing import Any, List

from rdkit import Chem

logger = logging.getLogger(__name__)


def _has_nn_bond(mol: Any, atoms) -> bool:
    """True when an N atom of ``atoms`` is bonded to another N (a hydrazine-type link)."""
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N' and any(n.GetSymbol() == 'N' for n in atom.GetNeighbors()):
            return True
    return False


def _has_pseudohalide(mol: Any, atoms) -> bool:
    """True when ``atoms`` hold X-C#N (X = O, S, Se, Te: cyanate and its chalcogen analogues) or
    N=C=X (isocyanate, isothiocyanate, isoselenocyanate)."""
    chalcogen = ('O', 'S', 'Se', 'Te')
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        bonds = [(b.GetBondType(), b.GetOtherAtom(atom).GetSymbol()) for b in atom.GetBonds()]
        if (any(t == Chem.BondType.TRIPLE and e == 'N' for t, e in bonds)
                and any(t == Chem.BondType.SINGLE and e in chalcogen for t, e in bonds)):
            return True
        doubles = [e for t, e in bonds if t == Chem.BondType.DOUBLE]
        if 'N' in doubles and any(e in chalcogen for e in doubles):
            return True
    return False


def _on_carbonic_parent(features: Any) -> bool:
    """True when the chain parent is one carbon with a double bond to N, O or S and two or more
    single bonds to N, O or S: a carbonic-acid derivative (urea, carbamate, carbonate, isourea,
    guanidine). An aminal or acetal carbon has no C=X bond and is not one."""
    chain = getattr(features, 'principal_chain', None) or ()
    if len(chain) != 1:
        return False
    atom = features.mol.GetAtomWithIdx(chain[0])
    hetero = [(b.GetBondType(), b.GetOtherAtom(atom).GetSymbol()) for b in atom.GetBonds()]
    return (sum(1 for t, e in hetero if t == Chem.BondType.SINGLE and e in ('N', 'O', 'S')) >= 2
            and any(t == Chem.BondType.DOUBLE and e in ('N', 'O', 'S') for t, e in hetero))


def _claimed_by(mol: Any, atoms, prefixes: List[Any], own: Any, strict: bool):
    """The prefix fragment other than ``own`` whose atoms contain ``atoms`` (strictly
    when ``strict``), or None."""
    for g in prefixes:
        if g is own or not g.atoms:
            continue
        if (atoms < g.atoms) if strict else (atoms <= g.atoms):
            return g
    return None


def _drop_double_spelled(mol: Any, prefixes: List[Any]):
    """S1. Returns the list without the fragments another fragment already spells, or
    None when a container holds an N-N bond or a dropped prefix is a pseudohalide group (the
    whole list is then left unchanged)."""
    kept = []
    for f in prefixes:
        atoms = f.atoms
        if atoms:
            g = _claimed_by(mol, atoms, prefixes, f, strict=True)
            if g is not None:
                if _has_nn_bond(mol, g.atoms) or _has_pseudohalide(mol, atoms):
                    return None
                logger.debug("prefix_claims: %r (atoms %s) is spelled by %r; dropped",
                             f.text, sorted(atoms), g.text)
                continue
        kept.append(f)
    return kept


def _respell_chain_amides(features: Any, prefixes: List[Any]) -> List[Any]:
    """S2. Respell the chain-carbon primary amides of a 'carbamoyl' fragment as
    'amino' + 'oxo'; the rest of the fragment keeps 'carbamoyl'."""
    chain = set(getattr(features, 'principal_chain', None) or ())
    amides = (getattr(features, 'functional_groups', None) or {}).get('primary_amide') or []
    on_chain = [m for m in amides if len(m) >= 3 and m[0] in chain]
    if not on_chain:
        return prefixes
    target = next((f for f in prefixes if f.text == 'carbamoyl' and f.atoms), None)
    if target is None:
        return prefixes
    covered = [m for m in on_chain if set(m) <= target.atoms]
    if not covered:
        return prefixes
    # What the fragment still has to say: its other matches, unless a branch spells them.
    rest = [m for m in amides
            if m not in covered and set(m) <= target.atoms
            and _claimed_by(features.mol, frozenset(m), prefixes, target, strict=False) is None]
    from ..composer import NameFragment, _get_fg_locants
    if len(rest) > 1:
        return prefixes  # a multiplied 'carbamoyl' is not rebuilt here; the list stays as built
    out: List[Any] = []
    for f in prefixes:
        if f is not target:
            out.append(f)
            continue
        if rest:
            out.append(NameFragment(
                text='carbamoyl',
                locants=tuple(sorted(set(_get_fg_locants(features, 'primary_amide', rest)))),
                fragment_type='prefix', atoms=frozenset(rest[0])))
        for c, o, n in (m[:3] for m in covered):
            locant = _get_fg_locants(features, 'primary_amide', [(c, o, n)])
            if not locant:
                return prefixes
            out.append(NameFragment(text='amino', locants=(locant[0],),
                                    fragment_type='prefix', atoms=frozenset({n})))
            out.append(NameFragment(text='oxo', locants=(locant[0],),
                                    fragment_type='prefix', atoms=frozenset({c, o})))
    return out


def reconcile_prefix_claims(features: Any, prefixes: List[Any]) -> List[Any]:
    """Return ``prefixes`` with every atom spelled once (S1) and chain amides spelled
    'amino' + 'oxo' (S2); the input list when nothing applies or a hydrazine branch is
    involved, a pseudohalide prefix would be dropped, or the parent is a carbonic-acid derivative."""
    mol = getattr(features, 'mol', None)
    if mol is None or not prefixes:
        return prefixes
    if _on_carbonic_parent(features):
        return prefixes
    kept = _drop_double_spelled(mol, prefixes)
    if kept is None:
        return prefixes
    return _respell_chain_amides(features, kept)
