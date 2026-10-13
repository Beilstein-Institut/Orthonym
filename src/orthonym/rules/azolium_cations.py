"""Names of ring cations of the azolium kind: the added hydron and the indicated hydrogen of the
parent hydride sit on two ring nitrogens, the Blue Book), and the substituents
replace the hydrogens. The producers of ions._emit_ring_cumulative_suffix name a ring cation
from the numbering of its neutral ring, which cannot place the indicated hydrogen and the '-ium' on
different nitrogens when both carry a substituent, or when both are protonated."""
import re

from rdkit import Chem

from ..errors import is_failure_name
from .ions import (
    _elide_terminal_e,
    _join_prefix_stem,
    _p74_ring_substituent_groups,
    _p74_ring_substituent_prefix,
    _ring_iupac_locants,
    _ring_locant_order_key,
)


def azolium_hydron_centres(mol, ring_system, center_idx):
    """The ring nitrogens at which the charge of the azolium ``mol`` can be drawn, ``center_idx``
    first: an aromatic ring N bearing an exocyclic substituent and no hydrogen, with the positive
    charge on it in a resonance structure of ``mol`` and none elsewhere. In 1,3-dimethylimidazolium
    both nitrogens are such centres; the cation is one species and the name cannot depend on which
    nitrogen the drawing charges."""
    try:
        sup = Chem.ResonanceMolSupplier(mol, 0, 64)
        structs = [sup[i] for i in range(len(sup))]
    except Exception:
        return [center_idx]
    found = {center_idx}
    for r in structs:
        if r is None:
            continue
        charged = [i for i in ring_system if r.GetAtomWithIdx(i).GetFormalCharge() == 1]
        if len(charged) != 1 or charged[0] in found:
            continue
        a = mol.GetAtomWithIdx(charged[0])
        if (a.GetAtomicNum() == 7 and a.GetIsAromatic() and a.GetTotalNumHs() == 0
                and a.GetDegree() == 3):
            found.add(charged[0])
    return [center_idx] + sorted(found - {center_idx})


def _bare_parent(mol, ring_system, centre):
    """The neutral parent hydride of an azolium whose ring nitrogens all carry a substituent:
    the ring with every substituent replaced by hydrogen, one aromatic ring nitrogen (the
    indicated-hydrogen position) bearing that hydrogen, ``centre`` (the position of the added
    hydron) bearing none. ``(parent_mol, ring_sorted)``, ``ring_sorted[k]`` being the atom of
    ``mol`` that is atom ``k`` of ``parent_mol``; None when it does not sanitize."""
    ring = sorted(set(ring_system))
    work = Chem.RWMol(mol)
    pyrrole_like = {i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                    and mol.GetAtomWithIdx(i).GetIsAromatic()
                    and mol.GetAtomWithIdx(i).GetDegree() == 3}
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in ring),
                      reverse=True):
        work.RemoveAtom(idx)
    try:
        for k, i in enumerate(ring):
            a = work.GetAtomWithIdx(k)
            a.SetFormalCharge(0)
            a.SetNoImplicit(False)
            if i == centre:
                a.SetNumExplicitHs(0)
            elif i in pyrrole_like:
                a.SetNumExplicitHs(1)
                a.SetNoImplicit(True)
        parent = work.GetMol()
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return None
    return parent, ring


def azolium_demote_name(mol, ring_system, center_idx):
    """ (heading the Blue Book): "A cation derived formally by adding one or more
    hydrons to any position of a neutral parent hydride... is named by replacing the final
    letter 'e' of the parent hydride name, if any, by the suffix 'ium'"; "1H-imidazol-3-ium (PIN)"
    (:41396). An azolium whose ring nitrogens all carry a substituent (1,3-dimethylimidazolium,
    1-ethyl-3-methylimidazolium, 1,3-dimethylbenzimidazolium) is that cation with the
    substituents replacing the hydrogens: the indicated hydrogen of the parent hydride and the
    added hydron sit on the two nitrogens, and a substituent stands on each.

    The parent hydride is the ring with every substituent replaced by hydrogen, one nitrogen
    bearing the indicated hydrogen (``_bare_parent``); its own numbering puts that
    hydrogen at the lowest locant, (heading:3219) criterion (b) "indicated hydrogen for
    unsubstituted compounds" (:3246), and the '-ium' takes the other nitrogen, criterion (c)
    "principal characteristic groups and free valences (suffixes)" (:3256). Which nitrogen is the
    indicated-hydrogen position is left open by the structure, so each nitrogen that can carry the
    charge is tried and the cation is named once, by the choice the rules leave: the locant set
    of the prefixes (f) "detachable alphabetized prefixes, all considered together in a series of
    increasing numerical order" (:3301), then (g) "lowest locants for the substituent cited first
    as a prefix in the name" (:3307; '1-methyl-4-nitronaphthalene (PIN)',:3318). The drawing therefore
    does not change the name. Every candidate is read back by OPSIN with the hydron on the same
    atoms (``_cation_name_rt_ok``); '' (decline) when none is, or the ring is not an azolium of
    this kind."""
    center = mol.GetAtomWithIdx(center_idx)
    if (center.GetAtomicNum() != 7 or not center.GetIsAromatic() or center.GetFormalCharge() != 1
            or center.GetTotalNumHs() != 0 or center.GetDegree() != 3):
        return ''
    ring = set(ring_system)
    others = [i for i in ring if i != center_idx and mol.GetAtomWithIdx(i).GetAtomicNum() == 7
              and mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetDegree() == 3]
    # Scope: another substituted aromatic nitrogen, no ring nitrogen with hydrogen (the
    # in-place and hydron-site constructions name those) and no other charge.
    if not others:
        return ''
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 and mol.GetAtomWithIdx(i).GetTotalNumHs()
           for i in ring):
        return ''
    if any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != center_idx):
        return ''
    from ..assembly.naming_utils import alpha_sort_key
    from .charged_router import _cation_name_rt_ok
    candidates = []
    for centre in azolium_hydron_centres(mol, ring_system, center_idx):
        built = _bare_parent(mol, ring_system, centre)
        if built is None:
            continue
        parent, ring_sorted = built
        try:
            from ..namer import Orthonym
            stem = Orthonym(style='pin', _disable_opsin_validity_gate=True
                            ).name(Chem.MolToSmiles(parent))
        except (RecursionError, ValueError, RuntimeError):
            continue
        if not stem or is_failure_name(stem):
            continue
        parent_locants = _ring_iupac_locants(parent)
        if not parent_locants or len(parent_locants) != len(ring_sorted):
            continue
        locants = {ring_sorted[k]: v for k, v in parent_locants.items() if k < len(ring_sorted)}
        if len(locants) != len(ring_sorted) or centre not in locants:
            continue
        # The indicated hydrogen the stem cites is the nitrogen it numbers: a stem that names
        # another position than the numbering used for the prefixes would join two numberings.
        m = re.match(r'^(\d+)H-', stem)
        nh = [i for i in ring_sorted if i != centre and i in others + [center_idx]]
        if m and (len(nh) != 1 or str(locants[nh[0]]) != m.group(1)):
            continue
        groups = _p74_ring_substituent_groups(mol, ring_system, locants, set())
        if groups is None:
            continue
        prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, set())
        if prefix is None:
            continue
        name = (f"{_join_prefix_stem(prefix, _elide_terminal_e(stem))}"
                f"-{locants[centre]}-ium")
        lk = _ring_locant_order_key
        every = sorted((lk(x) for locs in groups.values() for x in locs))
        by_token = tuple(tuple(sorted(lk(x) for x in groups[t]))
                         for t in sorted(groups, key=alpha_sort_key))
        candidates.append(((lk(locants[centre]), every, by_token, name), name))
    for _key, name in sorted(candidates):
        if _cation_name_rt_ok(name, mol):
            return name
    return ''


def ring_has_other_nh_heteroatom(mol, ring_system, center_idx):
    """True if a ring heteroatom other than ``center_idx`` carries hydrogen (the NH of a
    benzimidazolium or a triazolium drawn with both nitrogens protonated). The neutral
    parent then cites an indicated hydrogen ('1H-'), a position of its own numbering, and
    the '-ium' is cited at the centre's locant read from the numbering of the neutral
    ring, which does not know which of two equivalent nitrogens holds that hydrogen: the
    two locants can name one nitrogen twice ('1H-1,3-benzimidazol-1-ium', the NH2+ cation)."""
    return any(i != center_idx and mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)
               and mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1
               for i in ring_system)
