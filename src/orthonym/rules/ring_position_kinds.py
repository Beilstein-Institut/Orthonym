"""Kinds of substitutable position of a free ring or a parent hydride, for the writers of a
ring prefix and of an ionic or amine suffix."""
from rdkit import Chem, rdBase


def one_kind_of_ring_position(mol, ring_atoms) -> bool:
    """ (the Blue Book): "The locant is omitted in monosubstituted symmetrical
    parent hydrides or parent compounds where there is only one kind of substitutable
    hydrogen" -- 'oxiranyl' (:18929, '(oxiranylmethoxy)silane (PIN)'), not 'oxiran-2-yl'.
    Read on the ring as its own parent hydride: the ring atoms and ring bonds alone
    (implicit hydrogens recomputed), positions = the ring atoms that carry hydrogen there.
    A ring that cannot be read this way counts as having more than one kind, so the
    caller cites the locant,:2869).

    A position that carries an isotope label (a labelled ring atom, or an isotopic hydrogen
    on one) is its own kind of position, so the licence does not apply to a labelled ring:
     Letter and/or numeral locants (:44208): "When locants are needed for defining
    the structure of the parent structure or of a unit of structure as defined by its
    appropriate enclosing marks, then all locants must be cited for the parent structure or
    that structural unit" -- the labelled ring cites its free-valence locant. The label is
    read from ``mol`` and from the labelled molecule that the isotope path declares while it
    names the label-free skeleton."""
    from ..assembly.locant_omission import isotope_label_on_atoms

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsotope() or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                                    for n in atom.GetNeighbors()):
            return False
    if isotope_label_on_atoms(mol, ring_atoms):
        return False
    try:
        with rdBase.BlockLogs():
            frag = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(
                mol, atomsToUse=list(ring_atoms), isomericSmiles=False))
    except Exception:  # noqa: BLE001 - an unreadable ring is not shown to be symmetric
        return False
    if frag is None:
        return False
    ranks = list(Chem.CanonicalRankAtoms(frag, breakTies=False))
    kinds = {ranks[a.GetIdx()] for a in frag.GetAtoms() if a.GetTotalNumHs() > 0}
    return len(kinds) == 1


def one_kind_of_substitutable_position(parent) -> bool:
    """ (the Blue Book): "The locant is omitted in monosubstituted symmetrical
    parent hydrides or parent compounds where there is only one kind of substitutable
    hydrogen". ``parent`` is the parent hydride or parent compound as one molecule, ring or
    chain: every atom that carries a hydrogen is of one kind (one symmetry class), so a suffix
    cited on it needs no locant ('benzenaminide (PIN)',:41057). A parent that carries other
    groups, or that has two kinds of position, is not, and the caller cites the locant
    ,:2869)."""
    ranks = list(Chem.CanonicalRankAtoms(parent, breakTies=False, includeChirality=False))
    return len({ranks[a.GetIdx()] for a in parent.GetAtoms() if a.GetTotalNumHs() > 0}) == 1
