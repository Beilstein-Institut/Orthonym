"""Partially ionized polycarboxylic acids on one monocyclic ring method (1))."""
from typing import Dict

from rdkit import Chem

from .ions import (
    _anion_name_keeps_hydron_sites,
    _carboxyl_ring_anchor,
    _charged_ring_locants,
    _is_isolated_benzene_atom,
    _join_prefix_stem,
    _order_ring_cycle,
    _p74_bare_ring_stem,
    _p74_ring_substituent_prefix,
)


def name_ring_acid_salt_anion(mol, anion_site: Dict) -> str:
    """ method (1) (heading **** "Acid salts of polybasic organic acids are
    named in two ways", the Blue Book) for a partially ionized polycarboxylic acid whose
    ionized carboxy groups stand on one monocyclic ring: "(1) by substitutive nomenclature in
    which the free acid is cited as a prefix to the name of the anion" (:31593), and "Method (1)
    generates preferred IUPAC names, except when the structure of the acid salt is unknown."
    (:31596). The anion is senior to the acid, '4 Anions' before '7 Acids',:18167),
    so the ionized groups are the suffix and each un-ionized group is the 'carboxy' prefix:
    '2-carboxybenzoate' for O=C([O-])c1ccccc1C(=O)O, '4-carboxybenzoate' for the terephthalate,
    '3-carboxypyridine-2-carboxylate'. The retained 'benzoic acid' is the parent of a benzene ring
    with one ionized group,:29717, 'C6H5-COOH benzoic acid (PIN)'); any other ring,
    or two or more ionized groups, takes the '-carboxylate' / '-dicarboxylate' suffix on the
    parent hydride.

    ONE numbering for the whole name, the one the zwitterion builder uses: the ring
    heteroatoms, the indicated hydrogen, then the ionized groups (the suffix, (c)), then the
    prefixes (:3219; ``_charged_ring_locants`` with the ionized attachment atoms as the centres);
    every other substituent is cited in it by ``_p74_ring_substituent_prefix`` and the stem is the
    ring with every substituent deleted (``_p74_bare_ring_stem``). Declines ('') for a fused or
    bridged ring (no simple cycle), a ring system that is not the only ring, any charge outside
    the ionized groups, stereochemistry, a substituent the authoritative namer cannot spell, and
    a molecule that is not partially ionized. The name ships only when OPSIN reads it back to
    ``mol`` with the hydrons on the same atoms."""
    from ..perception.rings import get_ring_systems
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return ''
    if mol.GetRingInfo().NumRings() != 1:
        return ''
    if (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())):
        return ''
    from rdkit.Chem import MolFromSmarts
    free_pat = MolFromSmarts('[CX3](=O)[OX2H1]')
    if free_pat is None or not mol.HasSubstructMatch(free_pat):
        return ''
    charged = [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()]
    if not charged:
        return ''
    try:
        ring_system = None
        for rs in get_ring_systems(mol, include_spiro=True):
            ring_system = rs
            break
    except (RuntimeError, ValueError):
        return ''
    if ring_system is None or _order_ring_cycle(mol, ring_system) is None:
        return ''
    carboxyl_atoms = set()
    attach = []
    for idx in charged:
        o = mol.GetAtomWithIdx(idx)
        if o.GetAtomicNum() != 8 or o.GetFormalCharge() != -1:
            return ''
        anchor = _carboxyl_ring_anchor(mol, idx, ring_system)
        if anchor is None:
            return ''
        carboxyl_c, ring_attach = anchor
        attach.append(ring_attach)
        carboxyl_atoms.add(carboxyl_c)
        for nb in mol.GetAtomWithIdx(carboxyl_c).GetNeighbors():
            if nb.GetAtomicNum() == 8:
                carboxyl_atoms.add(nb.GetIdx())
    if len(set(attach)) != len(attach):
        return ''
    locants, _indicated_h = _charged_ring_locants(mol, ring_system, tuple(attach))
    if not locants:
        return ''
    prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, carboxyl_atoms)
    if prefix is None:
        return ''
    k = len(attach)
    locs = sorted(locants[a] for a in attach)
    if k == 1 and _is_isolated_benzene_atom(mol, attach[0]):
        name = f"{prefix}benzoate"
    else:
        stem = _p74_bare_ring_stem(mol, ring_system, attach[0])
        if not stem:
            return ''
        from ..assembly.naming_utils import get_suffix_multiplier_prefix
        try:
            mult = get_suffix_multiplier_prefix(k, 'carboxylate') if k > 1 else ''
        except ValueError:
            return ''
        name = (f"{_join_prefix_stem(prefix, stem)}-{','.join(str(x) for x in locs)}"
                f"-{mult}carboxylate")
    return name if _anion_name_keeps_hydron_sites(mol, name) else ''
