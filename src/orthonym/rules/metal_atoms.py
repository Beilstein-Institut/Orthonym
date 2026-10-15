"""Names of the atom, the ions and the isotopes of a metal, and of the hydride ion beside a metal.

One home for the element names of the metals. A metal atom takes its element name; a metal cation
adds its charge number in parentheses, the notation of 'pentaammine(ethanido)osmium(1+) chloride'
, the Blue Book); a nuclide goes in parentheses before the element name, "enclosed in
parentheses... before the part of the compound that is isotopically substituted",
:43718). Two callers read the names:

* the adduct of ``rules/adducts.py``, which cites a one-atom metal component by them at the
  best-effort tier (``metal_atom_name``): "preferred IUPAC names cannot be assigned to mixed adducts"
  (:4667), so such a name is never a PIN;
* the class ``METAL_ATOM`` of the dispatch table, for a molecule that is one atom of a metal
  admitted to the mechanism ``ELEMENT`` of ``data/metal_scope.py`` (``admitted_metal_atom_name``).
  The name is a verified systematic name and no PIN: PREFERRED IUPAC NAMES (:2056) labels
  "the names of compounds whose parent hydride contains at least one of the following elements: B,
  Si,... At, and that also contain at least one carbon atom in their structure", and zinc is none
  of them (``rules/metal_labels.is_metal_atom_or_ion_name_without_pin`` sets the label).

``inorganic_anion_word`` gives the word of a one-atom or oxo anion in a salt: the table of retained
inorganic anions, and 'hydride' for the hydride ion beside metals admitted to ``CARBANION_SALTS``.
``ammine_component_names`` gives the one-atom component 'ammonia' of an adduct beside metals admitted
to ``AMMINE``.
"""
from typing import Dict, Optional

from ..data.ion_retained_names import INORGANIC_ANIONS
from ..data.metal_scope import AMMINE, CARBANION_SALTS, ELEMENT, has_mechanism

# The monoatomic metal components of a disconnected depiction -- 'CC(N)C(=O)O.CC(N)C(=O)O.[Ni]', six
# '[C-]#N' with '[Fe+2]'. "Mixed organic - inorganic adducts" (the Blue Book):
# "organic components in order as described in, inorganic components in order as described
# in Ref 12"; and (:4667) "preferred IUPAC names cannot be assigned to mixed adducts because
# preferred IUPAC names have not yet been determined for inorganic components". (:39735)
# notes no PIN for the Group 1-12 metals either. Such a name is therefore never a PIN.
#
# A metal atom takes its element name; a metal cation adds its charge number in parentheses, the
# notation of 'pentaammine(ethanido)osmium(1+) chloride',:39793). The element set is
# perception.metals.METAL_ELEMENT_SYMBOLS; every spelling below was read back by OPSIN 2.9.0 to the
# bare atom and to its 1+ and 2+ ions (all 94).
METAL_ELEMENT_NAMES: Dict[str, str] = {
    'Ac': 'actinium', 'Ag': 'silver', 'Al': 'aluminium', 'Am': 'americium',
    'As': 'arsenic', 'Au': 'gold', 'B': 'boron', 'Ba': 'barium',
    'Be': 'beryllium', 'Bh': 'bohrium', 'Bi': 'bismuth', 'Bk': 'berkelium',
    'Ca': 'calcium', 'Cd': 'cadmium', 'Ce': 'cerium', 'Cf': 'californium',
    'Cm': 'curium', 'Cn': 'copernicium', 'Co': 'cobalt', 'Cr': 'chromium',
    'Cs': 'caesium', 'Cu': 'copper', 'Db': 'dubnium', 'Ds': 'darmstadtium',
    'Dy': 'dysprosium', 'Er': 'erbium', 'Es': 'einsteinium', 'Eu': 'europium',
    'Fe': 'iron', 'Fm': 'fermium', 'Fr': 'francium', 'Ga': 'gallium',
    'Gd': 'gadolinium', 'Ge': 'germanium', 'Hf': 'hafnium', 'Hg': 'mercury',
    'Ho': 'holmium', 'Hs': 'hassium', 'In': 'indium', 'Ir': 'iridium',
    'K': 'potassium', 'La': 'lanthanum', 'Li': 'lithium', 'Lr': 'lawrencium',
    'Lu': 'lutetium', 'Md': 'mendelevium', 'Mg': 'magnesium', 'Mn': 'manganese',
    'Mo': 'molybdenum', 'Mt': 'meitnerium', 'Na': 'sodium', 'Nb': 'niobium',
    'Nd': 'neodymium', 'Ni': 'nickel', 'No': 'nobelium', 'Np': 'neptunium',
    'Os': 'osmium', 'Pa': 'protactinium', 'Pb': 'lead', 'Pd': 'palladium',
    'Pm': 'promethium', 'Po': 'polonium', 'Pr': 'praseodymium', 'Pt': 'platinum',
    'Pu': 'plutonium', 'Ra': 'radium', 'Rb': 'rubidium', 'Re': 'rhenium',
    'Rf': 'rutherfordium', 'Rg': 'roentgenium', 'Rh': 'rhodium', 'Ru': 'ruthenium',
    'Sb': 'antimony', 'Sc': 'scandium', 'Sg': 'seaborgium', 'Si': 'silicon',
    'Sm': 'samarium', 'Sn': 'tin', 'Sr': 'strontium', 'Ta': 'tantalum',
    'Tb': 'terbium', 'Tc': 'technetium', 'Te': 'tellurium', 'Th': 'thorium',
    'Ti': 'titanium', 'Tl': 'thallium', 'Tm': 'thulium', 'U': 'uranium',
    'V': 'vanadium', 'W': 'tungsten', 'Y': 'yttrium', 'Yb': 'ytterbium',
    'Zn': 'zinc', 'Zr': 'zirconium',
}

#: the canonical SMILES of the hydride ion, the fragment ``inorganic_anion_word`` cites as 'hydride'
HYDRIDE_ION = '[H-]'

#: the neutral one-atom components that an adduct of an ammine complex cites besides the table of
#: ``rules/adducts.SINGLE_ATOM_COMPONENT_NAMES``, by the canonical SMILES of the fragment.
#: Retained names (the Blue Book): "The common names water, ammonia,... are used in these
#: recommendations"; the table of the other components cites 'water' and 'hydrogen chloride' alike.
AMMINE_COMPONENT_NAMES: Dict[str, str] = {'N': 'ammonia'}


def metal_atom_name(frag_mol) -> Optional[str]:
    """The name of a one-atom metal component, or None.

    'nickel' for [Ni], 'iron(2+)' for [Fe+2], '(99Tc)technetium' for [99Tc]
    (the nuclide descriptor of in front of the element name). None for
    a non-metal, a metal carrying hydrogen ([NaH] is a hydride, not a metal atom)
    or a negative metal ion (an '-ide' of a metal is not built here).
    """
    if frag_mol.GetNumAtoms() != 1:
        return None
    atom = frag_mol.GetAtomWithIdx(0)
    name = METAL_ELEMENT_NAMES.get(atom.GetSymbol())
    if name is None or atom.GetTotalNumHs() != 0:
        return None
    charge = atom.GetFormalCharge()
    if charge < 0:
        return None
    if charge > 0:
        name = f"{name}({charge}+)"
    if atom.GetIsotope():
        name = f"({atom.GetIsotope()}{atom.GetSymbol()}){name}"
    return name


def admitted_metal_atom_name(mol) -> Optional[str]:
    """The name of ``mol`` when it is one atom of a metal admitted to ``ELEMENT`` in
    ``data/metal_scope.py`` ('zinc', 'zinc(2+)', '(65Zn)zinc'), else None.

    A pure function of the structure: the predicate and the handler of the dispatch class
    ``METAL_ATOM`` both call it. The name is read back by OPSIN 2.9.0 to the standard InChIKey of
    the atom or ion (the gate of the namer); this function builds it only."""
    if mol is None or mol.GetNumAtoms() != 1:
        return None
    if not has_mechanism(mol.GetAtomWithIdx(0).GetSymbol(), ELEMENT):
        return None
    return metal_atom_name(mol)


def inorganic_anion_word(smiles: str, cation_frags) -> Optional[str]:
    """The word of the anion fragment ``smiles`` (canonical SMILES) in a salt, or None.

    The table of retained inorganic anions (``data/ion_retained_names.INORGANIC_ANIONS``: chloride,
    sulfate,...), and 'hydride' for the hydride ion when every cation of the salt is one atom of a
    metal admitted to ``CARBANION_SALTS`` ('zinc dihydride'): (the Blue Book),
    "binary names formed by citing the name of the cation followed by that of the anion". A hydride
    beside any other cation has no word here, so the salts of the other metals keep their names."""
    word = INORGANIC_ANIONS.get(smiles)
    if word is not None:
        return word
    if smiles == HYDRIDE_ION and cation_frags and all(
            frag['mol'].GetNumAtoms() == 1
            and has_mechanism(frag['mol'].GetAtomWithIdx(0).GetSymbol(), CARBANION_SALTS)
            for frag in cation_frags):
        return 'hydride'
    return None


def ammine_component_names(mol) -> Dict[str, str]:
    """The one-atom components that ``mol`` may cite beside its metals, besides the table of
    ``rules/adducts.SINGLE_ATOM_COMPONENT_NAMES``: ``AMMINE_COMPONENT_NAMES`` when ``mol`` holds a true
    metal and every true metal of it is admitted to ``AMMINE`` in ``data/metal_scope.py``, else ``{}``.

    A bare nitrogen is the ammonia of an ammine complex only beside a metal that the table admits;
    beside any other metal, and in an assembly with no metal, it stays a fragment that no producer
    names, as before. The guard reads the whole assembly: one metal outside the table keeps the rest
    of the assembly outside it."""
    from ..perception.metals import _TRUE_METAL_SYMBOLS_FOR_VETO
    metals = {a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() in _TRUE_METAL_SYMBOLS_FOR_VETO}
    if metals and all(has_mechanism(symbol, AMMINE) for symbol in metals):
        return dict(AMMINE_COMPONENT_NAMES)
    return {}
