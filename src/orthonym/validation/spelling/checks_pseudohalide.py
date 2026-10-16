"""Spelling check: the prefixes of -O-CN, -S-CN and -Se-CN (registered with
:func:`orthonym.validation.pin_spelling.register`).

 "Cyanic acid" (the Blue Book),:30981: "Preferred prefixes derived from cyanic acid are
'cyano' for -CN and 'cyanato' for -O-CN, 'thiocyanato' for -S-CN, 'selenocyanato' for -Se-CN";
'3-(thiocyanato)propanoic acid (PIN)' (:31013). 'cyanooxy', 'cyanosulfanyl' and 'cyanoselanyl' (a
cyano group on an O, S or Se substituent) are the substitutive spellings of the same groups and are
not preferred. The name side is the three words; the structure side is a divalent O, S or Se
bonded to a nitrile carbon. No naming code, no OPSIN.
"""
from __future__ import annotations

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register

#: the non-preferred word -> (the element it is written for, the preferred prefix)
_WORDS = {
    'cyanooxy': ('O', 'cyanato'),
    'cyanosulfanyl': ('S', 'thiocyanato'),
    'cyanoselanyl': ('Se', 'selenocyanato'),
}


def _has_chalcogen_cyanide(mol, element: str) -> bool:
    """An atom of ``element`` with two heavy-atom neighbours, one of them a nitrile carbon."""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != element or atom.GetDegree() != 2:
            continue
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if (bond.GetBondType() == Chem.BondType.SINGLE and other.GetSymbol() == 'C'
                    and any(b.GetBondType() == Chem.BondType.TRIPLE
                            and b.GetOtherAtom(other).GetSymbol() == 'N' for b in other.GetBonds())):
                return True
    return False


@register('P-65.2.2')
def pseudohalide_prefix_check(mol, name):
    """ (:30981): the name writes -O-CN / -S-CN / -Se-CN as 'cyanooxy' / 'cyanosulfanyl' /
    'cyanoselanyl' where 'cyanato' / 'thiocyanato' / 'selenocyanato' are the preferred prefixes."""
    for word, (element, preferred) in _WORDS.items():
        if word in name and _has_chalcogen_cyanide(mol, element):
            return SpellingFailure('P-65.2.2', f"'{word}' written for the {element}-C#N group; "
                                               f"the preferred prefix is '{preferred}'")
    return None
