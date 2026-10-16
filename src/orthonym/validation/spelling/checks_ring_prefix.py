"""Spelling check of a prefix the PIN tier's promotion re-run assembles (PB lane R; registered with
:func:`orthonym.validation.pin_spelling.register`).

Read lexically (:mod:`.lexer`); no naming code, no OPSIN.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register
from .lexer import UnbalancedMarksError, enclosures, normalise

_WORD_END = re.compile(r"[^a-z]|$")


@register('P-14.3.4')
def methyl_prefix_has_no_locant(mol, name):
    """ (a) (the Blue Book) "The locant '1' is omitted in substituted mononuclear parent
    hydrides" ('chloromethane (PIN)'): a substituent group of the parent hydride methane never cites a
    locant. A prefix that reads ``1-[X]methyl`` -- the name of ``1-[X]methanamine`` cut down to a prefix --
    carries a locant the rule omits ('N-{[1-(...)azetidin-3-yl]methyl}', not 'N-{1-[1-(...)azetidin-3-yl]methyl}').
    ``1-[X]methanamine`` itself keeps the locant (it tells the carbon from the nitrogen,,:2869)
    and is not read here."""
    s = normalise(name)
    try:
        encs = enclosures(s)
    except UnbalancedMarksError:
        return None
    for e in encs:
        # a locant '1' directly before an enclosed substituent that ends the word as 'methyl'
        # (not an isotope descriptor: '1-(13C)methyl(2-13C)benzene', the Blue Book style)
        if e.kind != 'subst' or s[max(0, e.start - 2):e.start] != '1-':
            continue
        before = s[max(0, e.start - 3):e.start - 2]
        if before and before.isalnum():
            continue                       # '11-', '21-': another locant
        after = s[e.end + 1:]
        if after.startswith('methyl') and _WORD_END.match(after[len('methyl'):]):
            return SpellingFailure(
                'P-14.3.4', f"the prefix '1-{s[e.start:e.end + 1]}methyl' cites the locant 1 of methyl")
    return None


_OXO_WORD = re.compile(r"oxo")


@register('P-14.1.3')
def ring_chalcogen_oxide_has_lambda(mol, name):
    """ "Nonstandard bonding numbers" (the Blue Book): a neutral skeletal atom of nonstandard
    bonding number takes the symbol 'λn' cited with its locant. A ring S, Se or Te that carries an
    exocyclic =O has bonding number 4 or 6 and its Blue Book name cites it ('1λ6-thiolane-1,1-dione',
    '1λ6,2-thiazinane-1,1-dione (PIN)':33260 ff.; '1λ4,2-thiazinan-1-one (PIN)' against the non-PIN
    '1,2-thiazinane 1-oxide':33290), so a name that writes the oxygens as an 'oxo' prefix and no 'λ' at all
    ('1,1-dioxothiolan-3-ol') does not spell the PIN. The functional class names ('thiolane 1,1-dioxide')
    are not read here: the gold rows W2C-B4-SOXIDE-01..03 assert them."""
    if 'λ' in name or 'lambda' in name or _OXO_WORD.search(name) is None:
        return None
    ring_info = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() in ('S', 'Se', 'Te') and atom.GetFormalCharge() == 0
                and ring_info.NumAtomRings(atom.GetIdx()) > 0
                and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(atom).GetSymbol() == 'O'
                        for b in atom.GetBonds())):
            return SpellingFailure(
                'P-14.1.3', "a ring atom of nonstandard bonding number is written without 'λ'")
    return None
