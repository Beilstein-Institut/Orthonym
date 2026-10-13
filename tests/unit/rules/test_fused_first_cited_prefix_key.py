""" (g): the prefix cited first in the name takes the lowest locant, and the
prefix cited first is the first one in alphanumerical order as the assembler cites
them (``alpha_sort_key``), not the first one in the order of the plain lowercased
string.

``_first_cited_prefix_key`` used to order the prefix names by ``n.lower``, while the
assembler (``_assemble_fused_heterocycle_name``) cites them in ``alpha_sort_key`` order
: ``tert-`` and ``sec-`` are italicised, a leading locant is not part of the
name, so ``tert-butyl`` sorts under ``b`` and ``4-methylphenyl`` under ``m``). The two
disagreed whenever a name began with ``tert-``, ``sec-``, a locant or ``bis``, and a
tie between two numberings then took the lowest locant for the wrong prefix.

Governing text, the Blue Book, NUMBERING (the criteria list at line 3219):
"(g) lowest locants for the substituent cited first as a prefix in the name".
"""
import pytest

from orthonym import Orthonym
from orthonym.rules.fused_rings import _first_cited_prefix_key, _fused_locant_num
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit


def _loc(*numbers):
    return tuple(_fused_locant_num(n) for n in numbers)


def test_prefix_names_are_ordered_as_the_assembler_cites_them():
    # 'ethyl' (e) is cited before '4-methylphenyl' (m: the locant is not part of the
    # name): the key lists the locants of ethyl first. The lowercased-string order
    # put '4-methylphenyl' first ('4' sorts before 'e').
    subs = {'c_substituents': {'ethyl': [2]},
            'other': [{'name': '4-methylphenyl', 'locant': 3}]}
    assert _first_cited_prefix_key(subs) == (_loc(2), _loc(3))
    swapped = {'c_substituents': {'ethyl': [3]},
               'other': [{'name': '4-methylphenyl', 'locant': 2}]}
    assert _first_cited_prefix_key(swapped) == (_loc(3), _loc(2))


def test_a_tert_prefix_is_cited_under_its_stem():
    # 'tert-butyl' is cited under 'b', before 'ethyl'; the lowercased-string order
    # put 'ethyl' first because 't' sorts after 'e'.
    subs = {'c_substituents': {'tert-butyl': [2], 'ethyl': [3]}}
    assert _first_cited_prefix_key(subs) == (_loc(2), _loc(3))
    swapped = {'c_substituents': {'tert-butyl': [3], 'ethyl': [2]}}
    assert _first_cited_prefix_key(swapped) == (_loc(3), _loc(2))


def test_the_locants_of_one_prefix_stay_sorted_and_together():
    subs = {'c_substituents': {'methyl': [7, 2], 'tert-butyl': [5]}}
    assert _first_cited_prefix_key(subs) == (_loc(5), _loc(2, 7))


def test_no_substituents_give_an_empty_key():
    assert _first_cited_prefix_key(None) == ()
    assert _first_cited_prefix_key({}) == ()


# 2-tert-butyl-3-ethyl-5,6,7,8-tetrahydroquinoxaline, constructed from its name (OPSIN's
# reading, recorded here). The ring system has a twofold symmetry: 2/3 and 5/8, 6/7 swap.
# Hydro locants (5,6,7,8) and prefix locants (2,3) tie, so (g) decides: 'tert-butyl' is
# cited first (under 'b') and takes locant 2. The drawing is the same molecule whichever
# way round the two substituents are written.
_TETRAHYDROQUINOXALINES = (
    "C(C)(C)(C)C1=NC=2CCCCC2N=C1CC",
    "C(C)(C)(C)C=1C(=NC=2CCCCC2N1)CC",
    "CCC1=NC2=C(CCCC2)N=C1C(C)(C)C",
)


@pytest.mark.parametrize("smiles", _TETRAHYDROQUINOXALINES)
def test_the_prefix_cited_first_takes_the_lowest_locant_of_a_hydro_fused_parent(smiles):
    jar_or_skip("opsin")
    expected = "2-tert-butyl-3-ethyl-5,6,7,8-tetrahydroquinoxaline"
    name = Orthonym().name_tiered(smiles).get("name")
    assert name == expected, name
    assert name_is_rt_exact(name, smiles), name
