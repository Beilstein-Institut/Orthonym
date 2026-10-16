"""Spelling check (validation/spelling/checks_pseudohalide.py): 'cyanooxy', 'cyanosulfanyl' and
'cyanoselanyl' for -O-CN, -S-CN and -Se-CN, where 'cyanato', 'thiocyanato' and 'selenocyanato' are the
preferred prefixes (the Blue Book, "Cyanic acid"; '3-(thiocyanato)propanoic acid (PIN)':31013).

Precision 1.0 by measurement (2026-10-16): 0 hits on the 3,243 Blue Book PIN names, on the 1,969 gold and
pack PINs, and on 10,384 distinct (structure, name) pairs the engine emitted for the 4,000-row sample and the
7,243-row screen at the PIN and best-effort tiers.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules
from orthonym.validation.spelling.checks_pseudohalide import pseudohalide_prefix_check

pytestmark = pytest.mark.unit

BAD = [
    ("OC(=O)CC(CSC#N)CC([N+](=O)[O-])C", "3-[(cyanosulfanyl)methyl]-5-nitrohexanoic acid", "thiocyanato"),
    ("OCCOC#N", "2-(cyanooxy)ethan-1-ol", "cyanato"),
    ("CC[Se]C#N", "(cyanoselanyl)ethane", "selenocyanato"),
]


@pytest.mark.parametrize("smiles,name,preferred", BAD)
def test_the_non_preferred_word_is_flagged(smiles, name, preferred):
    failure = pseudohalide_prefix_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-65.2.2" and preferred in failure.detail


@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)CCSC#N", "3-(thiocyanato)propanoic acid"),          # the Blue Book PIN
    ("OCCOC#N", "2-(cyanato)ethan-1-ol"),
    ("N#CCSC", "(methylsulfanyl)acetonitrile"),                 # a nitrile beside S, not S-C#N
    ("CSC", "(cyanosulfanyl)methane"),                          # the word without the structure
    ("N#CSSC", "(cyanodisulfanyl)methane"),                     # NC-SS-: 'cyanodisulfanyl' is preferred
])
def test_preferred_spellings_and_other_structures_pass(smiles, name):
    assert pseudohalide_prefix_check(Chem.MolFromSmiles(smiles), name) is None


def test_the_check_is_registered():
    assert "P-65.2.2" in registered_rules()
    mol = Chem.MolFromSmiles(BAD[0][0])
    assert [f.rule for f in check_pin_spelling(mol, BAD[0][1])] == ["P-65.2.2"]
