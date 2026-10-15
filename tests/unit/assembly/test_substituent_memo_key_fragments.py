"""``substituent_enumerator._substituent_memo_key`` and a molecule that has more than one fragment.

The key roots the fragment SMILES at the attachment atom. RDKit refuses a rooted fragment SMILES of
a molecule with more than one fragment ("rootedAtAtom can only be used with molecules that have a
single fragment") and logs the refusal on every call. The key function asks the question itself
and returns ``None`` (no caching, the same outcome the raised error had), so nothing reaches RDKit
and nothing is logged. A fragment that is not rooted (the attachment atom is outside it) and a
molecule with one fragment are keyed as before.
"""
import pytest
from rdkit import Chem, RDLogger, rdBase

from orthonym.assembly.substituent_enumerator import _substituent_memo_key

pytestmark = pytest.mark.unit


def _key(smiles, frag_atoms, attach_idx):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return _substituent_memo_key(mol, frag_atoms, attach_idx, False, False)


def test_a_rooted_fragment_of_a_multi_fragment_molecule_is_not_keyed_and_nothing_is_logged(capfd):
    status = rdBase.LogStatus()
    RDLogger.EnableLog("rdApp.*")
    try:
        # the control: RDKit's own refusal is logged
        mol = Chem.MolFromSmiles("CCO.[Na+]")
        with pytest.raises(RuntimeError):
            Chem.MolFragmentToSmiles(mol, atomsToUse=[0, 1, 2], rootedAtAtom=0)
        assert "rootedAtAtom" in capfd.readouterr().err
        for smiles, frags, attach in (("CCO.[Na+]", [0, 1, 2], 0),
                                      ("CCOC(=O)Cc1ccc(cc1)[CH2-].[Zn+]Br", [0, 1, 2, 3, 4], 2),
                                      ("[Cl-].[Zn+]c1ccccc1", [1, 2, 3, 4, 5, 6, 7], 1)):
            assert _key(smiles, frags, attach) is None
        assert capfd.readouterr().err == ""
    finally:
        for line in status.splitlines():
            channel, _, state = line.partition(":")
            (RDLogger.EnableLog if state == "enabled" else RDLogger.DisableLog)(channel)


def test_a_single_fragment_molecule_and_an_unrooted_fragment_are_keyed():
    key = _key("CCO", [0, 1, 2], 0)
    assert key is not None and key[1] is True                          # rooted: attach_in_frag
    assert _key("CCOC(=O)C", [0, 1, 2], 0) is not None
    unrooted = _key("CCO.[Na+]", [1, 2], 0)                            # the attachment atom is outside the fragment
    assert unrooted is not None and unrooted[1] is False
