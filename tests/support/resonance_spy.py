"""Count ResonanceMolSupplier constructions, by the resonance site that asks for one."""
import contextlib
import sys

from rdkit import Chem
from rdkit.Chem import rdchem

#: the functions that own a supplier call: site 1 (namer), site 2 (ions), site 3 (protonation_identity)
SITE_FUNCTIONS = ("_charge_forms", "_ring_cation_hydron_site", "_analogue_inchis")


def _asking_site(frame):
    """The nearest enclosing site function of ``frame``; the name of ``frame`` itself when none encloses it."""
    here = frame
    while here is not None:
        if here.f_code.co_name in SITE_FUNCTIONS:
            return here.f_code.co_name
        here = here.f_back
    return frame.f_code.co_name


@contextlib.contextmanager
def supplier_calls(forbid=False):
    """Yield a list of ``(asking site function, heavy atoms)``, one entry per supplier built in the block.

    The real supplier is still built, unless ``forbid`` is set: then a call is recorded and raises ``RuntimeError``
    instead, so a molecule that RDKit never finishes cannot hang the test that asserts the list is empty. Sites 1
    and 3 reach the supplier through ``rdchem`` and ``Chem`` respectively, so both names are patched."""
    calls = []
    real = rdchem.ResonanceMolSupplier

    def spy(mol, *args, **kwargs):
        calls.append((_asking_site(sys._getframe(1)), mol.GetNumHeavyAtoms()))
        if forbid:
            raise RuntimeError("a resonance supplier must not be built here")
        return real(mol, *args, **kwargs)

    saved = (Chem.ResonanceMolSupplier, rdchem.ResonanceMolSupplier)
    Chem.ResonanceMolSupplier = spy
    rdchem.ResonanceMolSupplier = spy
    try:
        yield calls
    finally:
        Chem.ResonanceMolSupplier, rdchem.ResonanceMolSupplier = saved
