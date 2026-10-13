"""The pyrrole-type atom of a monocyclic mancude parent."""
from rdkit import Chem


def is_pyrrole_type(atom) -> bool:
    """True for a ring nitrogen or phosphorus that holds three single bonds, whatever the third
    bond carries: the N-H, or the N-substituent of 1,2-dimethyl-4,5-dihydro-1H-imidazole. It is
    the atom the mancude parent carries its indicated hydrogen on, the Blue Book,
    "imidazole (1H-isomer shown; the PIN is 1H-imidazole)"; '2-hydrazinyl-4,5-dihydro-1H-imidazole
    (PIN)',:19389). Taking only the N-H left a molecule with no N-H to the order of its atoms,
    which put the hydrogen on the =N- and sent one drawing of the ring to the Hantzsch-Widman name
    ('1,2-dimethyl-4,5-dihydro-1H-1,3-diazole') and another to imidazole."""
    if atom.GetTotalNumHs() >= 1:
        return True
    return (atom.GetTotalDegree() >= 3 and atom.GetFormalCharge() == 0
            and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()))
