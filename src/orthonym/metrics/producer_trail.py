"""Which code built the name: a per-call record kept outside the provenance variables.

``Orthonym.name_tiered`` opens a record for the call (``open_trail``); the producers'
call sites report the string a producer returned (``noted``) and the steps that
rewrite a string pass the label on (``carried``); the row reads the entry of the
name that ships (``lookup``). The record is bound to the string, so a candidate that
is rejected leaves an entry for a string that never ships and nothing needs rolling
back. It is not a provenance variable: it never calls ``_touch``, is not a key of
``get_provenance`` and is not restored by ``restore_provenance``, so the replay
memo, the substituent-fragment memo and the hang budgets cannot see it.

Every function here returns at once when no record is open (``name_compound`` and
the nested namings), never raises, and ``noted`` / ``carried`` return their argument
unchanged.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Dict, Optional

_TRAIL: contextvars.ContextVar = contextvars.ContextVar("orthonym_producer_trail", default=None)


class _Trail:
    __slots__ = ("n_heavy", "by_name")

    def __init__(self, n_heavy: int) -> None:
        self.n_heavy = n_heavy
        self.by_name: Dict[str, str] = {}


def open_trail(smiles) -> contextvars.Token:
    """Open the record of one ``name_tiered`` call; pair with:func:`close_trail`."""
    n_heavy = -1
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        n_heavy = mol.GetNumHeavyAtoms() if mol is not None else -1
    except Exception:  # noqa: BLE001 - no count, no whole-molecule seam is recorded
        pass
    return _TRAIL.set(_Trail(n_heavy))


def close_trail(token: contextvars.Token) -> None:
    """Close the record that:func:`open_trail` opened; the record that was open before
    it, if any, is current again."""
    try:
        _TRAIL.reset(token)
    except Exception:  # noqa: BLE001
        _TRAIL.set(None)


@contextmanager
def suspended():
    """No record inside the block (the strict twin, which names for comparison only)."""
    token = _TRAIL.set(None)
    try:
        yield
    finally:
        _TRAIL.reset(token)


def _name_of(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)            # NamingResult, GeneralEngineResult
    return name if isinstance(name, str) else None


def note(producer: str, name, mol=None) -> None:
    """Bind ``name`` to ``producer`` in the open record. ``mol`` is the molecule the
    producer named; a name built for a part of the input (a different heavy-atom count)
    is not the input's name and is not recorded."""
    trail = _TRAIL.get()
    if trail is None or not isinstance(name, str) or not name:
        return
    try:
        if mol is not None and mol.GetNumHeavyAtoms() != trail.n_heavy:
            return                                  # a part of the molecule, not the input
        from ..errors import is_failure_name
        if is_failure_name(name):
            return
        trail.by_name.setdefault(name, producer)    # the first producer to return it keeps it
    except Exception:  # noqa: BLE001
        return


def noted(producer: str, value, mol=None):
    """Record that ``producer`` returned ``value`` (a name, or an object with ``.name``) for
    the input; return ``value`` itself."""
    if _TRAIL.get() is not None:
        try:
            note(producer, _name_of(value), mol)
        except Exception:  # noqa: BLE001 - a value whose.name raises is not recorded
            pass
    return value


def carried(name_in, name_out):
    """A step turned ``name_in`` into ``name_out``: the label follows. Returns ``name_out``."""
    trail = _TRAIL.get()
    if trail is not None and isinstance(name_in, str) and isinstance(name_out, str) \
            and name_out != name_in:
        try:
            label = trail.by_name.get(name_in)
            if label is not None:
                from ..errors import is_failure_name
                if not is_failure_name(name_out):
                    trail.by_name.setdefault(name_out, label)
        except Exception:  # noqa: BLE001
            pass
    return name_out


def lookup(name) -> Optional[str]:
    """The producer recorded for ``name`` in the open record, or ``None``."""
    trail = _TRAIL.get()
    if trail is None or not isinstance(name, str):
        return None
    return trail.by_name.get(name)
