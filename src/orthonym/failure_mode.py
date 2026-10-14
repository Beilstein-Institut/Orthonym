"""What a public naming call returns for a structure it cannot name (GitHub issue 45).

Three modes, chosen with the ``on_failure`` argument of:func:`orthonym.name_compound` and
:meth:`orthonym.Orthonym.name`:

``"empty"`` (the default)
    Return ``""`` and log one warning on the logger ``orthonym`` with the SMILES, the limit
    code and the label (``UNNAMEABLE: unknown organic compound``). The label is not a name.
``"label"``
    Return the label, as the engine did before the option existed.
``"raise"``
    Raise:class:`orthonym.errors.OrthonymLimitError` (the same as ``raise_on_limit=True``).

The choice applies to the caller's own call only, the outermost naming call on the stack
(:func:`is_callers_own_call`). The engine names parts of a molecule through the same public
entries (``name_compound`` and ``Orthonym.name`` inside handlers, producers and rescues) and reads
what they return as a string, so a nested call always gets the label, whatever the argument says.

``"empty"`` keeps the name the ``"label"`` call would return: the call is made as a ``"label"``
call and a result that:func:`orthonym.errors.is_failure_name` calls a failure is blanked. It is
not made through ``raise_on_limit``, because that option also skips the late recoveries and the
trivial fallback of the wider tiers (measured on 159 development molecules at the best-effort
tier: 7 names the plain call gives came back as errors). The limit code is read from what the
call recorded (the default tier's decline, the abstention record), else classified from the
structure (:func:`orthonym.errors.classify_failure_limit`).
"""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Optional

from rdkit import Chem

from . import input_mol as _input_mol
from .errors import (
    OrthonymLimitError,
    classify_failure_limit,
    classify_scope_limit,
    is_failure_name,
    no_verified_pin,
    unsupported_ring_system,
)

MODES = ("empty", "label", "raise")
DEFAULT_MODE = "empty"

#: The logger the unnamed-structure warning goes to. The record carries ``orthonym_smiles``,
#: ``orthonym_code`` and ``orthonym_label``, so a handler can read them without parsing the text.
logger = logging.getLogger("orthonym")


def is_callers_own_call() -> bool:
    """Whether a naming call now being made is the caller's own, outside any naming."""
    from .assembly.fragment_naming import _session_depth, name_scope_depth
    return name_scope_depth() == 0 and _session_depth() == 0


def resolve(on_failure: str, raise_on_limit: bool = False) -> str:
    """The mode of a call: ``on_failure`` checked, ``"raise"`` when ``raise_on_limit`` is set,
    and ``"label"`` in place of ``"empty"`` for a call made inside a naming.

    Raises ValueError for a value that is not one of:data:`MODES`."""
    if on_failure not in MODES:
        raise ValueError(f"on_failure must be one of {', '.join(map(repr, MODES))}; "
                         f"got {on_failure!r}")
    if raise_on_limit:
        return "raise"
    return "label" if on_failure == "empty" and not is_callers_own_call() else on_failure


def _text_of(given) -> str:
    """The SMILES of what a naming call was given (a Mol is named as RDKit's SMILES of it)."""
    if isinstance(given, str):
        return given
    try:
        return _input_mol.smiles_for(given)
    except Exception:  # noqa: BLE001 - only the text of a log line
        return repr(given)


def limit_of(namer, smiles: str, produced: Optional[str]) -> OrthonymLimitError:
    """The limit behind ``produced``, a label (or ``""``) that a naming call returned.

    The default tier's decline of a name that is not a verified PIN (recorded by ``namer``),
    the ring-system refusal (recorded as the abstention detail), else the structure's own
    classification, as for:func:`orthonym.classify_limit`."""
    from .metrics.abstention import peek_abstention
    mol = Chem.MolFromSmiles(smiles)
    scope = classify_scope_limit(mol)
    if scope is not None:
        scope.smiles = smiles
        return scope
    stash = getattr(namer, "_default_tier_row", None)
    if (stash is not None and stash[0] == produced
            and stash[1].get("limit_code") == "NO_VERIFIED_PIN"):
        return no_verified_pin(produced, smiles=smiles)
    if getattr(peek_abstention(), "detail", None) == "UNSUPPORTED_RING_SYSTEM":
        return unsupported_ring_system(smiles)
    return classify_failure_limit(mol, smiles=smiles)


def warn_unnamed(smiles: str, limit: dict) -> None:
    """Log the one warning for a structure that got no name. ``limit`` is a limit as a dict
    (:meth:`OrthonymLimitError.as_dict`)."""
    logger.warning("no name for %s: %s: %s", smiles, limit["code"], limit["message"],
                   extra={"orthonym_smiles": smiles, "orthonym_code": limit["code"],
                          "orthonym_label": limit["message"]})


def blank(namer, given, produced: Optional[str]) -> str:
    """``produced`` when it is a name; else ``""`` with the warning (``"empty"`` mode)."""
    if produced and not is_failure_name(produced):
        return produced
    smiles = _text_of(given)
    warn_unnamed(smiles, limit_of(namer, smiles, produced).as_dict())
    return ""


def settle_confidence(mode: str, record: dict, smiles: str) -> dict:
    """``record``, the dict of ``name_compound(..., include_confidence=True)``, settled for
    ``mode``. A structure that got no name has a ``limit`` in its record (or no name, or a
    label for it): ``"empty"`` sets its ``"name"`` to ``""`` and logs the warning,
    ``"raise"`` raises the limit, ``"label"`` leaves the record as it is."""
    if mode == "label" or not (record.get("limit") is not None
                               or is_failure_name(record.get("name"))):
        return record
    limit = record.get("limit") or {"code": "UNNAMEABLE", "message": "unknown",
                                    "design_note_ref": None}
    if mode == "raise":
        raise OrthonymLimitError(limit["code"], limit["message"],
                                 limit.get("design_note_ref"), smiles)
    warn_unnamed(smiles, limit)
    return {**record, "name": ""}


def failure_mode_scope(fn):
    """The ``on_failure`` argument of ``Orthonym.name``, outside every other scope of it.

    ``fn`` is the whole ``name``. A nested call, and a call with a mode other than
    ``"empty"``, run ``fn`` as they always did (``"raise"`` is ``raise_on_limit=True``);
    the caller's own ``"empty"`` call runs it as a ``"label"`` call and blanks a failure."""
    @functools.wraps(fn)
    def _wrapper(self, *args, on_failure: str = DEFAULT_MODE, **kwargs):
        mode = resolve(on_failure, kwargs.get("raise_on_limit", False))
        kwargs["raise_on_limit"] = mode == "raise"
        if mode != "empty":
            return fn(self, *args, **kwargs)
        self._default_tier_row = None
        produced = fn(self, *args, **kwargs)
        return blank(self, args[0] if args else kwargs.get("smiles"), produced)

    signature = inspect.signature(fn)
    _wrapper.__signature__ = signature.replace(parameters=[
        *signature.parameters.values(),
        inspect.Parameter("on_failure", inspect.Parameter.KEYWORD_ONLY,
                          default=DEFAULT_MODE, annotation=str)])
    return _wrapper
