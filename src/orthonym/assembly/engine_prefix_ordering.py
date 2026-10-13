"""The ordering name of a prefix is asked the way the general engine names that prefix.

The ring numberings of NUMBERING (the Blue Book) decide a tie by criterion (g), "lowest
locants for the substituent cited first as a prefix in the name" (:3307), over the alphanumerical
key of each prefix on the ring (``substituent_enumerator.name_substituent_for_ordering``). The
general engine cites a prefix with ``name_substituent(..., allow_mancude=allow_aromatic_general)``
while the four propagation flags of the context stay as the clean fall-through resets them
(``namer._try_besteffort_clean_general_fallthrough``), so the prefix it cites is the one the
universal recursion builds, and the key of (g) is the name of that same prefix. Asked of the
cascade alone, which does not name an N-substituted thiocarbamoyl or carbamoyl group, the key was
missing, (g) was skipped, and the canonical ranks decided which prefix of a 1,4-disubstituted
piperazine took locant 1.
"""
import contextlib
import contextvars

#: The naming-session depth of the general-engine run that names its prefixes with ``allow_mancude``
#: (``None``: no engine run is asking).
_ENGINE_DEPTH = contextvars.ContextVar("orthonym_engine_ordering_depth", default=None)


@contextlib.contextmanager
def engine_prefix_ordering(allow_mancude: bool):
    """Mark the body as the general engine naming its prefixes with ``allow_mancude``.

    The mark holds for the naming-session depth of the engine call only: a substituent the engine
    names is named one session deeper, by a namer whose own prefixes are asked the cascade as
    before. ``allow_mancude`` False marks nothing."""
    if not allow_mancude:
        yield
        return
    from .fragment_naming import _session_depth
    token = _ENGINE_DEPTH.set(_session_depth())
    try:
        yield
    finally:
        _ENGINE_DEPTH.reset(token)


def engine_orders_with_mancude() -> bool:
    """True when the current naming call is the engine run ``engine_prefix_ordering`` marked."""
    depth = _ENGINE_DEPTH.get()
    if depth is None:
        return False
    from .fragment_naming import _session_depth
    return _session_depth() == depth
