"""Does the shipped name state every stereo element the input configures?

The provenance flag ``stereo_unexpressed`` of ``Orthonym.name_tiered`` is ``True`` when a stereo
element of the input (a stereocentre, or a double bond drawn with ``/`` and ``\\``) is not stated in
the name. The flag used to be set for one producer only, the general engine under
``ORTHONYM_BE_STRIP_STEREO``, so a name of the strict path or of the composer that left the
configuration out carried ``False``:

    C[N+](C)=C/C=C/C=C/N1CCCC1 -> 1-[5-(dimethylamino)penta-2,4-dien-1-ylidene]pyrrolidin-1-ium
                                     (pin_verified; two configured double bonds in the input, none in the name)

The two double bonds are single bonds in the form of the cation that gives the PIN, so the name
cannot state them; the standard InChIKey does not see a bond that is single in the other form, so
every gate that compares keys passes (``namer._self_consistency_verdict`` accepts on equal full
keys before it counts stereo). The name is right and it is the PIN; the row now says that it
leaves the configuration out (`internal notes`).

The test is the one the gate already has: ``namer._name_omits_input_stereo``, the number of
configured stereo elements of OPSIN's read-back of the name against the input's. The read-back
is ``namer._validity_gate_name_to_smiles``, the oracle's cached parse: the parse the final check
made, so a name it read costs no OPSIN call. A name OPSIN cannot read (lowercase ``r`` and ``s``,
a grammar carve-out) has no read-back and is not flagged: the flag never guesses from the string
of the name (a token count would miss ``L-phenylalanine`` and over-count ``bis{(E)-...}``).

Blue Book (no PIN decision here; the name and its tier are the ones the row had before):
 "Omission of stereodescriptors" (the Blue Book),:44637: the omission of the
descriptors of double bonds is recommended only for three- to seven-membered unsaturated alicyclic
rings, and "From nine-membered rings onwards, stereodescriptors are required."
"Compounds containing more than one double bond" (:47102),:47104: "The stereodescriptors 'E' and
'Z' related to the parent are cited... before the complete name of the compound". The book gives
no PIN for the input's configuration in a form whose double bonds are single bonds; the tier is
not moved (ASSUMED: the user ruling of 2026-10-13, a name that had a PIN keeps it).
"""
from typing import Optional

#: a SMILES configures stereo only with these characters (``@``, ``@@``, ``/``, ``\``)
_STEREO_MARKS = frozenset("/\\@")


def name_states_fewer_stereo(input_smiles: Optional[str], name: Optional[str]) -> bool:
    """True iff ``name`` has an OPSIN read-back that configures FEWER stereo elements than
    ``input_smiles``. False for an input with no stereo mark, a decline, a name OPSIN cannot read and
    a name whose read-back states every element (or more)."""
    if not name or not input_smiles or _STEREO_MARKS.isdisjoint(input_smiles):
        return False
    from ..errors import is_failure_name
    if is_failure_name(name):
        return False
    from ..namer import _name_omits_input_stereo, _validity_gate_name_to_smiles
    read_back = _validity_gate_name_to_smiles(name)
    return bool(read_back) and _name_omits_input_stereo(input_smiles, read_back)
