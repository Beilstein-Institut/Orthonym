"""Shared helpers of the OPSIN-ambiguity tests: name a structure at a tier, read OPSIN's
``APPEARS_AMBIGUOUS`` warnings for the name, and check the round trip by full InChIKey.

A test that names a structure carries ``@pytest.mark.opsin_gate`` (``tests/conftest.py``
switches the OPSIN validity gate off otherwise, and a name the gate would refuse is shipped)."""
from functools import lru_cache
from typing import Optional

from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_bridge import opsin_stdout, opsin_warnings


@lru_cache(maxsize=None)
def _engine(tier: str) -> Orthonym:
    if tier == "best-effort":   # the flags of the paper's best-effort worker
        return Orthonym(general_fallback=True, general_fallback_unverified=True,
                        allow_aromatic_general=True)
    return Orthonym()


def name_row(smiles: str, tier: str = "best-effort") -> dict:
    return _engine(tier).name_tiered(smiles)


def ambiguity_messages(name: str) -> list:
    return [m for t, m in opsin_warnings(name, allow_radicals=True) if t == "APPEARS_AMBIGUOUS"]


def round_trips(smiles: str, name: str) -> bool:
    text, served = opsin_stdout(name, allow_radicals=True)
    assert served, "in-process OPSIN not available"
    out = text.strip()
    a = Chem.MolFromSmiles(smiles)
    b = Chem.MolFromSmiles(out) if out else None
    return bool(a and b and Chem.MolToInchiKey(a) == Chem.MolToInchiKey(b))


def assert_clean(smiles: str, tier: str = "best-effort", must_contain: Optional[str] = None) -> dict:
    row = name_row(smiles, tier)
    name = row["name"]
    assert row["tier"] not in (None, "abstain"), row
    assert not ambiguity_messages(name), (name, ambiguity_messages(name))
    assert round_trips(smiles, name), name
    if must_contain is not None:
        assert must_contain in name, name
    return row
