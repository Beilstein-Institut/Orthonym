"""The ``producer`` key is the only thing the record adds to a row.

``Orthonym.name_tiered`` opens a per-call record of which code built the name (``metrics.producer_trail``)
and reads it for the ``producer`` key. Nothing the record does may change a name, a tier, a label or any
other field of the row, or a decision of the substituent-fragment memo. Each case is one molecule of
``tests/fixtures/producer_identity_rows.jsonl`` (SMILES only: the 22 hand-picked rows that cover every
family of producer, then every fifth row of the dev set):

* ``test_row_is_identical_with_the_record_off``: the row at both tiers equals the row when no record is
  ever opened (``open_trail`` returns ``None``, so every ``noted``, ``carried`` and ``suspended`` call is
  a no-op), in every field but ``producer``, which is ``None`` when no record is open;
* ``test_memo_decisions_are_equal``: the number of substituent fragments named from scratch is the same
  with and without the record, so the memo stores and replays the same calls (``test_the_counter_sees_the_memo``
  shows that the count moves when the memo does; a record that wrote a provenance variable is caught by
  ``tests/unit/metrics/test_producer_trail_static.py``, because the calls that run seams are not storable
  in the memo today either way);
* ``test_repeat_is_stable``: three consecutive calls in one process give the same row, ``producer``
  included.

The cases of the first 22 rows (and of the memo and stability subsets, below) run in the public
pull-request run; the others are ``slow``, as are the cobalamins, which take about 20 s at the
best-effort tier.
"""
import json
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.assembly import memo, substituent_naming
from orthonym.cli import _emit_tier_flags
from orthonym.metrics import producer_trail

# The suite switches the OPSIN validity gate off unless a test asks for it; the rows of the shipped
# configuration are the ones with the gate on.
pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "producer_identity_rows.jsonl"
ROWS = [json.loads(line)["smiles"] for line in FIXTURE.read_text().splitlines() if line.strip()]
TIERS = ("pin", "best-effort")

PUBLIC_IDENTITY = 22        # the hand-picked rows, which cover every family of producer
PUBLIC_MEMO = 22            # of the rows 0, 3, 6,...
PUBLIC_STABILITY = 40       # of the rows 0, 4, 8,...
HEAVY_ATOMS = 80            # a molecule this large (the cobalamins) takes about 20 s at best-effort


def _cases(rows_with_index, public_below):
    """One pytest case per row: public when its index is below ``public_below`` and the molecule is
    small, ``slow`` otherwise."""
    cases = []
    for k, smiles in rows_with_index:
        small = Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() <= HEAVY_ATOMS
        marks = () if k < public_below and small else (pytest.mark.slow,)
        cases.append(pytest.param(smiles, id=f"row{k}", marks=marks))
    return cases


def _row(smiles, tier):
    flags = {} if tier == "pin" else _emit_tier_flags(tier)
    return Orthonym(style="pin", **flags).name_tiered(smiles)


def _no_record(monkeypatch):
    # _name_tiered_impl imports the three functions at call time, so the patch is seen
    monkeypatch.setattr(producer_trail, "open_trail", lambda smiles: None)


@pytest.mark.parametrize("smiles", _cases(list(enumerate(ROWS)), PUBLIC_IDENTITY))
def test_row_is_identical_with_the_record_off(monkeypatch, smiles):
    with_record = {tier: _row(smiles, tier) for tier in TIERS}
    _no_record(monkeypatch)
    without_record = {tier: _row(smiles, tier) for tier in TIERS}
    for tier in TIERS:
        on, off = with_record[tier], without_record[tier]
        assert set(on) == set(off), (tier, set(on) ^ set(off))
        assert off["producer"] is None, (tier, off)
        assert {k: v for k, v in on.items() if k != "producer"} == \
            {k: v for k, v in off.items() if k != "producer"}, (tier, on, off)


class _Counter:
    """Calls of ``_name_substituent_fragment_uncached``: the fragments that the memo did not answer."""

    def __init__(self, monkeypatch):
        self.n = 0
        real = substituent_naming._name_substituent_fragment_uncached

        def counting(*a, **k):
            self.n += 1
            return real(*a, **k)

        monkeypatch.setattr(substituent_naming, "_name_substituent_fragment_uncached", counting)

    def count(self, smiles, tier):
        self.n = 0
        _row(smiles, tier)
        return self.n


@pytest.mark.parametrize("smiles", _cases(list(enumerate(ROWS))[::3], PUBLIC_MEMO))
def test_memo_decisions_are_equal(monkeypatch, smiles):
    # The record writes no provenance variable (tests/unit/metrics/test_producer_trail_static.py), so the
    # fragment memo stores and replays the same calls. A seam that read or wrote a provenance variable
    # inside a fragment naming would make a storable call unstorable and raise this count.
    counter = _Counter(monkeypatch)
    for tier in TIERS:
        with_record = counter.count(smiles, tier)
        with monkeypatch.context() as off:
            _no_record(off)
            without_record = counter.count(smiles, tier)
        assert with_record == without_record, (tier, with_record, without_record)


def test_the_counter_sees_the_memo(monkeypatch):
    # the counter is not vacuous: with the memo off, rows that name a fragment twice count more calls
    counter = _Counter(monkeypatch)
    on = [counter.count(ROWS[k], "pin") for k in (24, 36, 45)]
    monkeypatch.setattr(memo, "_MODE", "off")
    off = [counter.count(ROWS[k], "pin") for k in (24, 36, 45)]
    assert any(o > n for o, n in zip(off, on)), (on, off)


@pytest.mark.parametrize("smiles", _cases(list(enumerate(ROWS))[::4], PUBLIC_STABILITY))
def test_repeat_is_stable(smiles):
    for tier in TIERS:
        first = _row(smiles, tier)
        for _ in range(2):
            assert _row(smiles, tier) == first, tier
