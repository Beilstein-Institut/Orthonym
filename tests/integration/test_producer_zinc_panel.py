"""Every named row of the zinc panel reports a producer.

The zinc lane added producers (the metal atom class, the other drawing of a metal drawing, the dative
split, the macrocycle and ammine components, the organozinc handlers). They report through the seams
of the outer classes and the recoveries that call them, and ``tests/support/producer_pairs.py::ZINC``
pins one id per kind. This file is the measurement that no row of the panel falls between the seams:
a row the panel names at the best-effort tier has a ``producer``, and a row that is declined has none
(``guide/use/provenance.md``, "Which code built the name"). A named row with ``producer`` null is a
producer that builds a name and reports nowhere; ``scripts/producer_census.py`` counts the same thing
over a run file.
"""
import csv
from pathlib import Path

import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name

# The suite switches the OPSIN validity gate off unless a test asks for it; the producers of the
# shipped configuration are the ones with the gate on.
pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

PANEL = Path(__file__).resolve().parents[1] / "fixtures" / "zinc" / "panel_pairs.tsv"


def _panel():
    with open(PANEL, newline="", encoding="utf-8") as f:
        return [(r["id"], r["drawing"], r["smiles"]) for r in csv.DictReader(f, delimiter="\t")]


ROWS = _panel()


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("row_id,drawing,smiles", ROWS,
                         ids=[f"{i}-{(d.split() or ['row'])[0]}-{n}" for n, (i, d, _s) in enumerate(ROWS)])
def test_a_named_zinc_row_reports_its_producer(tier, row_id, drawing, smiles):
    flags = {} if tier == "pin" else _emit_tier_flags(tier)
    row = Orthonym(style="pin", **flags).name_tiered(smiles)
    named = bool(row["name"]) and not is_failure_name(row["name"]) and row["tier"] != "abstain"
    if named:
        assert row["producer"], row
    else:
        assert row["producer"] is None, row
