"""The pages that describe the ``producer`` key say what the code reports.

``Orthonym.name_tiered`` returns a ``producer`` id in every row. Three things keep its
documentation from drifting:

* the docstring of ``name_tiered`` and the provenance page name the key, and the page lists
  ``spelling_failures``, which every row carries;
* every id the code reports is on the page: the fixed ids are read from the seams in ``src``
  (the literal first argument of ``noted``), the ``outer:`` and ``inner:`` families from the
  dispatch tables, and the page lists exactly those fixed ids;
* every example row on the four pages that print a full row has the keys of a live row, and the
  ``producer`` it shows matches the documented ids.

The test reads ``src``, ``guide/`` and ``tests/support/producer_pairs.py`` only.
"""
import ast
import json
import re
from pathlib import Path

import pytest

import orthonym
from orthonym import Orthonym
from orthonym.assembly import handlers  # noqa: F401 (registers the inner handlers)
from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
from orthonym.cli import _emit_tier_flags
from orthonym.routing.dispatch_table import DISPATCH_TABLE
from tests.support.producer_pairs import PAIRS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(orthonym.__file__).resolve().parent
PROVENANCE = ROOT / "guide" / "use" / "provenance.md"
ROW_PAGES = [PROVENANCE, ROOT / "guide" / "declines.md", ROOT / "guide" / "start" / "first-name.md",
             ROOT / "guide" / "tiers.md"]

OUTER_IDS = frozenset(entry.class_id.value for entry in DISPATCH_TABLE.values())
INNER_IDS = frozenset(INNER_DISPATCH_TABLE)
FAMILY_ROWS = ("outer:<class>", "inner:<handler>")


def documented(producer):
    """True when ``producer`` is an id the code can report (the page documents the same set)."""
    if producer in _fixed_ids():
        return True
    kind, _, rest = producer.partition(":")
    return (kind == "outer" and rest in OUTER_IDS) or (kind == "inner" and rest in INNER_IDS)


def _fixed_ids():
    """The literal first argument of every ``noted(...)`` seam in ``src``: the fixed ids."""
    global _FIXED
    if _FIXED is None:
        found = set()
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and (getattr(node.func, "id", None) in ("noted", "_noted")
                             or getattr(node.func, "attr", None) in ("noted", "_noted"))):
                    found.add(node.args[0].value)
        _FIXED = frozenset(found)
    return _FIXED


_FIXED = None


def _page_rows(page):
    """The full example rows of a page: the JSON objects with a ``limit_code`` key that stand in
    its fenced blocks (one per line, or pretty-printed)."""
    text = page.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S):
        pos = 0
        while True:
            start = block.find("{", pos)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(block, start)
            except ValueError:
                pos = start + 1
                continue
            if isinstance(obj, dict) and "limit_code" in obj:
                rows.append(obj)
            pos = end
    return rows


def _row(smiles, tier="pin"):
    flags = {} if tier == "pin" else _emit_tier_flags(tier)
    return Orthonym(style="pin", **flags).name_tiered(smiles)


# ---------------------------------------------------------------------------------------------
# the text
# ---------------------------------------------------------------------------------------------

def test_the_docstring_and_the_page_name_the_key():
    assert "``producer``" in Orthonym.name_tiered.__doc__
    assert "`producer`" in PROVENANCE.read_text(encoding="utf-8")


def test_the_page_lists_spelling_failures():
    # every row carries the key; the field list of the page used to omit it
    assert "`spelling_failures`" in PROVENANCE.read_text(encoding="utf-8")


def test_the_id_table_lists_exactly_the_ids_the_code_reports():
    text = PROVENANCE.read_text(encoding="utf-8")
    table = text.split("|:--|:--|", 1)[1]               # the rows under the header of the id table
    listed = set(re.findall(r"^\| `([^`]+)` \|", table, flags=re.M))
    assert listed == _fixed_ids() | set(FAMILY_ROWS), (
        f"on the page, not in the code: {sorted(listed - _fixed_ids() - set(FAMILY_ROWS))}; "
        f"in the code, not on the page: {sorted(_fixed_ids() - listed)}")
    # the scan of src finds the seams (a page and a code side that are both empty would agree)
    assert len(_fixed_ids()) >= 11, sorted(_fixed_ids())


def test_the_dispatch_tables_are_the_families():
    # 'outer:<class>' is a class id in lower case, 'inner:<handler>' a handler id
    assert len(OUTER_IDS) == len(DISPATCH_TABLE) and OUTER_IDS
    assert INNER_IDS and all(entry.handler_id == key for key, entry in INNER_DISPATCH_TABLE.items())


def test_the_paths_the_docs_name_exist():
    architecture = (ROOT / "guide" / "architecture.md").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "metrics/producer_trail.py" in architecture and (SRC / "metrics" / "producer_trail.py").is_file()
    static = "tests/unit/metrics/test_producer_trail_static.py"
    assert static in contributing and (ROOT / static).is_file()


# ---------------------------------------------------------------------------------------------
# the ids
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,tier,expected", PAIRS, ids=[f"{e}-{i}" for i, (_s, _t, e) in enumerate(PAIRS)])
def test_the_expected_ids_follow_the_documented_patterns(smiles, tier, expected):
    assert documented(expected), expected


def test_the_checker_of_the_patterns():
    assert documented("composer") and documented("table:np_parent")
    assert documented("outer:retained_name") and documented("inner:tier_a_ring")
    assert not documented("outer:tier_a_ring") and not documented("inner:retained_name")
    assert not documented("outer:") and not documented("engine") and not documented("table:other")


def _first_row_of_each_family():
    families = ("outer:", "inner:", "composer", "engine:", "decomposition", "isotope", "multifragment",
                "retained_fused", "table:")
    chosen = []
    for family in families:
        chosen.append(next(pair for pair in PAIRS if pair[2].startswith(family)))
    return chosen


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,tier,expected", _first_row_of_each_family(),
                         ids=[expected for _s, _t, expected in _first_row_of_each_family()])
def test_a_live_row_reports_a_documented_id(smiles, tier, expected):
    producer = _row(smiles, tier)["producer"]
    assert producer is not None and documented(producer), producer


# ---------------------------------------------------------------------------------------------
# the example rows
# ---------------------------------------------------------------------------------------------

def test_the_example_rows_have_the_keys_of_a_live_row():
    live = list(_row("CCO"))
    assert "producer" in live and "spelling_failures" in live
    count = 0
    for page in ROW_PAGES:
        rows = _page_rows(page)
        assert rows, f"{page.relative_to(ROOT)} prints no full row"
        for row in rows:
            count += 1
            assert set(row) == set(live), (page.relative_to(ROOT), set(row) ^ set(live))
            producer = row["producer"]
            assert producer is None or documented(producer), (page.relative_to(ROOT), producer)
    assert count >= 7, count


def test_a_decline_row_shows_a_null_producer():
    # the pages print the decline rows of the declines page: nothing ships, so no code is named
    rows = _page_rows(ROOT / "guide" / "declines.md")
    declines = [row for row in rows if row["tier"] == "abstain"]
    assert declines and all(row["producer"] is None for row in declines)
