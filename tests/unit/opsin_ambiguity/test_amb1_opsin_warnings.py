"""OPSIN's warnings for one name, read through the in-process bridge.

``opsin_warnings`` is the per-name channel for the ``APPEARS_AMBIGUOUS`` warning. The OPSIN
command line prints that warning on stderr without the name, so only the in-process path can
attribute it to a name.
"""
import pytest

from orthonym.jvm_bridge import OpsinUnavailable, opsin_warnings


def test_ambiguous_name_reports_warning():
    # GENERAL METHODOLOGY FOR NAMING SUBSTITUENT GROUPS, method (2) (the Blue Book):
    # "except for mononuclear parent hydrides or the suffix 'ylidyne', the locant '1' must be cited."
    assert ("APPEARS_AMBIGUOUS", "Positioning of radical on: aziridin") in opsin_warnings(
        "methyl aziridinylacetate"
    )


def test_name_with_locant_has_no_warning():
    # Same rule as above: the cited locant removes the ambiguity.
    assert opsin_warnings("methyl (aziridin-1-yl)acetate") == ()


def test_rejected_name_has_no_warning():
    # A name OPSIN rejects has no structure to be ambiguous about.
    assert opsin_warnings("notaname") == ()


def test_isotope_warning_is_reported():
    # Omission of locants (the Blue Book): "if isotopic modification requires a
    # locant to specify its position, then all locants must be specified and none are omitted."
    msgs = [m for t, m in opsin_warnings("(15N)imidazolidine-2,4-dione") if t == "APPEARS_AMBIGUOUS"]
    assert msgs == ["Position of isotope on imidazolidin"]


def test_embedded_newline_cannot_be_served():
    # The command line reads one name per line, so a name with a newline has no equivalent form.
    with pytest.raises(OpsinUnavailable):
        opsin_warnings("eth\nane")
