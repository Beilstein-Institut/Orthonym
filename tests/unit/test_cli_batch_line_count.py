"""``--batch`` writes exactly one output line per input line (GitHub issue #42).

Line N of the output belongs to line N of the input, so a caller can pair them
by position: an empty input gives an empty output, and a blank input line
gives a blank output line in the same place.
"""
from orthonym.cli import _process_batch


def _run(tmp_path, text, to_file=True, capsys=None):
    src = tmp_path / "in.smi"
    src.write_text(text)
    if to_file:
        out = tmp_path / "out.txt"
        rc = _process_batch(str(src), str(out), "pin", verbose=False)
        return rc, out.read_text()
    rc = _process_batch(str(src), None, "pin", verbose=False)
    return rc, capsys.readouterr().out


def test_empty_file_gives_empty_output_file(tmp_path):
    rc, out = _run(tmp_path, "")
    assert rc == 0
    assert out == ""


def test_empty_file_prints_nothing(tmp_path, capsys):
    rc, out = _run(tmp_path, "", to_file=False, capsys=capsys)
    assert rc == 0
    assert out == ""


def test_blank_line_keeps_its_place(tmp_path):
    rc, out = _run(tmp_path, "CCO\n\nCC(=O)O\n")
    assert rc == 0
    lines = out.split("\n")
    assert lines[-1] == ""                 # the file ends with a newline
    assert lines[:-1] == ["CCO\tethanol", "", "CC(=O)O\tacetic acid"]


def test_whitespace_only_line_is_a_blank_line(tmp_path, capsys):
    rc, out = _run(tmp_path, "  \nCCO\n", to_file=False, capsys=capsys)
    assert rc == 0
    assert out == "\nCCO\tethanol\n"


def test_input_without_final_newline(tmp_path):
    rc, out = _run(tmp_path, "CCO")
    assert rc == 0
    assert out == "CCO\tethanol\n"


def test_output_line_count_equals_input_line_count(tmp_path):
    text = "\nCCO\n\n\nCC(=O)O\nC\n\n"
    rc, out = _run(tmp_path, text)
    assert rc == 0
    assert out.count("\n") == text.count("\n")
