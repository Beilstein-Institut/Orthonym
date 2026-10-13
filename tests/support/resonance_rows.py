"""Rows of tests/fixtures/resonance/*.tsv (first line = column names) and the constructed ladders of the resonance tests."""
import csv
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "resonance"


def rows(name, directory=None):
    """The rows of a TSV fixture as dicts keyed by its first line (``directory`` defaults to ``FIXTURES``)."""
    with open(Path(directory or FIXTURES) / name, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def smiles_lines(name, directory=None):
    """The first token of every non-blank line of a one-SMILES-per-line fixture."""
    with open(Path(directory or FIXTURES) / name) as handle:
        return [line.split()[0] for line in handle if line.strip()]


def _rings(n):
    """n para-linked phenylenes: 'c1ccc(cc1)-' x (n - 1), then 'c1ccccc1'."""
    return "-".join(["c1ccc(cc1)"] * (n - 1) + ["c1ccccc1"])


def azide_ladder(n):
    """An aryl azide on n phenylenes."""
    return "[N-]=[N+]=N" + _rings(n)


def imidazolium_ladder(n):
    """A 4(5)-aryl imidazolium-NH on n phenylenes."""
    return "[nH+]1c[nH]cc1-" + _rings(n)


def benzimidazolium_ladder(n):
    """A benzimidazolium-NH, N-substituted by n phenylenes."""
    return "[N+]1(=CNc2ccccc21)-" + _rings(n)


def imidazolium_with_work(count):
    """A 1-methyl-imidazolium-NH on para-linked phenylenes whose ``charge_shift.pi_work`` is ``count`` (at least 13):
    5 pi atoms + 6 per ring + 1 per dimethylamino group on the last ring (its methyl groups add none), plus 2 for the
    charged atom."""
    rings, extra = divmod(count - 7, 6)
    last = "c1" + "".join("c(N(C)C)" if i < extra else "c" for i in range(5)) + "1"
    return "C[n+]1c[nH]cc1-" + "-".join(["c1ccc(cc1)"] * (rings - 1) + [last])


def carbene_ladder(n):
    """An N-methyl-N'-aryl benzimidazol-2-ylidene on n phenylenes, drawn as the zwitterion [N+]=[C-]. The two nitrogen
    atoms carry different groups, so swapping the +1 between them would be another canonical SMILES."""
    return "C[N+]1=[C-]N(-" + _rings(n) + ")c2ccccc21"


def n_azide_ladder(n):
    """An N-methyl-N-aryl azide (an acyclic dipole on a tertiary N) on n phenylenes."""
    return "CN(N=[N+]=[N-])" + _rings(n)


_GROUPS = {"azide": "N=[N+]=[N-]", "diazonium": "[N+]#N", "nitro": "[N+](=O)[O-]", "carboxylate": "C(=O)[O-]",
           "azo": "N=NC", "nitroso": "N=O", "diazo": "C=[N+]=[N-]", "isocyanate": "N=C=O",
           "cyano": "C#N", "acetyl": "C(C)=O", "vinyl": "C=C", "oxime": "C=NO", "isothiocyanate": "N=C=S",
           "nitrile_oxide": "C#[N+][O-]", "sulfonyl_azide": "S(=O)(=O)N=[N+]=[N-]"}
#: the groups of ``flags0_family``, in the order of the calibration of the flags-0 work bound
FLAGS0_GROUPS = tuple(_GROUPS)


def flags0_family(core, group, k, n):
    """An NH azolium core (``imidazolium`` or ``benzimidazolium``), n para-linked phenylenes, then a phenyl ring with
    k groups (a key of ``_GROUPS``) at the positions para, meta, meta', ortho, ortho', in that order."""
    sub = {p: ("(%s)" % _GROUPS[group] if i < k else "") for i, p in enumerate((4, 3, 5, 2, 6))}
    ring = "c3c%sc%sc%sc%sc3%s" % (sub[2], sub[3], sub[4], sub[5], sub[6])
    link = "c4ccc(cc4)-" * n
    if core == "imidazolium":
        return "C[n+]1c[nH]cc1-" + link + ring
    return "C[n+]1c(-" + link + ring + ")[nH]c2ccccc21"
