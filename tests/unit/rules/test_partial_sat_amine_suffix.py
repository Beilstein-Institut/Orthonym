"""Ring `-amine` suffix on a partially-saturated (hydro-fused) carbocycle
(follow-on to DEFECT-hydro-fused-substituent-locants.md).
`1,2,3,4-tetrahydronaphthalen-1-amine` is a Blue Book PIN (the Blue Book);
the `-ol` sibling already worked, this adds the amine. Each emission full-InChIKey
round-trips; seniority: amine < alcohol) and 0-wrong are asserted.
"""
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


def _name(smi):
    return Orthonym().name_tiered(smi).get("name")


def _full(smi, name):
    o = opsin_parse(name)
    return bool(o) and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


def test_tetrahydronaphthalen_amine():
    for smi, exp in [
        ("NC1CCc2ccccc2C1", "1,2,3,4-tetrahydronaphthalen-2-amine"),
        ("NC1CCCc2ccccc21", "1,2,3,4-tetrahydronaphthalen-1-amine"),
        ("NC1CCc2ccccc21", "2,3-dihydro-1H-inden-1-amine"),
    ]:
        name = _name(smi)
        assert name == exp, (smi, name)
        assert _full(smi, name)


def test_diamine_keeps_terminal_e():
    smi = "NC1CCc2ccc(N)cc2C1"
    name = _name(smi)
    assert name and "diamine" in name and _full(smi, name), name


def test_alcohol_is_senior_to_amine():
    # (the Blue Book): class 17 hydroxy compounds before class 19 amines, so with both
    # -OH and -NH2 the alcohol takes the suffix and the amine path must NOT claim it. It used to abstain
    # ('unknown organic compound') because the tetralin-class veto withheld the right name whenever the
    # hydro-fused producer refused; the name is the PIN now. change-asserted-value: independent check =
    # the read-back below, mutation = tests/unit/perception/test_partial_sat_veto_scope.py.
    smi = "NC1CCc2ccc(O)cc2C1"
    name = _name(smi)
    assert name == "7-amino-5,6,7,8-tetrahydronaphthalen-2-ol", name
    assert _full(smi, name)


def test_ol_and_plain_unchanged():
    assert _name("OC1CCc2ccccc2C1") == "1,2,3,4-tetrahydronaphthalen-2-ol"
    assert _name("CC1CCc2ccccc2C1") == "2-methyl-1,2,3,4-tetrahydronaphthalene"
    assert _name("C1CCCc2ccccc21") == "1,2,3,4-tetrahydronaphthalene"
