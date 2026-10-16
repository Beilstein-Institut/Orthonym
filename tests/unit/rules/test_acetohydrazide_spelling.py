"""The two-carbon hydrazide is 'acetohydrazide', never 'ethanehydrazide'.

 (the Blue Book): "Names of hydrazides are formed by changing the 'ic acid' or '-oic acid'
ending of the retained names of carboxylic acids into 'ohydrazide'"; (:33938): "Only the following
five names are preferred IUPAC names and can be substituted in the same way as corresponding amides...
acetohydrazide (PIN) benzohydrazide (PIN) oxalohydrazide (PIN)" (:33944; formohydrazide and cyanohydrazide
beside them). The producers wrote the systematic stem for a substituted two-carbon hydrazide ('N'-methylethane
hydrazide', '2-(methylsulfanyl)ethanehydrazide') and shipped it as pin_verified, while the amide of the same
chain has had the retained name since (:32691). The retained C2 spelling lives in ONE table,
``composition_primitives.RETAINED_C2_ACYL_BY_SUFFIX``, read by the general assembler, the polyfunctional
producer and (through the assembler) the N/N'-substituted hydrazide producer in composer.py.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.assembly.composition_primitives import RETAINED_C2_ACYL_BY_SUFFIX
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _same(smi, name):
    out = opsin_parse(name)
    m_out = Chem.MolFromSmiles(out) if out else None
    return m_out is not None and inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) == inchi.MolToInchiKey(m_out)


@pytest.mark.parametrize("smiles,expected", [
    ("CNNC(=O)C", "N'-methylacetohydrazide"),                         # the review's row
    ("CN(N)C(C)=O", "N-methylacetohydrazide"),
    ("NNC(=O)CSC", "2-(methylsulfanyl)acetohydrazide"),
    ("NNC(=O)CCl", "2-chloroacetohydrazide"),
])
def test_substituted_acetohydrazides(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert _same(smiles, expected)


@pytest.mark.parametrize("smiles,expected", [
    ("NNC(=O)C", "acetohydrazide"),                                    #:33944
    ("NNC(=O)CC", "propanehydrazide"),
    ("NNC(=O)CCC", "butanehydrazide"),
    ("NC(=O)CSC", "2-(methylsulfanyl)acetamide"),                      # the amide of the same chain: unchanged
    ("NNC(=S)C", None),                                                # thiohydrazide: not in the table
])
def test_neighbours(namer, smiles, expected):
    name = namer.name(smiles)
    if expected is None:
        assert "acetohydrazide" not in name
    else:
        assert name == expected


def test_the_table_is_the_one_home():
    assert RETAINED_C2_ACYL_BY_SUFFIX == {"amide": "acetamide", "hydrazide": "acetohydrazide"}


@pytest.mark.parametrize("name", [
    "4-(1-sulfanylideneethyl)benzoic acid",
    "4-chloro-7-[(3-chloro-4-nitroquinolin-7-yl)sulfanyl]-3-nitroquinoline",
    "2-(cyanosulfonyl)cyclohexane-1-carboxylic acid",
    "1-(2-oxophosphepan-3-yl)propan-1-one",
])
def test_the_four_refuted_names_are_not_pin_verified(namer, name):
    out = opsin_parse(name)
    if out:
        row = namer.name_tiered(out)
        assert not (row["tier"] == "pin_verified" and row["name"] == name)
