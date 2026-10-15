"""LO4b: a ring system with a tetravalent fusion atom is named by fusion nomenclature.

The fused-guanidine rows of the fu4 census (`internal notes`, cause A3) that
declined at ``selection.fused_split`` are the saxitoxin / gonyautoxin skeletons: three rings (5, 5, 6)
around one carbon with four ring bonds. Two of the rings (the imidazoline and the pyrrolidine) meet only
at that carbon; the six-membered ring shares a bond with each. The skeleton is purine with a pyrrole ring
fused on its side c (N3-C4, whose C4 is a fusion atom of purine): 'pyrrolo[1,2-c]purine'.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* (:11865): "Two rings that have only two atoms and one bond in common are said to be
  *ortho*-fused"; (:11871): a ring "ortho-fused to different sides of two other rings that are
  themselves ortho-fused together (i.e. there are three common atoms between the first ring and the other
  two)" is ortho- and peri-fused. The pyrrole ring has two atoms in common with purine (N3 and C4), so it is
  ortho-fused to purine, not peri-fused.
* Omission of locants in fusion descriptors, (:13996, sentence:14007): "Both terminal fusion atom
  locants for ortho-fusion must be cited even if one is a fusion atom." Example (:14011): 'naphtho[1,8a-b]
  azirine (PIN)': the azirine ring is fused on the bond C1-C8a of naphthalene, so C8a has four ring bonds
  and lies in three rings, the topology of the saxitoxin skeleton. The book prints it as a PIN.
* Five-membered ring requirement (:23710): "Fusion nomenclature gives preferred IUPAC names only
  to compounds having at least two rings of at least five or more members."
* Limitations to fusion nomenclature (:23757, sentence:23759): "It is not possible by these principles to name a
  system in which a third component is *ortho*- and *peri*-fused to two components that are themselves
  *ortho*- or *ortho*- and *peri*-fused together", then 'a' replacement nomenclature on a hydrocarbon with a
  retained name,:23787: '1,3a1,4,9-tetraazaphenalene (PIN)',:23798). The cylindrospermopsin
  skeleton (an interior nitrogen common to a 5-, a 6- and a 6-membered ring, each pair sharing a bond) is such
  a system, and its interior nitrogen is cited by a superscript locant,:12658) that OPSIN
  2.9.0 cannot read, so it stays declined (``test_the_peri_fused_interior_nitrogen_skeleton_stays_declined``).
ASSUMED (the book does not decide): the numbering of a tetravalent skeleton is OPSIN 2.9.0's reading of its
name (as every fusion numbering of the builder); ``compute_fused_numbering`` has no tetravalent atom in its
grid, so the one independent check is ``fusion_components.periphery_numbering_ok`` (the locants run 1, 2,...
round the periphery,,:12501). That OPSIN's locants for saxitoxin (1, 2, 3, 3a, 4, 5, 6, 7, 8, 9,
10, 10a: N1 the imidazoline NH, 2 and 6 the amino carbons, 10 the diol carbon) are the ones the saxitoxin
literature uses is remembered, not read (ASSUMED).
"""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import (
    build_fused,
    hetero_fusion,
    parents,
    selection,
)
from orthonym.rules.bridged_fused_pin import (
    fusion_components as fc,
)
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _skeleton(smiles):
    """The all-single-bond element graph of the ring system of ``smiles`` (``parents._skeleton``)."""
    mol = Chem.MolFromSmiles(smiles)
    sk, _ = parents._skeleton(mol, selection.ring_system(mol))
    return Chem.MolFromSmiles(Chem.MolToSmiles(sk))


# The structure OPSIN 2.9.0 reads from the book's 'naphtho[1,8a-b]azirine' (:14011)
BOOK_AZIRINE = "C1=CC2=CC=CC3=NC23C=C1"
# saxitoxin without its carbamate and sulfate: the skeleton of the item's rows
SAXITOXIN_CORE = "NC1=N[C@H]2[C@H](CO)N=C(N)N3CC[C@@H](O)[C@]23N1"


def test_the_book_example_names_itself():
    """ (:14007,:14011): 'naphtho[1,8a-b]azirine (PIN)'; the old code had no fusion name
    for a skeleton with a tetravalent fusion atom."""
    assert name_is_rt_exact("naphtho[1,8a-b]azirine", BOOK_AZIRINE)
    assert build_fused(Chem.MolFromSmiles(BOOK_AZIRINE))[0] == "naphtho[1,8a-b]azirine"
    assert_pin_at_both_tiers(BOOK_AZIRINE, "naphtho[1,8a-b]azirine")


def test_the_skeleton_of_the_item_is_pyrrolo_1_2_c_purine():
    km = _skeleton(SAXITOXIN_CORE)
    assert hetero_fusion.rule_name(km) == "pyrrolo[1,2-c]purine"
    system = selection.ring_system(Chem.MolFromSmiles(SAXITOXIN_CORE))
    assert selection.fused_split(Chem.MolFromSmiles(SAXITOXIN_CORE), system).parent == "pyrrolo[1,2-c]purine"


# the item's rows that the builder now names (the principal group is on the ring system), at both tiers
ITEM_ROWS = [
    # saxitoxin core: a ring -OH and the hydroxymethyl of neosaxitoxin's family
    (SAXITOXIN_CORE,
     "(3aS,4R,10R,10aS)-2,6-diamino-4-(hydroxymethyl)-3a,4,9,10-tetrahydro-1H,8H-pyrrolo[1,2-c]purin-10-ol"),
    ("NC1=N[C@H]2[C@H](CO)N=C(N)N3CC[C@H](O)[C@]23N1",
     "(3aS,4R,10S,10aS)-2,6-diamino-4-(hydroxymethyl)-3a,4,9,10-tetrahydro-1H,8H-pyrrolo[1,2-c]purin-10-ol"),
    ("C[C@@H]1N=C(N)N2CC[C@H](O)[C@@]23NC(N)=N[C@@H]13",
     "(3aS,4S,10S,10aS)-2,6-diamino-4-methyl-3a,4,9,10-tetrahydro-1H,8H-pyrrolo[1,2-c]purin-10-ol"),
    ("C[C@@H]1N=C(N)N2CC[C@@H](O)[C@@]23NC(N)=N[C@@H]13",
     "(3aS,4S,10R,10aS)-2,6-diamino-4-methyl-3a,4,9,10-tetrahydro-1H,8H-pyrrolo[1,2-c]purin-10-ol"),
]


@pytest.mark.parametrize("smiles,pin", ITEM_ROWS)
def test_item_rows_are_named_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_the_bare_skeleton_and_its_diamine():
    assert_pin_at_both_tiers("C1CN2CNCC3NCNC32C1", "octahydro-1H,8H-pyrrolo[1,2-c]purine")
    assert_pin_at_both_tiers("NC1=NC2CN=C(N)N3CCCC23N1", "3a,4,9,10-tetrahydro-1H,8H-pyrrolo[1,2-c]purine-2,6-diamine")


# other skeletons of the class: a retained bicycle (indole, quinoline, isoindole) with a ring fused on a
# bond that ends at a fusion atom (the same shape as the book's naphtho[1,8a-b]azirine)
CLASS_ROWS = [
    ("C1CCC23CCCC2CCNC3C1", "dodecahydrocyclopenta[d]quinoline"),
    ("C1CCC23CCCC2CNCC3C1", "dodecahydrocyclopenta[d]isoquinoline"),
    ("C1CCC23CCCCN2CCC3C1", "decahydro-6H-pyrido[2,1-i]indole"),
    ("C1CN2CCC3CNCC32C1", "octahydro-7H-pyrrolo[3,4-g]pyrrolizine"),
    ("C1CCC23COCC2COC3C1", "hexahydro-1H,3H,7H-furo[3,4-c][1]benzofuran"),
]


@pytest.mark.parametrize("smiles,pin", CLASS_ROWS)
def test_neighbours_of_the_class(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# --------------------------------------------------------------------------------------------------
# the ring perception
# --------------------------------------------------------------------------------------------------
def test_a_tetravalent_fusion_atom_is_in_three_rings():
    km = _skeleton(SAXITOXIN_CORE)
    rings = fc.ortho_rings(km)
    assert sorted(len(r) for r in rings) == [5, 5, 6]
    tetra = fc.tetravalent_fusion_atoms(km, rings)
    assert len(tetra) == 1
    (atom,) = tetra
    assert km.GetAtomWithIdx(atom).GetDegree() == 4 and sum(1 for r in rings if atom in r) == 3


@pytest.mark.parametrize("smiles,why", [
    ("C1=CC2=CC=CC3=CC=CC(=C1)C23", "1H-phenalene: an interior atom with three ring bonds is peri-fusion"),
    ("C1CC2CCC1C2", "two rings sharing three atoms: a bridged system"),
    ("C1CCC2(CC1)CCCC2", "a spiro atom: two rings, one atom"),
])
def test_other_ring_systems_are_still_not_ortho_fused(smiles, why):
    assert fc.ortho_rings(_skeleton(smiles)) is None, why


def test_the_peri_fused_interior_nitrogen_skeleton_stays_declined():
    """The cylindrospermopsin skeleton: N interior to a 5-, a 6- and a 6-membered ring.
    (:23759) sends it to skeletal replacement on acenaphthylene, whose interior atom the Blue Book
    cites with a superscript locant,:12658: '1,3a1,4,9-tetraazaphenalene (PIN)',
    :23798). OPSIN 2.9.0 cannot read that locant, so no name of it can be checked and the builder
    declines (``fused_rings.py`` records the same decision for every interior heteroatom)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smiles = "C[C@H]1[C@@H](OS(=O)(=O)O)C[C@@H]2C[C@H](Cc3cc(=O)[nH]c(=O)[nH]3)NC3=NC[C@H]1N32"
    mol = Chem.MolFromSmiles(smiles)
    assert fc.ortho_rings(_skeleton(smiles)) is None
    assert selection.fused_split(mol, selection.ring_system(mol)) is None
    assert opsin_parse("1,3a1,4,9-tetraazaphenalene") is None
    assert opsin_parse("1,4,9,9b-tetraazaphenalene") is not None   # the old interior numbering it reads
    assert name_default(smiles).get("tier") != "pin_verified"
    assert "pyrrolo" not in (name_breadth(smiles).get("name") or "")


# --------------------------------------------------------------------------------------------------
# the numbering check
# --------------------------------------------------------------------------------------------------
def _opsin_numbering(smiles):
    km = _skeleton(smiles)
    name = hetero_fusion.hetero_component_name(km)
    smi, locants = hetero_fusion.opsin_structure(name, km.GetNumAtoms())
    iso = parents._iso(Chem.MolFromSmiles(smi), km)
    return km, {iso[i]: loc for i, loc in enumerate(locants)}


def test_opsin_numbering_of_the_skeleton_is_a_peripheral_numbering():
    km, numbering = _opsin_numbering(SAXITOXIN_CORE)
    assert sorted(numbering.values(), key=lambda x: (int("".join(c for c in x if c.isdigit())), x)) == [
        "1", "2", "3", "3a", "4", "5", "6", "7", "8", "9", "10", "10a"]
    assert fc.periphery_numbering_ok(km, numbering)


def test_a_numbering_that_is_not_peripheral_is_refused():
    km, numbering = _opsin_numbering(SAXITOXIN_CORE)
    by_loc = {loc: atom for atom, loc in numbering.items()}
    swapped = dict(numbering)
    swapped[by_loc["4"]], swapped[by_loc["5"]] = "5", "4"
    assert not fc.periphery_numbering_ok(km, swapped)
    letters_moved = dict(numbering)
    letters_moved[by_loc["3a"]], letters_moved[by_loc["10a"]] = "10a", "3a"
    assert not fc.periphery_numbering_ok(km, letters_moved)


def test_a_skeleton_without_a_tetravalent_atom_is_not_judged():
    km = _skeleton("c1ccc2ncccc2c1")
    assert fc.periphery_numbering_ok(km, {})


def test_purine_is_a_parent_with_its_traditional_numbering():
    """ (:12493): purine keeps its traditional numbering, whose fusion carbons C4 and C5 carry
    plain numbers; the parent was not named by the builder at all before."""
    mol = Chem.MolFromSmiles("c1ncc2[nH]cnc2n1")
    got = parents.fused_parent(mol, set(range(mol.GetNumAtoms())))
    assert got is not None and got.name == "purine"
    assert sorted(got.numberings[0].values()) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
