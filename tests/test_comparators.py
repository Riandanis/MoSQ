"""Tests for the faithful comparator reimplementations.

These pin the properties that make the models *faithful* rather than merely
functional. A shape test would pass on any architecture; what matters here is
that the propagation rule, feature widths and ontology wiring match the
published implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gastro_transformer.comparators import DeepCDR, DrugCell, GraphConv, TCNNSHybrid
from gastro_transformer.comparators.featurize import (
    ATOM_FEATURE_DIM,
    molecular_graphs,
    smiles_onehot,
)

pytest.importorskip("rdkit")

SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
}


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------

def test_atom_feature_width_is_75():
    """deepchem's ConvMolFeaturizer width. DeepCDR's first GCN layer assumes it."""
    assert ATOM_FEATURE_DIM == 75


def test_molecular_graphs_shapes_and_symmetry():
    features, adjacency, drug_ids = molecular_graphs(SMILES)
    assert features.shape[0] == adjacency.shape[0] == len(SMILES)
    assert features.shape[2] == 75
    assert adjacency.shape[1] == adjacency.shape[2] == features.shape[1]
    np.testing.assert_array_equal(adjacency, adjacency.transpose(0, 2, 1))
    assert drug_ids == sorted(SMILES)


def test_adjacency_has_no_self_loops():
    """DeepCDR applies the raw adjacency; self-loops would change the model."""
    _, adjacency, _ = molecular_graphs(SMILES)
    for i in range(adjacency.shape[0]):
        assert adjacency[i].diagonal().sum() == 0


def test_benzene_graph_is_a_six_ring():
    features, adjacency, drug_ids = molecular_graphs({"benzene": "c1ccccc1"})
    assert adjacency[0, :6, :6].sum() == 12  # six bonds, both directions
    assert features[0, :6].sum(axis=1).min() > 0
    assert features[0, 6:].sum() == 0  # padding stays zero


def test_padding_rows_are_zero():
    features, adjacency, drug_ids = molecular_graphs(SMILES)
    small = drug_ids.index("ethanol")
    assert features[small, 3:].sum() == 0
    assert adjacency[small, 3:, :].sum() == 0


def test_smiles_onehot_is_one_hot_per_position():
    tensor, drug_ids, chars, length = smiles_onehot(SMILES)
    assert tensor.shape == (len(SMILES), length, len(chars))
    per_position = tensor.sum(axis=2)
    assert set(np.unique(per_position)).issubset({0.0, 1.0})
    # Each molecule has exactly len(smiles) occupied positions.
    for i, drug_id in enumerate(drug_ids):
        assert per_position[i].sum() == len(SMILES[drug_id])


# ---------------------------------------------------------------------------
# DeepCDR
# ---------------------------------------------------------------------------

def test_graphconv_matches_reference_formula():
    """DeepCDR's GraphConv is A^T(HW + b), unnormalised and without self-loops."""
    layer = GraphConv(4, 3)
    features = torch.randn(2, 5, 4)
    adjacency = torch.randint(0, 2, (2, 5, 5)).float()

    expected = torch.bmm(adjacency.transpose(1, 2), layer.linear(features))
    torch.testing.assert_close(layer(features, adjacency), expected)


def test_graphconv_differs_from_normalised_gcn():
    """Guards against silently substituting a standard GCN propagation rule."""
    layer = GraphConv(4, 3)
    features = torch.randn(2, 5, 4)
    adjacency = torch.randint(0, 2, (2, 5, 5)).float()

    transformed = layer.linear(features)
    degree = adjacency.sum(dim=2, keepdim=True).clamp(min=1)
    normalised = torch.bmm(adjacency.transpose(1, 2), transformed) / degree
    assert not torch.allclose(layer(features, adjacency), normalised)


def test_deepcdr_forward_and_isolated_omics():
    model = DeepCDR().eval()
    features, adjacency, _ = molecular_graphs(SMILES)
    batch = len(SMILES)
    out = model(
        torch.from_numpy(features), torch.from_numpy(adjacency),
        torch.zeros(batch, 34673), torch.zeros(batch, 697), torch.zeros(batch, 808),
    )
    assert out.shape == (batch,)
    assert torch.isfinite(out).all()


def test_deepcdr_can_drop_omics_branches():
    model = DeepCDR(use_methylation=False, use_mutation=False).eval()
    features, adjacency, _ = molecular_graphs(SMILES)
    out = model(
        torch.from_numpy(features), torch.from_numpy(adjacency),
        expression=torch.zeros(len(SMILES), 697),
    )
    assert out.shape == (len(SMILES),)


def test_deepcdr_responds_to_drug_identity():
    """Different molecules must give different predictions, or the graph branch is dead."""
    torch.manual_seed(0)
    model = DeepCDR().eval()
    features, adjacency, _ = molecular_graphs(SMILES)
    batch = len(SMILES)
    out = model(
        torch.from_numpy(features), torch.from_numpy(adjacency),
        torch.zeros(batch, 34673), torch.zeros(batch, 697), torch.zeros(batch, 808),
    )
    assert out.std() > 0


# ---------------------------------------------------------------------------
# DrugCell
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_ontology():
    """Small DAG: root <- {a, b}; a <- leaf; genes attached at several levels."""
    children = {"root": ["a", "b"], "a": ["leaf"], "b": [], "leaf": []}
    genes = {"root": [0], "a": [1, 2], "b": [3, 4, 5], "leaf": [6, 7]}
    order = ["leaf", "a", "b", "root"]
    return children, genes, order, "root", 8


def test_drugcell_forward(toy_ontology):
    children, genes, order, root, n_genes = toy_ontology
    model = DrugCell(children, genes, order, root, n_genes=n_genes, drug_dim=16).eval()
    out = model(torch.rand(4, n_genes).round(), torch.rand(4, 16).round())
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_drugcell_term_fan_in_follows_ontology(toy_ontology):
    """A term's input width must be children x neurons + its own gene count."""
    children, genes, order, root, n_genes = toy_ontology
    model = DrugCell(children, genes, order, root, n_genes=n_genes, drug_dim=16,
                     neurons_per_term=6)
    assert model.term_linear["leaf"].in_features == 2          # genes only
    assert model.term_linear["a"].in_features == 6 + 2         # one child + 2 genes
    assert model.term_linear["root"].in_features == 6 * 2 + 1  # two children + 1 gene


def test_drugcell_rejects_disconnected_term():
    with pytest.raises(ValueError, match="neither children nor genes"):
        DrugCell({"x": []}, {"x": []}, ["x"], "x", n_genes=4)


def test_drugcell_uses_mutation_input(toy_ontology):
    torch.manual_seed(0)
    children, genes, order, root, n_genes = toy_ontology
    model = DrugCell(children, genes, order, root, n_genes=n_genes, drug_dim=16).eval()
    drug = torch.rand(4, 16).round()
    a = model(torch.zeros(4, n_genes), drug)
    b = model(torch.ones(4, n_genes), drug)
    assert not torch.allclose(a, b)


@pytest.mark.slow
def test_real_ontology_is_a_single_rooted_dag():
    """The shipped GO file must parse to one root with a valid leaves-first order."""
    from pathlib import Path

    from gastro_transformer.comparators.data import load_drugcell_ontology

    path = Path(__file__).resolve().parent.parent / "data/comparators/drugcell_drugcell_ont.txt"
    if not path.exists():
        pytest.skip("comparator data not downloaded")

    children, genes, order, root, gene_index = load_drugcell_ontology()
    assert root.startswith("GO:")
    assert len(gene_index) == 3008
    position = {term: i for i, term in enumerate(order)}
    for parent, kids in children.items():
        for child in kids:
            assert position[child] < position[parent], f"{child} must precede {parent}"


# ---------------------------------------------------------------------------
# tCNNS hybrid
# ---------------------------------------------------------------------------

# The published network pools by 3 three times, so inputs shorter than 27 are
# invalid by construction. Real SMILES here run to 196 characters; the toy set
# above is shorter, so these tests pad to a realistic length.
REAL_LENGTH = 196


def test_tcnns_forward_shape():
    tensor, _, chars, length = smiles_onehot(SMILES, length=REAL_LENGTH)
    model = TCNNSHybrid(n_smiles_chars=len(chars), smiles_length=length, cell_dim=256).eval()
    out = model(torch.from_numpy(tensor), torch.randn(len(SMILES), 256))
    assert out.shape == (len(SMILES),)


def test_tcnns_output_is_unbounded_by_default():
    """Targets are LN(IC50), not [0,1]-normalised, so no sigmoid."""
    tensor, _, chars, length = smiles_onehot(SMILES, length=REAL_LENGTH)
    model = TCNNSHybrid(n_smiles_chars=len(chars), smiles_length=length).eval()
    assert model.output_activation == "identity"
    with torch.no_grad():
        model.fc[-1].bias.fill_(50.0)
        out = model(torch.from_numpy(tensor), torch.randn(len(SMILES), 256))
    assert out.max() > 1.0


def test_tcnns_drug_branch_channel_widths():
    """Published spec: three Conv1d layers, kernel 7, 40 -> 80 -> 60 channels."""
    tensor, _, chars, length = smiles_onehot(SMILES, length=REAL_LENGTH)
    model = TCNNSHybrid(n_smiles_chars=len(chars), smiles_length=length)
    convs = [m for m in model.drug_branch if isinstance(m, torch.nn.Conv1d)]
    assert [c.out_channels for c in convs] == [40, 80, 60]
    assert all(c.kernel_size == (7,) for c in convs)
    pools = [m for m in model.drug_branch if isinstance(m, torch.nn.MaxPool1d)]
    assert all(p.kernel_size == 3 for p in pools)


def test_tcnns_rejects_too_short_input():
    """A clear error beats an opaque max_pool1d failure deep in the stack."""
    with pytest.raises(ValueError, match="too short for tCNNS"):
        TCNNSHybrid(n_smiles_chars=29, smiles_length=20)
