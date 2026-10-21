"""Faithful PyTorch reimplementations of the published comparator models.

The versions previously in this repository were architectural *analogues* — MLP
branches with different fusion heads, all consuming the same 1080-d feature
vector. A like-for-like comparison with the published methods needs more: an analogue does
not provide one, because it holds the input representation fixed at ours. These
are ports of the published architectures, layer for layer, consuming the inputs
the original papers used.

Sources (both MIT licensed; tCNNS ships no licence so it is reimplemented from
its published architecture rather than adapted):
  DeepCDR   github.com/kimmo1019/DeepCDR       (Keras)
  DrugCell  github.com/idekerlab/DrugCell      (PyTorch)
  tCNNS     github.com/Lowpassfilter/tCNNS-Project (TensorFlow 1)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeepCDR (Liu et al. 2020)
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """DeepCDR's graph convolution: ``A^T (H W + b)``.

    Deliberately *not* a standard GCN layer. The published implementation applies
    the raw adjacency with no self-loops, and its degree normalisation is
    commented out in the source:

        features = K.dot(features, self.W) + self.b
        return K.batch_dot(K.permute_dimensions(edges, (0, 2, 1)), features)
        #      / (K.sum(edges, axis=2, keepdims=True) + K.epsilon())

    Substituting ``torch_geometric.nn.GCNConv`` here would apply symmetric
    normalisation with self-loops and therefore implement a different model.
    """

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # features [B, A, F], adjacency [B, A, A]
        transformed = self.linear(features)
        return torch.bmm(adjacency.transpose(1, 2), transformed)


class DeepCDR(nn.Module):
    """Hybrid graph-convolutional network over drug graphs and three omics.

    Layer sizes follow ``prog/model.py`` in the reference implementation:
    GCN units 256/256/256 then a final 100-unit GCN with global max pooling; a
    Conv2d mutation branch; dense expression and methylation branches; and a
    Conv2d fusion stack over the concatenated 300-d representation.
    """

    def __init__(
        self,
        atom_dim: int = 75,
        mutation_dim: int = 34673,
        expression_dim: int = 697,
        methylation_dim: int = 808,
        gcn_units: Sequence[int] = (256, 256, 256),
        dropout: float = 0.1,
        use_mutation: bool = True,
        use_expression: bool = True,
        use_methylation: bool = True,
    ):
        super().__init__()
        self.use_mutation = use_mutation
        self.use_expression = use_expression
        self.use_methylation = use_methylation

        # --- drug branch: GCN stack -> 100-d, global max pool ---
        dims = [atom_dim, *gcn_units]
        self.gcn_layers = nn.ModuleList(
            [GraphConv(dims[i], dims[i + 1]) for i in range(len(gcn_units))]
        )
        self.gcn_norms = nn.ModuleList([nn.BatchNorm1d(u) for u in gcn_units])
        self.gcn_final = GraphConv(gcn_units[-1], 100)
        self.gcn_final_norm = nn.BatchNorm1d(100)
        self.dropout = nn.Dropout(dropout)

        n_branches = 1  # drug
        # --- mutation branch: Conv2d over the [1, D] "image" ---
        if use_mutation:
            self.mut_conv1 = nn.Conv2d(1, 50, kernel_size=(1, 700), stride=(1, 5))
            self.mut_pool1 = nn.MaxPool2d((1, 5))
            self.mut_conv2 = nn.Conv2d(50, 30, kernel_size=(1, 5), stride=(1, 2))
            self.mut_pool2 = nn.MaxPool2d((1, 10))
            with torch.no_grad():
                probe = torch.zeros(1, 1, 1, mutation_dim)
                probe = self.mut_pool1(torch.tanh(self.mut_conv1(probe)))
                probe = self.mut_pool2(F.relu(self.mut_conv2(probe)))
                flat = probe.numel()
            self.mut_fc = nn.Linear(flat, 100)
            n_branches += 1

        # --- expression and methylation branches: 256 tanh -> 100 relu ---
        if use_expression:
            self.expr_fc1 = nn.Linear(expression_dim, 256)
            self.expr_norm = nn.BatchNorm1d(256)
            self.expr_fc2 = nn.Linear(256, 100)
            n_branches += 1
        if use_methylation:
            self.meth_fc1 = nn.Linear(methylation_dim, 256)
            self.meth_norm = nn.BatchNorm1d(256)
            self.meth_fc2 = nn.Linear(256, 100)
            n_branches += 1

        # --- fusion: 300-d -> Conv2d stack -> scalar ---
        self.fusion_fc = nn.Linear(100 * n_branches, 300)
        self.fuse_conv1 = nn.Conv2d(1, 30, kernel_size=(1, 150))
        self.fuse_pool1 = nn.MaxPool2d((1, 2))
        self.fuse_conv2 = nn.Conv2d(30, 10, kernel_size=(1, 5))
        self.fuse_pool2 = nn.MaxPool2d((1, 3))
        self.fuse_conv3 = nn.Conv2d(10, 5, kernel_size=(1, 5))
        self.fuse_pool3 = nn.MaxPool2d((1, 3))
        with torch.no_grad():
            probe = torch.zeros(1, 1, 1, 300)
            probe = self.fuse_pool1(F.relu(self.fuse_conv1(probe)))
            probe = self.fuse_pool2(F.relu(self.fuse_conv2(probe)))
            probe = self.fuse_pool3(F.relu(self.fuse_conv3(probe)))
            fuse_flat = probe.numel()
        self.head = nn.Linear(fuse_flat, 1)

    def forward(
        self,
        atom_features: torch.Tensor,
        adjacency: torch.Tensor,
        mutation: torch.Tensor | None = None,
        expression: torch.Tensor | None = None,
        methylation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = atom_features
        for layer, norm in zip(self.gcn_layers, self.gcn_norms):
            h = torch.relu(layer(h, adjacency))
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = self.dropout(h)
        h = torch.relu(self.gcn_final(h, adjacency))
        h = self.gcn_final_norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(h)
        drug = h.max(dim=1).values  # global max pool -> [B, 100]

        parts = [drug]
        if self.use_mutation:
            x = mutation.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, D]
            x = self.mut_pool1(torch.tanh(self.mut_conv1(x)))
            x = self.mut_pool2(F.relu(self.mut_conv2(x)))
            parts.append(self.dropout(F.relu(self.mut_fc(x.flatten(1)))))
        if self.use_expression:
            x = self.dropout(self.expr_norm(torch.tanh(self.expr_fc1(expression))))
            parts.append(F.relu(self.expr_fc2(x)))
        if self.use_methylation:
            x = self.dropout(self.meth_norm(torch.tanh(self.meth_fc1(methylation))))
            parts.append(F.relu(self.meth_fc2(x)))

        fused = self.dropout(torch.tanh(self.fusion_fc(torch.cat(parts, dim=-1))))
        x = fused.unsqueeze(1).unsqueeze(1)
        x = self.fuse_pool1(F.relu(self.fuse_conv1(x)))
        x = self.fuse_pool2(F.relu(self.fuse_conv2(x)))
        x = self.fuse_pool3(F.relu(self.fuse_conv3(x)))
        x = nn.functional.dropout(x.flatten(1), 0.2, self.training)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# DrugCell (Kuenzi et al. 2020)
# ---------------------------------------------------------------------------

class DrugCell(nn.Module):
    """Gene-Ontology-structured visible neural network plus a drug MLP.

    Connectivity mirrors the ontology: each GO term gets ``neurons_per_term``
    units whose input is the concatenation of its child terms' outputs and its
    directly annotated genes. Terms are evaluated in topological order from
    leaves to root. Every term also carries an auxiliary linear head, as in the
    reference implementation, which the original uses for interpretation and
    auxiliary loss.
    """

    def __init__(
        self,
        term_children: Dict[str, List[str]],
        term_genes: Dict[str, List[int]],
        term_order: List[str],
        root: str,
        n_genes: int,
        drug_dim: int = 2048,
        neurons_per_term: int = 6,
        drug_hidden: Sequence[int] = (100, 50, 6),
        final_hidden: int = 6,
    ):
        super().__init__()
        self.term_children = term_children
        self.term_genes = term_genes
        self.term_order = term_order
        self.root = root
        self.n_genes = n_genes
        self.neurons_per_term = neurons_per_term

        self.term_linear = nn.ModuleDict()
        self.term_norm = nn.ModuleDict()
        self.term_aux = nn.ModuleDict()
        for term in term_order:
            fan_in = neurons_per_term * len(term_children[term]) + len(term_genes[term])
            if fan_in == 0:
                raise ValueError(f"GO term {term} has neither children nor genes")
            key = self._key(term)
            self.term_linear[key] = nn.Linear(fan_in, neurons_per_term)
            self.term_norm[key] = nn.BatchNorm1d(neurons_per_term)
            self.term_aux[key] = nn.Linear(neurons_per_term, 1)

        layers, previous = [], drug_dim
        for width in drug_hidden:
            layers += [nn.Linear(previous, width), nn.Tanh(), nn.BatchNorm1d(width)]
            previous = width
        self.drug_mlp = nn.Sequential(*layers)

        self.final = nn.Sequential(
            nn.Linear(neurons_per_term + drug_hidden[-1], final_hidden),
            nn.Tanh(),
            nn.BatchNorm1d(final_hidden),
        )
        self.head = nn.Linear(final_hidden, 1)

    @staticmethod
    def _key(term: str) -> str:
        # ModuleDict keys cannot contain "."; GO ids use ":" which is also unsafe.
        return term.replace(":", "_").replace(".", "_")

    def forward(self, mutation: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        outputs: Dict[str, torch.Tensor] = {}
        for term in self.term_order:
            key = self._key(term)
            parts = [outputs[child] for child in self.term_children[term]]
            genes = self.term_genes[term]
            if genes:
                parts.append(mutation[:, genes])
            hidden = torch.tanh(self.term_linear[key](torch.cat(parts, dim=-1)))
            outputs[term] = self.term_norm[key](hidden)

        combined = torch.cat([outputs[self.root], self.drug_mlp(drug)], dim=-1)
        return self.head(self.final(combined)).squeeze(-1)


# ---------------------------------------------------------------------------
# tCNNS (Liu et al. 2019) - faithful drug branch, RNA-embedding cell branch
# ---------------------------------------------------------------------------

class TCNNSHybrid(nn.Module):
    """Twin 1-D CNN. Drug branch faithful; cell branch runs on our RNA embedding.

    The drug branch reproduces the published network exactly: three Conv1d layers
    with kernel width 7 and 40 -> 80 -> 60 channels, each followed by ReLU and
    max-pooling of width 3, over a one-hot SMILES matrix.

    The cell branch keeps the published *architecture* (identical Conv1d stack
    over a 1-channel vector) but consumes our 256-d BulkRNABERT embedding rather
    than the original 735-dim binary mutation/CNV panel, which is distributed only
    through an inaccessible mirror. This is a documented deviation: the drug-side
    representation, which is what the comparison is about, is exact; the cell-side
    input differs and must be stated as such in Methods.

    The published model applies a sigmoid because it regresses IC50 normalised to
    [0, 1]. Our targets are LN(IC50) on an unbounded scale, so ``output_activation``
    defaults to identity; set it to "sigmoid" only alongside normalised targets.
    """

    def __init__(
        self,
        n_smiles_chars: int,
        smiles_length: int,
        cell_dim: int = 256,
        dropout: float = 0.5,
        output_activation: str = "identity",
    ):
        super().__init__()
        self.output_activation = output_activation

        def branch(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_channels, 40, kernel_size=7, padding="same"), nn.ReLU(),
                nn.MaxPool1d(3, stride=3),
                nn.Conv1d(40, 80, kernel_size=7, padding="same"), nn.ReLU(),
                nn.MaxPool1d(3, stride=3),
                nn.Conv1d(80, 60, kernel_size=7, padding="same"), nn.ReLU(),
                nn.MaxPool1d(3, stride=3),
            )

        # Three stride-3 pools shrink the sequence by ~27x, so both inputs must be
        # long enough to survive them. Fail here with an explanation rather than
        # deep inside max_pool1d with "Invalid computed output size: 0".
        for label, size in (("smiles_length", smiles_length), ("cell_dim", cell_dim)):
            if size < 27:
                raise ValueError(
                    f"{label}={size} is too short for tCNNS: three max-pools of "
                    "stride 3 require at least 27 positions."
                )

        self.drug_branch = branch(n_smiles_chars)
        self.cell_branch = branch(1)

        with torch.no_grad():
            drug_flat = self.drug_branch(torch.zeros(1, n_smiles_chars, smiles_length)).numel()
            cell_flat = self.cell_branch(torch.zeros(1, 1, cell_dim)).numel()

        self.fc = nn.Sequential(
            nn.Linear(drug_flat + cell_flat, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )

    def forward(self, smiles_onehot: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        # smiles_onehot arrives as [B, length, chars]; Conv1d wants channels first.
        drug = self.drug_branch(smiles_onehot.transpose(1, 2)).flatten(1)
        cell = self.cell_branch(cell.unsqueeze(1)).flatten(1)
        out = self.fc(torch.cat([drug, cell], dim=-1)).squeeze(-1)
        return torch.sigmoid(out) if self.output_activation == "sigmoid" else out
