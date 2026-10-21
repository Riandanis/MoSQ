#!/usr/bin/env python3
"""Cross-validated benchmark of the faithful comparator reimplementations.

Runs DeepCDR, DrugCell and the tCNNS hybrid on the same folds, seeds and metrics
as everything else in the revision, and writes per-sample prediction dumps so the
downstream analyses treat them as ordinary arms.

Unlike the ``*_style`` arms in ``benchmark_literature_models.py``, these consume
the inputs their papers used — molecular graphs and three omics for DeepCDR, a
Gene-Ontology visible network over binary mutations for DrugCell, one-hot SMILES
for tCNNS — so the comparison controls for data and protocol without holding the
input representation fixed at ours.

Cell-line coverage differs per model and dictates the subset:
  deepcdr   544 cell lines  (its three-omics roster is the binding constraint)
  drugcell  963 cell lines
  tcnns     998 cell lines  (needs no external cell data)

Use ``--subset common`` to force every arm onto the 544-line intersection, which
is what makes a four-way table like-for-like. MoSQ must then be re-run on the
same subset via ``run_cv.py --cellline_subset``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gastro_transformer import splits  # noqa: E402
from gastro_transformer.comparators import DeepCDR, DrugCell, TCNNSHybrid  # noqa: E402
from gastro_transformer.comparators.data import (  # noqa: E402
    common_celllines,
    load_deepcdr_omics,
    load_depmap_id_map,
    load_drugcell_mutations,
    load_drugcell_ontology,
    normalise_cellline,
)
from gastro_transformer.comparators.featurize import molecular_graphs, smiles_onehot  # noqa: E402
from gastro_transformer.metrics import compute_regression_metrics  # noqa: E402
from gastro_transformer.predlog import append_manifest, dump_predictions  # noqa: E402
from gastro_transformer.seeding import seed_worker, set_all_seeds, torch_generator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ARMS = ("deepcdr", "drugcell", "tcnns")


class PairDataset(Dataset):
    """Rows of (cell line, drug, LN IC50) with per-arm inputs pre-materialised.

    Everything is indexed out of dense arrays rather than rebuilt per item: there
    are only a few hundred drugs and cell lines, so the full feature tensors fit
    comfortably in memory and the loader stays cheap.
    """

    def __init__(self, frame: pd.DataFrame, drug_feats: dict, cell_feats: dict):
        self.y = frame.ic50_value.to_numpy(dtype=np.float32)
        self.drug_idx = frame.drug_row.to_numpy(dtype=np.int64)
        self.cell_idx = frame.cell_row.to_numpy(dtype=np.int64)
        self.drug_feats = drug_feats
        self.cell_feats = cell_feats
        self.cellline_ids = frame.cellline_id.astype(str).tolist()
        self.drug_ids = frame.drug_id.astype(str).tolist()
        # Fold construction keys off this, exactly as IC50Dataset does.
        self.cellline_to_idx = {c: i for i, c in enumerate(sorted(set(self.cellline_ids)))}
        self.num_celllines = len(self.cellline_to_idx)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        d, c = self.drug_idx[i], self.cell_idx[i]
        item = {"ic50": torch.tensor(self.y[i])}
        for key, array in self.drug_feats.items():
            item[key] = torch.from_numpy(array[d])
        for key, array in self.cell_feats.items():
            item[key] = torch.from_numpy(array[c])
        return item


def build_inputs(arm: str, frame: pd.DataFrame, args):
    """Assemble (dataset, model_factory) for one arm."""
    smiles = pd.read_csv(ROOT / "data/drug_smiles.csv")
    smiles_by_drug = dict(zip(smiles.drug_id.astype(str), smiles.smiles))
    smiles_by_drug = {d: s for d, s in smiles_by_drug.items() if d in set(frame.drug_id)}

    cellline_names = sorted(frame.cellline_id.astype(str).unique())
    cell_row = {c: i for i, c in enumerate(cellline_names)}

    if arm == "tcnns":
        onehot, drug_ids, chars, length = smiles_onehot(smiles_by_drug)
        drug_row = {d: i for i, d in enumerate(drug_ids)}
        rna = pd.read_csv(ROOT / "data/ccle_rna_for_ic50.csv", index_col=0)
        emb = np.zeros((len(cellline_names), rna.shape[1]), dtype=np.float32)
        for name, i in cell_row.items():
            if name in rna.index:
                emb[i] = rna.loc[name].to_numpy(dtype=np.float32)
        drug_feats = {"smiles_onehot": onehot}
        cell_feats = {"cell": emb}

        def factory():
            return TCNNSHybrid(
                n_smiles_chars=len(chars), smiles_length=length, cell_dim=emb.shape[1]
            )

    elif arm == "drugcell":
        morgan = pd.read_csv(ROOT / "data/drug_embeddings_morgan.csv", index_col=0)
        morgan = morgan.loc[sorted(set(frame.drug_id))]
        drug_row = {d: i for i, d in enumerate(morgan.index.astype(str))}
        fp = morgan.to_numpy(dtype=np.float32)

        matrix, name_to_row = load_drugcell_mutations()
        mut = np.zeros((len(cellline_names), matrix.shape[1]), dtype=np.float32)
        for name, i in cell_row.items():
            key = normalise_cellline(name)
            if key in name_to_row:
                mut[i] = matrix[name_to_row[key]]
        children, genes, order, root, gene_index = load_drugcell_ontology()
        drug_feats = {"drug": fp}
        cell_feats = {"mutation": mut}

        def factory():
            return DrugCell(
                children, genes, order, root, n_genes=len(gene_index), drug_dim=fp.shape[1]
            )

    elif arm == "deepcdr":
        features, adjacency, drug_ids = molecular_graphs(smiles_by_drug)
        drug_row = {d: i for i, d in enumerate(drug_ids)}

        if args.cell_source == "mosq_rna":
            # Supplementary arm: DeepCDR's architecture on MoSQ's cell representation,
            # which isolates architecture from cell-line representation. The expression
            # branch is Linear(expression_dim, 256), so this needs no code change --
            # only the mutation and methylation branches are switched off.
            rna = pd.read_csv(ROOT / "data/ccle_rna_for_ic50.csv", index_col=0)
            block = np.zeros((len(cellline_names), rna.shape[1]), dtype=np.float32)
            for name, i in cell_row.items():
                if name in rna.index:
                    block[i] = rna.loc[name].to_numpy(dtype=np.float32)
            drug_feats = {"atom_features": features, "adjacency": adjacency}
            cell_feats = {"expression": block}

            def factory():
                return DeepCDR(
                    atom_dim=features.shape[2], expression_dim=block.shape[1],
                    use_mutation=False, use_methylation=False,
                )

            frame = frame.copy()
            frame["drug_row"] = frame.drug_id.astype(str).map(drug_row)
            frame["cell_row"] = frame.cellline_id.astype(str).map(cell_row)
            frame = frame.dropna(subset=["drug_row", "cell_row"])
            return PairDataset(frame, drug_feats, cell_feats), factory, frame

        omics = load_deepcdr_omics(("expression", "methylation", "mutation"))
        ach = load_depmap_id_map()

        cells = {}
        for key, frame_omics in omics.items():
            block = np.zeros((len(cellline_names), frame_omics.shape[1]), dtype=np.float32)
            for name, i in cell_row.items():
                a = ach.get(normalise_cellline(name))
                if a in frame_omics.index:
                    block[i] = frame_omics.loc[a].to_numpy(dtype=np.float32)
            cells[key] = block
        drug_feats = {"atom_features": features, "adjacency": adjacency}
        cell_feats = cells

        def factory():
            return DeepCDR(
                atom_dim=features.shape[2],
                mutation_dim=cells["mutation"].shape[1],
                expression_dim=cells["expression"].shape[1],
                methylation_dim=cells["methylation"].shape[1],
            )
    else:
        raise ValueError(arm)

    frame = frame.copy()
    frame["drug_row"] = frame.drug_id.astype(str).map(drug_row)
    frame["cell_row"] = frame.cellline_id.astype(str).map(cell_row)
    frame = frame.dropna(subset=["drug_row", "cell_row"])
    return PairDataset(frame, drug_feats, cell_feats), factory, frame


def forward(arm: str, model, batch, device):
    if arm == "deepcdr" and "mutation" not in batch:
        return model(batch["atom_features"].to(device), batch["adjacency"].to(device),
                     expression=batch["expression"].to(device))
    if arm == "tcnns":
        return model(batch["smiles_onehot"].to(device), batch["cell"].to(device))
    if arm == "drugcell":
        return model(batch["mutation"].to(device), batch["drug"].to(device))
    return model(
        batch["atom_features"].to(device), batch["adjacency"].to(device),
        batch["mutation"].to(device), batch["expression"].to(device),
        batch["methylation"].to(device),
    )


def train_and_eval(arm, factory, dataset, train_idx, val_idx, test_idx, args, device):
    model = factory().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    common = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,
        worker_init_fn=seed_worker, generator=torch_generator(args.seed), **common,
    )
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, **common)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=False, **common)

    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(forward(arm, model, batch, device),
                                          batch["ic50"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred = forward(arm, model, batch, device)
                total += nn.functional.mse_loss(
                    pred, batch["ic50"].to(device), reduction="sum"
                ).item()
                n += len(pred)
        val_loss = total / max(n, 1)
        if val_loss < best_loss - 1e-5:
            best_loss, stale = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            preds.append(forward(arm, model, batch, device).cpu().numpy())
            targets.append(batch["ic50"].numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--regime", default="ncc", choices=list(splits.REGIMES))
    parser.add_argument("--subset", default="common", choices=["common", "native", "full"],
                        help="'common' = the 544-line intersection (like-for-like); "
                             "'native' = whatever this arm can cover; 'full' = all 998")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--cell_source", default="native", choices=["native", "mosq_rna"],
                        help="'native' uses the model's published cell-line inputs; "
                             "'mosq_rna' substitutes MoSQ's BulkRNABERT embedding for the "
                             "matched-input supplementary analysis (DeepCDR only)")
    parser.add_argument("--experiment", default="comparators")
    parser.add_argument("--out_root", default=str(ROOT / "reports/revision"))
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    result_path = out_root / args.experiment / f"{args.regime}__{args.arm}__seed{args.seed}.json"
    if args.skip_existing and result_path.exists():
        logger.info(f"{result_path.name} exists, skipping")
        return

    set_all_seeds(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    started = time.time()

    ic50 = pd.read_csv(ROOT / "data/ic50_data.csv")
    known = set(pd.read_csv(ROOT / "data/drug_embeddings.csv", usecols=["drug_id"]).drug_id)
    ic50 = ic50[ic50.drug_id.isin(known)].drop_duplicates(subset=["cellline_id", "drug_id"])

    if args.subset == "common":
        keep, _ = common_celllines(ic50.cellline_id.astype(str).unique(),
                                   require=("deepcdr", "drugcell"))
        ic50 = ic50[ic50.cellline_id.astype(str).isin(keep)]
    elif args.subset == "native":
        require = {"deepcdr": ("deepcdr",), "drugcell": ("drugcell",), "tcnns": ()}[args.arm]
        if require:
            keep, _ = common_celllines(ic50.cellline_id.astype(str).unique(), require=require)
            ic50 = ic50[ic50.cellline_id.astype(str).isin(keep)]
    ic50 = ic50.reset_index(drop=True)
    logger.info(
        f"{args.arm} / subset={args.subset}: {len(ic50):,} pairs, "
        f"{ic50.cellline_id.nunique()} cell lines, {ic50.drug_id.nunique()} drugs"
    )

    dataset, factory, frame = build_inputs(args.arm, ic50, args)
    sample_fold = splits.build_folds(args.regime, dataset, n_folds=args.n_folds, seed=args.seed)

    folds = args.folds if args.folds is not None else list(range(args.n_folds))
    fold_metrics = []
    for fold in folds:
        set_all_seeds(args.seed * 1000 + fold)
        train_idx, val_idx, test_idx = splits.get_fold_splits(sample_fold, fold, seed=args.seed)
        logger.info(f"  fold {fold}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

        preds, targets = train_and_eval(
            args.arm, factory, dataset, train_idx, val_idx, test_idx, args, device
        )
        metrics = compute_regression_metrics(targets, preds)
        fold_metrics.append(metrics)

        rows = frame.iloc[test_idx]
        dump = pd.DataFrame({
            "experiment": args.experiment, "regime": args.regime, "arm": args.arm,
            "seed": np.int16(args.seed), "fold": np.int16(fold), "split": "test",
            "row_index": np.asarray(test_idx, dtype=np.int32),
            "cellline_id": rows.cellline_id.astype(str).to_numpy(),
            "drug_id": rows.drug_id.astype(str).to_numpy(),
            "has_rna": True,
            "y_true": targets.astype(np.float32), "y_pred": preds.astype(np.float32),
        })
        pred_dir = out_root / args.experiment / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        dump.to_parquet(
            pred_dir / f"{args.regime}__{args.arm}__seed{args.seed}__fold{fold}__test.parquet",
            index=False,
        )
        logger.info(f"  fold {fold}: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}")

    average = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    average.update({f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]})
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "arm": args.arm, "regime": args.regime, "seed": args.seed, "subset": args.subset,
        "n_pairs": len(ic50), "n_celllines": int(ic50.cellline_id.nunique()),
        "fold_metrics": fold_metrics, "average": average, "config": vars(args),
    }, indent=2, default=str))

    elapsed = time.time() - started
    append_manifest(out_root, {
        "experiment": args.experiment, "regime": args.regime, "arm": args.arm,
        "seed": args.seed, "subset": args.subset, "r2_mean": average["r2"],
        "wall_clock_s": round(elapsed, 1), "args": vars(args),
    })
    logger.info(
        f"=== {args.arm} {args.regime} seed{args.seed}: "
        f"R2={average['r2']:.4f}+-{average['r2_std']:.4f} in {elapsed/60:.1f} min ==="
    )


if __name__ == "__main__":
    main()
