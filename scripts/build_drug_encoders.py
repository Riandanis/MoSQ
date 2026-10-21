#!/usr/bin/env python3
"""Generate alternative drug representations for the NCD encoder sweep.

Generalisation to unseen drugs (NCD) is weak, and the drug embedding is the obvious
bottleneck to test. The shipped representation is a 768-d ChemBERTa-2 vector; this script produces the
alternatives so the bottleneck can be tested rather than asserted.

Encoders:
  chemberta   768-d, the shipped representation (copied through as the control)
  morgan      2048-bit ECFP4 count fingerprint -- the classic strong DRP baseline
  descriptors ~200 RDKit physicochemical descriptors -- cheap and interpretable,
              and a direct probe of whether NCD performance merely tracks size
              and lipophilicity
  chemberta_mtr
              ChemBERTa-77M-MTR: a chemical language model pretrained with
              multi-task regression on physicochemical properties, so its
              objective differs from the shipped ChemBERTa-2 rather than merely
              its weights

MoLFormer-XL was the first choice but its ``trust_remote_code`` module requires a
transformers API that this environment's 4.56.2 does not expose
(``create_bidirectional_mask``), and upgrading transformers mid-campaign would risk
the runs in flight. ChemBERTa-77M-MTR serves the same purpose: a second neural
chemical encoder with a genuinely different pretraining objective.

Deliberately no graph neural network: torch_geometric is installed but broken
against this torch build (undefined symbol on torch-scatter/torch-sparse import),
and repairing it would risk the environment the whole campaign runs in.

SMILES are not in this repo. They come from the CCLE ChemBERTa feature table in
the sibling working tree, joined on drug name. Coverage is asserted, not assumed.

Output: ``data/drug_embeddings_<encoder>.csv`` with columns ``drug_id,emb_0..emb_{d-1}``,
matching the format ``DrugEmbeddingDataset`` expects, plus ``data/drug_smiles.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Only needed to regenerate data/drug_smiles.csv, which is shipped with the repo.
SMILES_SOURCE = ROOT / "data/external/CCLE_chembert_features_output_v260120.csv"
IC50_CSV = ROOT / "data/ic50_data.csv"
DRUG_CSV = ROOT / "data/drug_embeddings.csv"


def build_smiles_table(out_path: Path) -> pd.DataFrame:
    """Join drug ids to SMILES, asserting full coverage of the drugs actually used."""
    if not SMILES_SOURCE.exists():
        raise FileNotFoundError(
            f"SMILES source not found: {SMILES_SOURCE}\n"
            "data/drug_smiles.csv is shipped with the repo; this source is only needed "
            "to regenerate it."
        )

    source = pd.read_csv(SMILES_SOURCE, usecols=["drug_name", "smiles", "smiles_type", "source"])
    source["key"] = source.drug_name.astype(str).str.strip().str.lower()
    source = source.dropna(subset=["smiles"]).drop_duplicates("key")

    needed = pd.read_csv(IC50_CSV, usecols=["drug_id"]).drug_id.astype(str).unique()
    frame = pd.DataFrame({"drug_id": needed})
    frame["key"] = frame.drug_id.str.strip().str.lower()
    frame = frame.merge(source[["key", "smiles", "smiles_type"]], on="key", how="left")

    missing = frame[frame.smiles.isna()]
    if len(missing):
        raise ValueError(
            f"No SMILES for {len(missing)} of {len(frame)} drugs, "
            f"e.g. {missing.drug_id.head().tolist()}"
        )

    frame = frame.drop(columns=["key"])
    frame.to_csv(out_path, index=False)
    logger.info(f"SMILES resolved for all {len(frame)} drugs -> {out_path}")
    return frame


def _write(frame: pd.DataFrame, features: np.ndarray, name: str) -> Path:
    out = pd.DataFrame(features, columns=[f"emb_{i}" for i in range(features.shape[1])])
    out.insert(0, "drug_id", frame.drug_id.values)
    path = ROOT / f"data/drug_embeddings_{name}.csv"
    out.to_csv(path, index=False)
    logger.info(f"{name}: {features.shape[1]}-d for {len(out)} drugs -> {path}")
    return path


def encode_morgan(frame: pd.DataFrame, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    features = np.zeros((len(frame), n_bits), dtype=np.float32)
    for i, smiles in enumerate(frame.smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"unparseable SMILES for {frame.drug_id.iloc[i]}, leaving zero vector")
            continue
        features[i] = np.asarray(generator.GetCountFingerprintAsNumPy(mol), dtype=np.float32)
    return features


def encode_descriptors(frame: pd.DataFrame) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    RDLogger.DisableLog("rdApp.*")
    names = [name for name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    features = np.zeros((len(frame), len(names)), dtype=np.float64)
    for i, smiles in enumerate(frame.smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        features[i] = np.asarray(calculator.CalcDescriptors(mol), dtype=np.float64)

    # Descriptors span wildly different scales and a few can be inf; standardise
    # here so the downstream projector sees something comparable to the others.
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = features.mean(axis=0), features.std(axis=0)
    features = (features - mean) / np.where(std > 0, std, 1.0)
    return features.astype(np.float32)


def encode_chemberta_mtr(frame: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    chunks = []
    smiles = frame.smiles.tolist()
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = tokenizer(
                smiles[start:start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            output = model(**batch)
            # Mean-pool over real tokens; the pooler head is randomly initialised
            # in this checkpoint and must not be used.
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


def encode_chemberta(frame: pd.DataFrame) -> np.ndarray:
    """Control arm: the shipped vectors, reordered to match the SMILES table."""
    shipped = pd.read_csv(DRUG_CSV)
    emb_cols = [c for c in shipped.columns if c.startswith("emb_")]
    lookup = shipped.set_index("drug_id")[emb_cols]
    missing = set(frame.drug_id) - set(lookup.index)
    if missing:
        raise ValueError(f"shipped embeddings missing {len(missing)} drugs: {sorted(missing)[:5]}")
    return lookup.loc[frame.drug_id].to_numpy(dtype=np.float32)


ENCODERS = {
    "chemberta": encode_chemberta,
    "morgan": encode_morgan,
    "descriptors": encode_descriptors,
    "chemberta_mtr": encode_chemberta_mtr,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=["morgan", "descriptors"],
                        choices=list(ENCODERS))
    args = parser.parse_args()

    smiles_path = ROOT / "data/drug_smiles.csv"
    frame = pd.read_csv(smiles_path) if smiles_path.exists() else build_smiles_table(smiles_path)

    for name in args.encoders:
        logger.info(f"Encoding with {name}...")
        features = ENCODERS[name](frame)
        _write(frame, features, name)


if __name__ == "__main__":
    main()
