#!/usr/bin/env python3
"""Pre-registered downstream case: drug mechanism (GDSC TARGET_PATHWAY) from per-drug
representations. See the pre-registration record, "Pre-registered downstream case — 2026-10-04".

  python scripts/novelty/downstream_moa.py extract   # GPU: DRA / RAND / CLRNA per-drug vectors
  python scripts/novelty/downstream_moa.py score     # CPU: classification + retrieval + bootstrap
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/novelty/downstream"
OUT.mkdir(parents=True, exist_ok=True)


def labels():
    sc = pd.read_csv(ROOT / "data/external/gdsc/screened_compounds_rel_8.5.csv")
    key = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    names = {}
    for _, r in sc.iterrows():
        syn = [x for x in str(r.SYNONYMS).split(",") if x.strip()] if isinstance(r.SYNONYMS, str) else []
        for n in [r.DRUG_NAME] + syn:
            names.setdefault(key(n), set()).add(r.TARGET_PATHWAY)
    drugs = sorted(pd.read_csv(ROOT / "data/ic50_data.csv", usecols=["drug_id"]).drug_id.astype(str).unique())
    lab = {d: names.get(key(d), set()) for d in drugs}
    lab = pd.Series({d: next(iter(v)) for d, v in lab.items() if len(v) == 1})
    lab = lab[~lab.isin(["Other", "Other, kinases"])]
    counts = lab.value_counts()
    return lab[lab.isin(counts[counts >= 5].index)]


def extract():
    spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts/benchmark_transfer.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    from gastro_transformer.data import DrugEmbeddingDataset, IC50Dataset
    ds = IC50Dataset(str(ROOT / "data/ic50_data.csv"), DrugEmbeddingDataset(str(ROOT / "data/drug_embeddings.csv"), drug_dim=768),
                     rna_csv_path=str(ROOT / "data/ccle_rna_for_ic50.csv"), rna_dim=256, add_tissue_ids=True)

    class A:  # build_encoder reads checkpoint paths from these attributes
        clrna_ckpt = str(ROOT / "saved_checkpoints/pretrained_clrna.pt")
        dra_ckpt = str(ROOT / "saved_checkpoints/pretrained_dra.pt")
        recon_ckpt = None

    drug = np.asarray(ds.drug_ids, dtype=object)
    idx = np.arange(len(ds))
    reps = {}
    for arm in ["dra", "randinit", "clrna"]:
        torch.manual_seed(0); np.random.seed(0)
        enc = tr.build_encoder(arm, A, ds, "cuda:1")
        X, _ = tr.extract_queries(enc, ds, idx, "cuda:1")
        reps[arm] = pd.DataFrame(X).groupby(drug).mean()
        print(arm, reps[arm].shape, flush=True)
        del enc; torch.cuda.empty_cache()
    for k, v in reps.items():
        v.to_parquet(OUT / f"drug_rep_{k}.parquet")


def raw_profiles():
    ic = pd.read_csv(ROOT / "data/ic50_data.csv", dtype={"cellline_id": str, "drug_id": str})
    ic = ic.drop_duplicates(["cellline_id", "drug_id"], keep="first")
    M = ic.pivot(index="drug_id", columns="cellline_id", values="ic50_value")
    Z = M.sub(M.mean(1), axis=0).div(M.std(1), axis=0).fillna(0.0)
    U = M.apply(lambda r: r.fillna(r.mean()), axis=1)
    return Z, U


def score():
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    lab = labels()
    Z, U = raw_profiles()
    struct = pd.read_csv(ROOT / "data/drug_embeddings.csv").set_index("drug_id")
    arms = {"DRA": pd.read_parquet(OUT / "drug_rep_dra.parquet"), "RAND": pd.read_parquet(OUT / "drug_rep_randinit.parquet"),
            "CLRNA": pd.read_parquet(OUT / "drug_rep_clrna.parquet"), "RAW": Z, "RAWU": U, "STRUCT": struct}
    drugs = lab.index.to_numpy()
    y = lab.to_numpy()
    classes = np.unique(y)
    print(f"{len(drugs)} drugs, {len(classes)} classes; chance balanced accuracy {1/len(classes):.3f}", flush=True)
    correct = {}  # arm -> [n_drugs, n_repeats] boolean
    preds = {}
    for arm, F in arms.items():
        X = F.loc[drugs].to_numpy(float)
        C = np.zeros((len(drugs), 10), bool); P = np.empty((len(drugs), 10), dtype=object)
        for r in range(10):
            for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
                clf = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                                   {"logisticregression__C": [0.01, 0.1, 1, 10]},
                                   cv=StratifiedKFold(4, shuffle=True, random_state=100 + r), scoring="balanced_accuracy")
                clf.fit(X[tr_i], y[tr_i])
                P[te_i, r] = clf.predict(X[te_i])
            C[:, r] = P[:, r] == y
        correct[arm], preds[arm] = C, P
        print(f"{arm} done", flush=True)

    def ba(Cm, idx=None):
        idx = np.arange(len(y)) if idx is None else idx
        yy = y[idx]
        return np.mean([np.mean([Cm[idx][yy == c, r].mean() for c in classes]) for r in range(Cm.shape[1])])

    def nn1(F):
        X = F.loc[drugs].to_numpy(float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        S = X @ X.T; np.fill_diagonal(S, -np.inf)
        return y[S.argmax(1)] == y

    rows = []
    for arm in arms:
        f1 = np.mean([f1_score(y, preds[arm][:, r].astype(str), average="macro") for r in range(10)])
        rows.append(dict(arm=arm, balanced_accuracy=ba(correct[arm]), macro_f1=f1, nn1_retrieval=nn1(arms[arm]).mean()))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "moa_summary.csv", index=False)
    print(summ.round(4).to_string(index=False))

    rng = np.random.default_rng(0)
    by_class = [np.flatnonzero(y == c) for c in classes]
    boots = [np.concatenate([rng.choice(ix, len(ix)) for ix in by_class]) for _ in range(2000)]
    nn = {a: nn1(F) for a, F in arms.items()}
    deltas = []
    for a, b in [("DRA", "RAW"), ("DRA", "RAWU"), ("DRA", "RAND"), ("DRA", "CLRNA"), ("DRA", "STRUCT"),
                 ("RAW", "STRUCT"), ("RAND", "RAW")]:
        d_ba = ba(correct[a]) - ba(correct[b])
        bb = np.array([ba(correct[a], ix) - ba(correct[b], ix) for ix in boots])
        d_nn = nn[a].mean() - nn[b].mean()
        bn = np.array([nn[a][ix].mean() - nn[b][ix].mean() for ix in boots])
        deltas.append(dict(comparison=f"{a} - {b}", d_balanced_accuracy=d_ba, ba_lo=np.percentile(bb, 2.5),
                           ba_hi=np.percentile(bb, 97.5), d_nn1=d_nn, nn1_lo=np.percentile(bn, 2.5), nn1_hi=np.percentile(bn, 97.5)))
    dt = pd.DataFrame(deltas)
    dt.to_csv(OUT / "moa_deltas.csv", index=False)
    print(dt.round(4).to_string(index=False))
    prim = dt.iloc[0]
    branch = "POSITIVE" if prim.ba_lo > 0 else "NEGATIVE" if prim.ba_hi < 0 else "NULL"
    print(f"\nPRIMARY DRA - RAW balanced accuracy {prim.d_balanced_accuracy:+.4f} "
          f"[{prim.ba_lo:+.4f}, {prim.ba_hi:+.4f}] -> {branch}")


if __name__ == "__main__":
    {"extract": extract, "score": score}[sys.argv[1]]()
