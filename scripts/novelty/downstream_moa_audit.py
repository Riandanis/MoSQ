#!/usr/bin/env python3
"""EXPLORATORY robustness audit of the pre-registered MoA case (not part of the pre-registration).
Checks: per-repeat stability, other classifiers, larger classes only, and an alternative DR-A
representation (the DR-A model's predicted IC50 profile, from the transductive multiseed dumps).
Output: reports/novelty/downstream/audit_robustness.csv
"""
import glob, importlib.util, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location("m", ROOT / "scripts/novelty/downstream_moa.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
D = ROOT / "reports/novelty/downstream"

lab = m.labels(); Z, U = m.raw_profiles()
# DR-A predicted profile: every pair is predicted once (when its cell line is in the test fold), seed 0
pr = pd.concat(pd.read_parquet(f) for f in glob.glob(str(ROOT / "reports/revision/multiseed/preds/ncc__multitoken_dra__seed0__fold*__test.parquet")))
P = pr.pivot_table(index="drug_id", columns="cellline_id", values="y_pred")
PZ = P.sub(P.mean(1), axis=0).div(P.std(1), axis=0).fillna(0.0)
pr0 = pd.concat(pd.read_parquet(f) for f in glob.glob(str(ROOT / "reports/revision/multiseed/preds/ncc__qformer_randinit__seed0__fold*__test.parquet")))
P0 = pr0.pivot_table(index="drug_id", columns="cellline_id", values="y_pred")
P0Z = P0.sub(P0.mean(1), axis=0).div(P0.std(1), axis=0).fillna(0.0)
arms = {"DRA": pd.read_parquet(D / "drug_rep_dra.parquet"), "RAW": Z,
        "DRA_pred_profile": PZ, "RANDINIT_pred_profile": P0Z}
drugs, y = lab.index.to_numpy(), lab.to_numpy()
clfs = {"logreg_C1": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000)),
        "linear_svm": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.1, max_iter=20000)),
        "knn5_cosine": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric="cosine"))}
rows = []
for subset, keep in [("all 15 classes", np.ones(len(y), bool)),
                     ("classes with >= 8 drugs", pd.Series(y).map(pd.Series(y).value_counts()).to_numpy() >= 8)]:
    for arm, F in arms.items():
        X = F.loc[drugs].to_numpy(float)[keep]; yy = y[keep]
        for cn, mk in clfs.items():
            bas = [balanced_accuracy_score(yy, cross_val_predict(mk(), X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=r)))
                   for r in range(10)]
            rows.append(dict(subset=subset, n_classes=len(np.unique(yy)), arm=arm, classifier=cn,
                             ba_mean=np.mean(bas), ba_min=np.min(bas), ba_max=np.max(bas), per_repeat=np.round(bas, 3).tolist()))
        print(subset, arm, "done", flush=True)
t = pd.DataFrame(rows); t.to_csv(D / "audit_robustness.csv", index=False)
p = t.pivot_table(index=["subset", "classifier"], columns="arm", values="ba_mean").round(3)
print(p.to_string())
# per-repeat dominance DRA vs RAW
for (s, c), g in t.groupby(["subset", "classifier"]):
    a = np.array(g[g.arm == "DRA"].per_repeat.iloc[0]); b = np.array(g[g.arm == "RAW"].per_repeat.iloc[0])
    print(f"{s:24s} {c:12s} DRA < RAW in {int((a < b).sum())}/10 repeats")
