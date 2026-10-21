#!/usr/bin/env python3
"""Build the manuscript number registry from reports/ (no hand-typed result values).

Writes reports/numbers/numbers.json and reports/numbers/numbers.tex: one macro per
manuscript-facing value, each with its source file and the computation that produced it.

CONVENTIONS (fixed here, once):
  * R2 of a run = mean over seeds of the per-seed mean of fold R2 (metrics_summary
    `r2_seed_mean`); "+-" = across-seed SD of per-seed means (`r2_seed_std`); the seed count is
    a separate macro. Single-seed rows carry the fold SD and are marked as such.
  * Drug-centred R2, per-line rho, top-10 recall: `analyse_run` pooled per seed, then
    mean +- across-seed SD (per_cellline_ranking.csv / recomputed the same way).
  * RNA gain: R2(RNA rows) - R2(no-RNA rows), pooled per seed (rna_subset.csv method,
    NCC rows only).
  * Paired deltas: seed-paired unless stated; t-CIs are 95%, labelled `tci`; bootstrap CIs
    are copied from the table that computed them and labelled `bci`.
Macro names are letters only; values are printed at the precision the manuscript uses.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from per_cellline_ranking import analyse_run  # noqa: E402

T = ROOT / "reports/revision/tables"
NOV = ROOT / "reports/novelty"
OUT = ROOT / "reports/numbers"
REG: dict[str, dict] = {}


def put(name, value, dec, source, note=""):
    assert name.isalpha(), name
    assert name not in REG, f"duplicate macro {name}"
    v = float(value)
    REG[name] = dict(value=v, decimals=dec, printed=f"{v:.{dec}f}", source=source, note=note, tier="core")


def put_int(name, value, source, note=""):
    assert name.isalpha(), name
    REG[name] = dict(value=int(value), decimals=0, printed=f"{int(value):,}".replace(",", "{,}"),
                     source=source, note=note)



_DIG = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]))


def cam(*parts):
    """Letters-only CamelCase macro fragment from table keys (digits spelled out)."""
    out = []
    for part in parts:
        for w in str(part).replace("|", "_").replace("-", "_").replace(" ", "_").split("_"):
            w = "".join(_DIG.get(ch, ch) if ch.isdigit() else ch for ch in w if ch.isalnum())
            if w:
                out.append(w[0].upper() + w[1:])
    return "".join(out)


def put_soft(name, value, dec, source, note=""):
    """Systematic per-row registration: skip NaN, never collide with a hand-named macro."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or name in REG:
        return
    put(name, value, dec, source, ("[sys] " + note).strip())
    REG[name]["tier"] = "systematic"

def r2(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def tci(d):
    d = np.asarray(d, float)
    return stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))


def dumps(pattern):
    fs = sorted(glob.glob(str(ROOT / pattern)))
    return pd.concat(pd.read_parquet(f) for f in fs) if fs else None


def foldmean(d):
    return float(np.mean([r2(g.y_true, g.y_pred) for _, g in d.groupby("fold")]))


# ------------------------------------------------------------------ sources
MS = pd.read_csv(T / "metrics_summary.csv")
SIG = pd.read_csv(T / "significance_r2.csv")
RANK = pd.read_csv(T / "per_cellline_ranking.csv")
RNA = pd.read_csv(T / "rna_subset.csv")
SRC_MS = "reports/revision/tables/metrics_summary.csv"


def ms(exp, arm, regime="ncc"):
    r = MS[(MS.experiment == exp) & (MS.arm == arm) & (MS.regime == regime)]
    assert len(r) == 1, (exp, arm, regime, len(r))
    return r.iloc[0]


def run(name, exp, arm, regime="ncc", dec=4):
    r = ms(exp, arm, regime)
    put(name + "R", r.r2_seed_mean, dec, SRC_MS, f"{exp}/{regime}/{arm} r2_seed_mean")
    sd = r.r2_seed_std if r.n_seeds > 1 else r.r2_std
    put(name + "Sd", sd, dec, SRC_MS, "across-seed SD" if r.n_seeds > 1 else "fold SD (single seed)")
    put_int(name + "N", r.n_seeds, SRC_MS)
    return r


def sig(exp, a, b, regime="ncc"):
    r = SIG[(SIG.experiment == exp) & (SIG.regime == regime) & (SIG.arm_a == a) & (SIG.arm_b == b)]
    assert len(r) == 1, (exp, a, b)
    return r.iloc[0]


def rank(exp, arm, regime="ncc"):
    return RANK[(RANK.experiment == exp) & (RANK.regime == regime) & (RANK.arm == arm)]


def main():
    # ============================================================ dataset
    prov = json.load(open(T / "dataset_provenance.json"))
    S = "reports/revision/tables/dataset_provenance.json"
    put_int("nRowsVSeventeen", prov["v173_rows"], S)
    put_int("nCellsVSeventeen", prov["v173_celllines"], S)
    put_int("nDrugsVSeventeen", prov["v173_drugs"], S)
    put_int("nRowsFiltered", prov["our_rows"], S)
    put_int("nDrugs", prov["our_drugs"], S)
    put_int("nCells", prov["our_celllines"], S)
    put_int("nDrugsTwoIds", prov["drugs_with_two_drug_ids"], S)
    put_int("nDrugsDropped", prov["v173_drugs"] - prov["our_drugs"] - 1, S,
            "v17.3 compounds minus kept minus the one dropped by feature filtering (216 -> 215)")
    ic = pd.read_csv(ROOT / "data/ic50_data.csv", dtype={"cellline_id": str, "drug_id": str})
    icd = ic.drop_duplicates(["cellline_id", "drug_id"], keep="first")
    put_int("nPairs", len(icd), "data/ic50_data.csv (keep-first de-duplication)")
    rnaf = pd.read_csv(ROOT / "data/ccle_rna_for_ic50.csv", usecols=["cellline_id"])
    n_rna = int(icd.cellline_id.isin(set(rnaf.cellline_id)).groupby(icd.cellline_id).first().sum())
    put_int("nCellsRna", n_rna, "data/ccle_rna_for_ic50.csv ∩ pairs")
    put_int("nCellsNoRna", icd.cellline_id.nunique() - n_rna, "derived")
    put("pctCellsRna", 100 * n_rna / icd.cellline_id.nunique(), 1, "derived from nCellsRna / nCells")
    cens = json.load(open(T / "censoring_decomposition.json"))
    S = "reports/revision/tables/censoring_decomposition.json"
    put("pctExtrapolated", 100 * cens["extrapolated_share"], 0, S, "62% of OUR pairs")
    put("pctBelowMin", 100 * prov["frac_below_min_conc"], 1, "reports/revision/tables/dataset_provenance.json",
        "GDSC-release-wide figure, not our pairs")
    put("pctBetweenStrata", cens["between_stratum_pct"], 1, S)
    put("censorMeanGap", cens["mean_gap"], 1, S)
    # bin-mean R2 (E8): global quintile edges over all pairs, each pair predicted by its bin mean
    y = icd.ic50_value.to_numpy(float)
    edges = np.percentile(y, [20, 40, 60, 80])
    b = np.searchsorted(edges, y, side="left")
    means = np.array([y[b == i].mean() for i in range(5)])
    put("binMeanRtwo", r2(y, means[b]), 4, "data/ic50_data.csv (recomputed here)", "E8, descriptive")
    for i, e in enumerate(edges):
        put("binEdge" + "ABCD"[i], e, 4, "data/ic50_data.csv (recomputed here)")

    # ============================================================ Level 1 (full set, multiseed NCC)
    run("lOneMlp", "multiseed", "standalone_mlp")
    run("lOneDeepcdr", "multiseed", "deepcdr_style")
    run("lOneTcnn", "multiseed", "tcnn_style")
    run("lOneDrugcell", "multiseed", "drugcell_style")
    run("lOneXgb", "multiseed", "xgboost")
    run("lOneRandinit", "multiseed", "qformer_randinit")
    run("lOneClrna", "multiseed", "qformer_clrna")
    run("lOneDra", "multiseed", "multitoken_dra")
    mb = json.load(open(T / "mean_baselines_summary_ncc.json"))
    S = "reports/revision/tables/mean_baselines_summary_ncc.json"
    put("drugMeanR", mb["('r2', 'mean')"]["drug_mean"], 4, S, "single partition (seed 42); SD is fold SD")
    put("drugMeanSd", mb["('r2', 'std')"]["drug_mean"], 4, S, "fold SD")
    five = [ms("multiseed", a).r2_seed_mean for a in
            ["qformer_randinit", "standalone_mlp", "deepcdr_style", "tcnn_style", "drugcell_style"]]
    put("bandLevelOne", max(five) - min(five), 4, SRC_MS,
        "max-min over MoSQ-no-pretraining, MLP, DeepCDR-, tCNN-, DrugCell-style ('five architectures, ours included')")
    put("randinitMinusBest", ms("multiseed", "qformer_randinit").r2_seed_mean - ms("multiseed", "xgboost").r2_seed_mean,
        4, SRC_MS, "vs best comparator (XGBoost)")
    s = sig("multiseed", "qformer_randinit", "standalone_mlp")
    put("randinitVsMlpD", s.delta, 4, "reports/revision/tables/significance_r2.csv")
    put("randinitVsMlpP", s.wilcoxon_p_seedwise, 4, "reports/revision/tables/significance_r2.csv")
    put("randinitVsMlpLo", s.boot_ci_low, 4, "reports/revision/tables/significance_r2.csv", "bci")
    put("randinitVsMlpHi", s.boot_ci_high, 4, "reports/revision/tables/significance_r2.csv", "bci")
    s = sig("multiseed", "multitoken_dra", "qformer_randinit")
    put("draGainD", s.delta, 4, "reports/revision/tables/significance_r2.csv", "10 seeds")
    put("draGainP", s.wilcoxon_p_seedwise, 4, "reports/revision/tables/significance_r2.csv")
    put("draGainLo", s.boot_ci_low, 4, "reports/revision/tables/significance_r2.csv", "bci")
    put("draGainHi", s.boot_ci_high, 4, "reports/revision/tables/significance_r2.csv", "bci")
    multi = MS[(MS.experiment == "multiseed") & (MS.regime == "ncc")]
    put("foldSdMin", multi.r2_std.min(), 4, SRC_MS); put("foldSdMax", multi.r2_std.max(), 4, SRC_MS)
    put("seedSdMin", multi.r2_seed_std.min(), 4, SRC_MS); put("seedSdMax", multi.r2_seed_std.max(), 4, SRC_MS)

    # ============================================================ Level 2 (544-line subset)
    run("lTwoDra", "mosq_common", "multitoken_dra")
    run("lTwoXgb", "mosq_common", "xgboost")
    run("lTwoMlp", "mosq_common", "standalone_mlp")
    run("lTwoTcnns", "comparators", "tcnns")
    run("lTwoDeepcdr", "comparators", "deepcdr")
    run("lTwoDrugcell", "comparators", "drugcell")
    probe = [json.load(open(f))["average"]["r2"] for f in
             sorted(glob.glob(str(ROOT / "reports/letter_audit/probe/mosq_common/ncc__qformer_randinit__seed*.json")))]
    S = "reports/letter_audit/probe/mosq_common/ncc__qformer_randinit__seed*.json"
    put("lTwoRandinitR", np.mean(probe), 4, S); put("lTwoRandinitSd", np.std(probe, ddof=1), 4, S)
    put_int("lTwoRandinitN", len(probe), S)
    dra_c = [json.load(open(f))["average"]["r2"] for f in
             sorted(glob.glob(str(ROOT / "reports/revision/mosq_common/ncc__multitoken_dra__seed*.json")))]
    d = np.array(dra_c) - np.array(probe)
    put("lTwoDraGain", d.mean(), 4, S + " & mosq_common DR-A", "seed-paired")
    put("lTwoDraGainSd", d.std(ddof=1), 4, S + " & mosq_common DR-A")
    rp = np.mean(probe)
    for nm, arm, exp in [("Tcnns", "tcnns", "comparators"), ("Deepcdr", "deepcdr", "comparators"),
                         ("Drugcell", "drugcell", "comparators"), ("Xgb", "xgboost", "mosq_common"),
                         ("Mlp", "standalone_mlp", "mosq_common")]:
        put("lTwoRandinitOver" + nm, rp - ms(exp, arm).r2_seed_mean, 4, SRC_MS + " & probe")
    # S4 on the subset (post-hoc, 2026-10-02)
    s4 = pd.read_csv(NOV / "sweep/tables/s4_common_subset_per_seed.csv")
    S = "reports/novelty/sweep/tables/s4_common_subset_per_seed.csv"
    x = s4[s4.model == "S4 per-drug ridge"]
    put("lTwoSfourR", x.r2_foldmean.mean(), 4, S, "POST-HOC"); put("lTwoSfourSd", x.r2_foldmean.std(), 4, S)
    piv = s4.pivot(index="seed", columns="model", values="r2_foldmean")
    for nm, m in [("Randinit", "MoSQ, no pretraining"), ("Tcnns", "tCNNS"), ("Deepcdr", "DeepCDR"),
                  ("Dra", "MoSQ (DR-A)")]:
        dd = piv["S4 per-drug ridge"] - piv[m]
        lo, hi = tci(dd)
        put("lTwoSfourOver" + nm, dd.mean(), 4, S, "seed-paired"); put("lTwoSfourOver" + nm + "Lo", lo, 4, S, "tci")
        put("lTwoSfourOver" + nm + "Hi", hi, 4, S, "tci")
    put("lTwoSfourOverDrugcell", (piv["S4 per-drug ridge"] - piv["DrugCell"]).dropna().mean(), 4, S, "1 seed")
    # Level 2b / 3 (style vs faithful; representation swap)
    for nm, sty, fa in [("Tcnn", "tcnn_style", "tcnns"), ("Deepcdr", "deepcdr_style", "deepcdr"),
                        ("Drugcell", "drugcell_style", "drugcell")]:
        put("styleOverNative" + nm, ms("style_common", sty).r2_seed_mean - ms("comparators", fa).r2_seed_mean, 4, SRC_MS)
    put("reprSwapMosq", ms("mosq_common", "multitoken_dra").r2_seed_mean
        - ms("matched_mosq_expression697", "multitoken_dra").r2_seed_mean, 3, SRC_MS, "~0.031")
    put("reprMosqOverDeepcdr", ms("matched_mosq_expression697", "multitoken_dra").r2_seed_mean
        - ms("comparators", "deepcdr").r2_seed_mean, 3, SRC_MS, "~0.036")
    put("reprDeepcdrGain", ms("matched_deepcdr_mosqrna", "deepcdr").r2_seed_mean
        - ms("comparators", "deepcdr").r2_seed_mean, 3, SRC_MS, "~0.007")

    # ============================================================ baseline sweep (inductive, 10 seeds)
    sw = pd.read_csv(NOV / "sweep/tables/inductive_per_seed.csv")
    sw = sw[sw.arm.notna()]
    S = "reports/novelty/sweep/tables/inductive_per_seed.csv"
    for nm, arm in [("Sone", "S1"), ("Stwo", "S2"), ("Sthree", "S3"), ("Sfour", "S4"), ("Sfive", "S5")]:
        g = sw[sw.arm == arm]
        for met, mn in [("r2", "R"), ("dc_r2", "Dc"), ("rho", "Rho"), ("top10", "Topten"), ("rna_gain", "RnaGain")]:
            put("sweep" + nm + mn, g[met].mean(), 4, S)
        put("sweep" + nm + "Sd", g["r2"].std(), 4, S)
    p = sw.pivot(index="seed", columns="arm", values="r2")
    dd = p["S4"] - p["qformer_randinit"]
    lo, hi = tci(dd)
    put("sweepSfourOverRandinit", dd.mean(), 4, S, "seed-paired, 10 seeds")
    put("sweepSfourOverRandinitLo", lo, 4, S, "tci"); put("sweepSfourOverRandinitHi", hi, 4, S, "tci")
    put("rangeDrugMeanToBest", REG["sweepSfourR"]["value"] - REG["drugMeanR"]["value"], 4,
        S + " & mean_baselines_summary_ncc.json", "per-drug ridge minus per-drug mean (Table 1 range)")

    # ============================================================ bins (Table S9) and steps 2/3
    run("binGlobal", "binmode_global", "multitoken_dra")
    run("binPerDrugRank", "binmode_per_drug_rank", "multitoken_dra")
    run("binPerDrugZ", "binmode_per_drug_z", "multitoken_dra")
    run("binRandom", "binmode_random", "multitoken_dra")
    put("binWithinDrug", ms("binmode_per_drug_rank", "multitoken_dra").r2_seed_mean
        - ms("binmode_random", "multitoken_dra").r2_seed_mean, 4, SRC_MS)
    put("binDrugIdentity", ms("binmode_global", "multitoken_dra").r2_seed_mean
        - ms("binmode_per_drug_rank", "multitoken_dra").r2_seed_mean, 4, SRC_MS)
    put("binRatio", REG["binWithinDrug"]["value"] / REG["binDrugIdentity"]["value"], 1, SRC_MS)
    for step, nm in [("a1", "stepTwo"), ("aux", "stepThree")]:
        r = pd.read_csv(NOV / f"tables/score_pass_{step}.csv").iloc[0]
        S = f"reports/novelty/tables/score_pass_{step}.csv"
        assert bool(r.final), step
        put(nm + "D", r.delta, 4, S); put(nm + "Lo", r.ci_lo, 4, S, "tci, 15 fold-pairs")
        put(nm + "Hi", r.ci_hi, 4, S, "tci"); put_int(nm + "Pos", r.n_positive, S)
        put(nm + "P", r.wilcoxon_p, 4, S); put(nm + "Dc", r.dc_r2, 4, S)
        put(nm + "RnaGain", r.rna_gain, 4, S)
    put("dcBar", pd.read_csv(NOV / "tables/score_pass_a1.csv").iloc[0].dc_bar, 4,
        "reports/novelty/tables/score_pass_a1.csv (read from per_cellline_ranking.csv)")

    # ============================================================ ranking / drug-centred
    for nm, exp, arm in [("Mlp", "multiseed", "standalone_mlp"), ("Xgb", "multiseed", "xgboost"),
                         ("Randinit", "multiseed", "qformer_randinit"), ("Dra", "multiseed", "multitoken_dra")]:
        g = rank(exp, arm)
        for met, mn in [("r2_drug_centered", "Dc"), ("per_cellline_spearman_mean", "Rho"), ("top10_recall_mean", "Topten")]:
            put("rank" + nm + mn, g[met].mean(), 4, "reports/revision/tables/per_cellline_ranking.csv")
            put("rank" + nm + mn + "Sd", g[met].std(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    g = rank("mean_baselines", "drug_mean")
    put("rankDrugMeanRho", g.per_cellline_spearman_mean.mean(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    put("rankDrugMeanTopten", g.top10_recall_mean.mean(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    mbn = pd.read_csv(T / "mean_baselines_ncc.csv")
    put("drugMeanDcMeasured", mbn[mbn.arm == "drug_mean"].r2_drug_centered.mean(), 4,
        "reports/revision/tables/mean_baselines_ncc.csv")
    trained = RANK[(RANK.experiment == "multiseed") & (RANK.regime == "ncc")].groupby("arm").top10_recall_mean.mean()
    put("toptenTrainedMin", trained.min(), 3, "reports/revision/tables/per_cellline_ranking.csv")
    put("toptenTrainedMax", trained.max(), 3, "reports/revision/tables/per_cellline_ranking.csv")
    ex = pd.read_csv(NOV / "tables/exposure_metrics_per_seed.csv")
    S = "reports/novelty/tables/exposure_metrics_per_seed.csv"
    for nm, arm in [("Trans", "transductive (multiseed s0-9)"), ("TransSthree", "transductive (s0-2)"),
                    ("Size", "size-matched leaky (s0-2)"), ("Ind", "inductive (s0-2)"),
                    ("RandSthree", "random init (s0-2)")]:
        g = ex[ex.arm == arm]
        put("exp" + nm + "Dc", g.dc_r2.mean(), 4, S); put("exp" + nm + "DcSd", g.dc_r2.std(), 4, S)
        put("exp" + nm + "Rho", g.rho.mean(), 4, S)
        put("exp" + nm + "RnaGain", g.rna_gain.mean(), 4, S); put_int("exp" + nm + "N", len(g), S)
    ind = rank("leakfree_strict_ncc", "multitoken_dra", "strict_ncc")
    put("rankIndTopten", ind.top10_recall_mean.mean(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    put("rankIndToptenSd", ind.top10_recall_mean.std(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    put("rankIndRhoSd", ind.per_cellline_spearman_mean.std(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    sil = pd.read_csv(T / "clustering_silhouette.csv").set_index("stage").silhouette_vs_ic50_quintile
    put("silRandinit", sil["Random init"], 3, "reports/revision/tables/clustering_silhouette.csv")
    put("silClrna", sil["CLRNA"], 3, "reports/revision/tables/clustering_silhouette.csv")
    put("silDra", sil["DR-A"], 3, "reports/revision/tables/clustering_silhouette.csv")

    # ============================================================ transfer
    for nm, arm in [("Raw", "raw__xgboost"), ("Randinit", "randinit__xgboost"), ("Clrna", "clrna__xgboost"),
                    ("Recon", "recon_only__xgboost"), ("Dra", "dra__xgboost")]:
        run("tr" + nm, "transfer", arm)
    put("trGainSubmitted", sig("transfer", "dra__xgboost", "raw__xgboost").delta, 4,
        "reports/revision/tables/significance_r2.csv")
    lf, raw = [], []
    for s in range(3):
        a = dumps(f"reports/final_experiments/transfer_leakfree/preds/ncc__dra_leakfree__xgboost__seed{s}__fold*__test.parquet")
        b = dumps(f"reports/final_experiments/transfer_leakfree/preds/ncc__raw__xgboost__seed{s}__fold*__test.parquet")
        for f in range(5):
            lf.append(r2(a[a.fold == f].y_true, a[a.fold == f].y_pred))
            raw.append(r2(b[b.fold == f].y_true, b[b.fold == f].y_pred))
    lf, raw = np.array(lf).reshape(3, 5), np.array(raw).reshape(3, 5)
    S = "reports/final_experiments/transfer_leakfree/preds"
    put("trLeakfreeR", lf.mean(1).mean(), 4, S); put("trLeakfreeSd", lf.mean(1).std(ddof=1), 4, S)
    dd = (lf - raw).ravel()
    put("trLeakfreeD", dd.mean(), 4, S, "fold-paired"); put_int("trLeakfreeNeg", (dd < 0).sum(), S)
    put("trLeakfreeP", stats.wilcoxon(dd).pvalue, 6, S)

    # ============================================================ RNA subsets (NCC only)
    rn = RNA[RNA.regime == "ncc"]
    S = "reports/revision/tables/rna_subset.csv (regime==ncc)"

    def rna_row(nm, exp, arm, regime="ncc"):
        g = RNA[(RNA.regime == regime) & (RNA.experiment == exp) & (RNA.arm == arm)].pivot_table(index="seed", columns="has_rna", values="r2")
        put("rna" + nm + "No", g[False].mean(), 4, S); put("rna" + nm + "Yes", g[True].mean(), 4, S)
        put("rna" + nm + "Gain", (g[True] - g[False]).mean(), 4, S)
    rna_row("DrugMean", "mean_baselines", "drug_mean")
    rna_row("Deepcdr", "multiseed", "deepcdr_style")
    rna_row("Mlp", "multiseed", "standalone_mlp")
    rna_row("Dra", "multiseed", "multitoken_dra")
    rna_row("Ind", "leakfree_strict_ncc", "multitoken_dra", regime="strict_ncc")
    comps = ["deepcdr_style", "drugcell_style", "qformer_clrna", "qformer_randinit", "standalone_mlp", "tcnn_style", "xgboost"]
    gains = [(rn[(rn.experiment == "multiseed") & (rn.arm == a)].pivot_table(index="seed", columns="has_rna", values="r2")
              .pipe(lambda g: (g[True] - g[False]).mean())) for a in comps]
    put("rnaCompGainMin", min(gains), 4, S); put("rnaCompGainMax", max(gains), 4, S)
    put("rnaDraRatio", REG["rnaDraGain"]["value"] / max(gains), 1, S)
    run("rnaFiltDra", "rnafiltered", "multitoken_dra")
    put("rnaFiltDraOverXgb", ms("rnafiltered", "multitoken_dra").r2_seed_mean - ms("rnafiltered", "xgboost").r2_seed_mean, 4, SRC_MS)

    # ============================================================ exposure decomposition (s0-2, fold-mean)
    T_, S_, I_, R_ = [], [], [], []
    for s in range(3):
        T_.append(foldmean(dumps(f"reports/revision/multiseed/preds/ncc__multitoken_dra__seed{s}__fold*__test.parquet")))
        S_.append(foldmean(dumps(f"reports/final_experiments/smlc_strict_ncc/preds/strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet")))
        I_.append(foldmean(dumps(f"reports/revision/leakfree_strict_ncc/preds/strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet")))
        R_.append(foldmean(dumps(f"reports/revision/multiseed/preds/ncc__qformer_randinit__seed{s}__fold*__test.parquet")))
    T_, S_, I_, R_ = map(np.array, (T_, S_, I_, R_))
    S = "per-seed fold-mean R2 from prediction dumps (multiseed, smlc_strict_ncc, leakfree_strict_ncc)"
    put("indR", I_.mean(), 4, S); put("indSd", I_.std(ddof=1), 4, S)
    put("sizeR", S_.mean(), 4, S); put("sizeSd", S_.std(ddof=1), 4, S)
    put("gapTI", (T_ - I_).mean(), 4, S, "seed-paired s0-2"); put("gapTISd", (T_ - I_).std(ddof=1), 4, S)
    put("gapExposure", (S_ - I_).mean(), 4, S); put("gapExposureSd", (S_ - I_).std(ddof=1), 4, S)
    put("gapCorpus", (T_ - S_).mean(), 4, S)
    put("gapMarginal", ms("multiseed", "multitoken_dra").r2_seed_mean - I_.mean(), 4, S + " & " + SRC_MS,
        "10-seed transductive minus 3-seed inductive")
    put("gapExposurePct", 100 * (S_ - I_).mean() / (T_ - I_).mean(), 0, S)
    put("indBelowRandinit", (I_ - R_).mean(), 4, S, "seed-paired"); put("indBelowRandinitSd", (I_ - R_).std(ddof=1), 4, S)
    fi, fr = [], []
    for s in range(3):
        a = dumps(f"reports/revision/leakfree_strict_ncc/preds/strict_ncc__multitoken_dra__seed{s}__fold*__test.parquet")
        b = dumps(f"reports/revision/multiseed/preds/ncc__qformer_randinit__seed{s}__fold*__test.parquet")
        for f in range(5):
            fi.append(r2(a[a.fold == f].y_true, a[a.fold == f].y_pred)); fr.append(r2(b[b.fold == f].y_true, b[b.fold == f].y_pred))
    dd = np.array(fi) - np.array(fr)
    put_int("indBelowRandinitNeg", (dd < 0).sum(), S); put("indBelowRandinitP", stats.wilcoxon(dd).pvalue, 6, S)

    # ============================================================ protocols table
    run("protRandom", "multiseed", "multitoken_dra", regime="random")
    run("protTenfold", "tenfold", "multitoken_dra")
    put("protTenfoldDelta", ms("tenfold", "multitoken_dra").r2_seed_mean - ms("multiseed", "multitoken_dra").r2_seed_mean, 4, SRC_MS)
    put("protRandomXgbGap", ms("multiseed", "multitoken_dra", "random").r2_seed_mean
        - ms("multiseed", "xgboost", "random").r2_seed_mean, 4, SRC_MS)
    ncd = run("ncdDra", "leakfree_ncd", "multitoken_dra", regime="ncd")
    put("ncdDraFoldSd", ncd.r2_std, 3, SRC_MS, "fold SD"); put("ncdDraSeedSd", ncd.r2_seed_std, 3, SRC_MS)
    put("ncdMlpR", ms("ncd_baselines", "standalone_mlp", "ncd").r2_seed_mean, 4, SRC_MS)
    put("ncdTcnnR", ms("ncd_baselines", "tcnn_style", "ncd").r2_seed_mean, 4, SRC_MS)
    mbd = json.load(open(T / "mean_baselines_summary_ncd.json"))
    put("ncdCellMeanR", mbd["('r2', 'mean')"]["cellline_mean"], 4, "reports/revision/tables/mean_baselines_summary_ncd.json")

    # ============================================================ EVAPA, censoring strata
    ev = pd.read_csv(T / "evapa.csv")
    e = lambda exp, arm: ev[(ev.experiment == exp) & (ev.arm == arm)].iloc[0]
    put("evapaDra", e("multiseed", "multitoken_dra").evapa_all, 2, "reports/revision/tables/evapa.csv")
    put("evapaDrugMean", e("mean_baselines", "drug_mean").evapa_all, 2, "reports/revision/tables/evapa.csv")
    put("evapaDraMeasured", e("multiseed", "multitoken_dra").evapa_measured, 2, "reports/revision/tables/evapa.csv")
    cs = pd.read_csv(T / "censoring_strata.csv")
    for st, nm in [("measured", "Measured"), ("extrapolated", "Extrap")]:
        g = cs[cs.stratum == st].set_index("arm").r2
        put("strat" + nm + "Dra", g["multitoken_dra"], 4, "reports/revision/tables/censoring_strata.csv")
        put("strat" + nm + "Margin", g["multitoken_dra"] - g.drop("multitoken_dra").max(), 4,
            "reports/revision/tables/censoring_strata.csv", "vs best other arm in the table")

    # ============================================================ capacity, params, recon
    pc = json.load(open(T / "param_counts.json"))
    put("paramsTotal", pc["deployed_dra_checkpoint"]["total"] / 1e6, 1, "reports/revision/tables/param_counts.json", "M")
    put("paramsSmall", pc["capacity_q8_l2"]["total"] / 1e6, 1, "reports/revision/tables/param_counts.json", "M")
    put("paramsInactive", (pc["deployed_dra_checkpoint"]["total"] - pc["deployed_dra_checkpoint"]["inference"]) / 1e6, 1,
        "reports/revision/tables/param_counts.json", "M")
    put("capacityCost", ms("capacity_q32_l6", "qformer_randinit").r2_seed_mean - ms("capacity_q8_l2", "qformer_randinit").r2_seed_mean, 3, SRC_MS)
    put("capacityParamCutPct", 100 * (1 - pc["capacity_q8_l2"]["total"] / pc["capacity_q32_l6"]["total"]), 0,
        "reports/revision/tables/param_counts.json")
    rk = pd.read_csv(ROOT / "reports/recon_investigation/tables/representation_rank.csv")
    print("representation_rank columns:", rk.columns.tolist())

    # ============================================================ panel completion (exploratory, E2-E6)
    fu = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(NOV / "kriging/rows/fu_rows_s*.csv"))))
    S = "reports/novelty/kriging/rows/fu_rows_s*.csv (EXPLORATORY)"
    ps = fu.groupby(["seed", "k", "base", "arm", "stratum"])[["r2", "rho"]].mean().reset_index()

    def arm(k, base, a, met="r2", st="all"):
        return ps[(ps.k == k) & (ps.base == base) & (ps.arm == a) & (ps.stratum == st)].set_index("seed")[met]
    d = arm("10", "randinit", "krig_own") - arm("10", "dra", "offset")
    put("panelKtenRandKrigOverDraOff", d.mean(), 3, S); put_int("panelKtenRandKrigOverDraOffPos", (d > 0).sum(), S)
    pk = arm("all", "drugmean", "krig_shared")
    put("panelHalfPureKrigR", pk.mean(), 4, S); put("panelHalfPureKrigSd", pk.std(), 4, S)
    put("panelHalfDraOffR", arm("all", "dra", "offset").mean(), 4, S)
    put("panelHalfRandKrigR", arm("all", "randinit", "krig_own").mean(), 4, S)
    put("panelHalfDraKrigR", arm("all", "dra", "krig_own").mean(), 4, S)
    for nm, b in [("Rand", "randinit"), ("Dra", "dra")]:
        put_int("panelHalf" + nm + "KrigOverPurePos", (arm("all", b, "krig_own") - pk > 0).sum(), S)
    for k, nm in [("5", "Kfive"), ("10", "Kten"), ("20", "Ktwenty"), ("all", "Half")]:
        d = arm(k, "dra", "krig_own") - arm(k, "randinit", "krig_own")
        lo, hi = tci(d)
        put("panel" + nm + "DraOverRandKrig", d.mean(), 4, S, "own Sigma")
        put("panel" + nm + "DraOverRandKrigLo", lo, 4, S, "tci"); put("panel" + nm + "DraOverRandKrigHi", hi, 4, S, "tci")
        put_int("panel" + nm + "DraOverRandKrigPos", (d > 0).sum(), S)
        ds = arm(k, "dra", "krig_shared") - arm(k, "randinit", "krig_shared")
        put("panel" + nm + "DraOverRandKrigShared", ds.mean(), 4, S, "shared Sigma")
    for nm, b in [("Rand", "randinit"), ("Dra", "dra")]:
        g = arm("all", b, "krig_own", st="rna") - arm("all", b, "krig_own", st="no_rna")
        put("panelHalf" + nm + "RnaAdv", g.mean(), 4, S)
    for k, nm in [("10", "Kten"), ("20", "Ktwenty")]:
        for b, bn in [("randinit", "Rand"), ("drugmean", "Pure")]:
            src = "krig_own" if b == "randinit" else "krig_shared"
            d = arm(k, b, src, "rho") - arm(k, b, "offset", "rho")
            put("panel" + nm + bn + "RhoGain", d.mean(), 3, S, f"{b}: kriging rho minus offset rho")
        for b, bn in [("drugmean", "Pure"), ("randinit", "Rand")]:
            src = "krig_shared" if b == "drugmean" else "krig_own"
            d = arm(k, b, src, "rho") - arm(k, "dra", "offset", "rho")
            lo, hi = tci(d)
            put("panel" + nm + bn + "KrigRhoOverDra", d.mean(), 4, S)
            put("panel" + nm + bn + "KrigRhoOverDraLo", lo, 4, S, "tci"); put("panel" + nm + bn + "KrigRhoOverDraHi", hi, 4, S, "tci")


    # ============================================================ misc main-text values
    S = "reports/revision/tables/param_counts.csv"
    pcc = pd.read_csv(T / "param_counts.csv").set_index("arm")
    if pd.notna(pcc.loc["capacity_q32_l6", "params_qformer"]) if "capacity_q32_l6" in pcc.index else False:
        put("paramsQformer", pcc.loc["capacity_q32_l6", "params_qformer"] / 1e6, 1, S, "M")
    facts = json.load(open(T / "dataset_facts.json"))
    S = "reports/revision/tables/dataset_facts.json"
    runs = facts["batch_composition"]["runs"]
    put("posPerAnchor", np.mean([r["pos_per_anchor_mean"] for r in runs.values()]), 1, S)
    put("posPerAnchorAnalytic", 255 / 5, 1, "analytic: (256-1)/5", "not a measurement")
    put("matrixCompletenessPct", 100 * facts["matrix_completeness"]["completeness"], 0, S)
    ht = pd.read_csv(ROOT / "reports/recon_investigation/tables/history_traces.csv")
    pub = ht[ht.checkpoint == "published_w0.5"].sort_values("epoch")
    S = "reports/recon_investigation/tables/history_traces.csv"
    put("supconFirst", pub.contrastive_loss.iloc[0], 2, S); put("supconLast", pub.contrastive_loss.iloc[-1], 2, S)
    put("cosFirst", pub.mean_cosine_sim.iloc[0], 2, S); put("cosLast", pub.mean_cosine_sim.iloc[-1], 2, S)
    rk = pd.read_csv(ROOT / "reports/recon_investigation/tables/representation_rank.csv")
    S = "reports/recon_investigation/tables/representation_rank.csv"
    pubr = rk[rk.checkpoint.str.contains("published")].iloc[0]
    put("effRank", pubr.effective_rank, 2, S); put("topOneVarPct", 100 * pubr.top1_var_share, 0, S)
    at = pd.read_csv(T / "attention_by_slot.csv")
    S = "reports/revision/tables/attention_by_slot.csv"
    L = at.layer.max()
    a = lambda st: at[(at.stage == st) & (at.layer == L) & (at.lineage == "ALL") & (at.slot == "drug") & at.has_rna].iloc[0]
    put("attnDrugRandinit", a("random_init").attention_mean, 3, S, "final layer, RNA lines, mean of 5 draws")
    put("attnDrugRandinitSd", a("random_init").sd_across_draws, 3, S, "SD across 5 seeded draws")
    put("attnDrugDra", a("dra").attention_mean, 3, S)
    put("attnDrawSpreadPct", 100 * a("random_init").sd_across_draws / (a("dra").attention_mean - a("random_init").attention_mean), 0, S)
    lin = at[(at.stage == "dra") & (at.layer == L) & (at.lineage != "ALL") & (at.slot == "drug") & at.has_rna]
    put("attnLineageMin", lin.attention_mean.min(), 3, S); put("attnLineageMax", lin.attention_mean.max(), 3, S)
    put_int("attnNLineages", lin.lineage.nunique(), S)
    en = pd.read_csv(T / "attention_entropy.csv")
    S = "reports/revision/tables/attention_entropy.csv"
    put("entropyDra", en[(en.stage == "dra") & (en.layer == en.layer.max())].entropy_mean.iloc[0], 3, S)
    put("entropyCeiling", en.uniform_ceiling.iloc[0], 3, S, "ln 4")
    aq = pd.read_csv(T / "attention_per_query.csv")
    put("querySd", aq[(aq.stage == "dra") & (aq.layer == aq.layer.max())].sd_across_queries.max(), 4,
        "reports/revision/tables/attention_per_query.csv", "max over slots, final layer")
    # objective ablations, seed-paired on shared seeds
    def paired(exp_a, exp_b, seeds, arm="multitoken_dra"):
        mf = pd.read_csv(T / "metrics_full.csv")
        g = lambda e: mf[(mf.experiment == e) & (mf.arm == arm) & (mf.regime == "ncc")].groupby("seed").r2.mean()
        a_, b_ = g(exp_a), g(exp_b)
        common = sorted(set(a_.index) & set(b_.index) & set(seeds))
        return (a_[common] - b_[common]).to_numpy(), common
    d, c = paired("dra_contrastive_only", "multiseed", range(10))
    lo, hi = tci(d)
    S = "reports/revision/tables/metrics_full.csv"
    put("reconD", -d.mean(), 4, S, f"two-term minus one-term, seeds {c}")
    put("reconLo", -hi, 4, S, "tci"); put("reconHi", -lo, 4, S, "tci")
    d, c = paired("dra_noclrna", "multiseed", range(3))
    lo, hi = tci(d)
    put("clrnaInitD", -d.mean(), 4, S, f"CLRNA-init minus random-init DR-A, seeds {c}")
    put("clrnaInitLo", -hi, 4, S, "tci"); put("clrnaInitHi", -lo, 4, S, "tci")
    mf = pd.read_csv(T / "metrics_full.csv")
    x = mf[(mf.experiment == "leakfree_ncd") & (mf.arm == "multitoken_dra")]
    put("ncdDraFoldSdWithin", x.groupby("seed").r2.std().mean(), 3, S, "mean within-seed fold SD")


    # ============================================================ added 2026-10-02 (authors' list)
    ce = json.load(open(T / "experimental_ceiling.json"))
    S = "reports/revision/tables/experimental_ceiling.json (pairing bug fixed 2026-10-02)"
    put_int("dupPairs", ce["n_replicate_pairs"], S)
    put("dupR", ce["replicate_pearson_r"], 4, S, "was 0.2235 from misaligned pairs")
    put("dupSdDiff", ce["sd_of_difference"], 4, S); put("dupMeanDiff", ce["mean_signed_difference"], 3, S)
    put("dupSigma", ce["sigma_upper_bound"], 4, S); put("noiseCeilingRtwo", ce["max_attainable_r2"], 4, S)
    put("noiseHeadroomPct", ce["ncc_full_998cl"]["headroom_used_pct"], 1, S)
    # Level 2 DR-A gain: per-seed range (RESULT.md labelled it a bootstrap CI over 3 seeds)
    put("lTwoDraGainMin", d.min() if False else (np.array(dra_c) - np.array(probe)).min(), 4,
        "reports/letter_audit/probe + reports/revision/mosq_common", "min of 3 per-seed gains (not a CI)")
    put("lTwoDraGainMax", (np.array(dra_c) - np.array(probe)).max(), 4,
        "reports/letter_audit/probe + reports/revision/mosq_common", "max of 3 per-seed gains (not a CI)")
    # RNA subset SD (DR-A, RNA-available, 10 seeds)
    g = RNA[(RNA.regime == "ncc") & (RNA.experiment == "multiseed") & (RNA.arm == "multitoken_dra")]
    put("rnaDraYesSd", g[g.has_rna].r2.std(), 4, "reports/revision/tables/rna_subset.csv (regime==ncc)")
    put("rnaDraNoSd", g[~g.has_rna].r2.std(), 4, "reports/revision/tables/rna_subset.csv (regime==ncc)")
    # DR-A vs MLP / XGBoost (10 seeds)
    for nm, b in [("Mlp", "standalone_mlp"), ("Xgb", "xgboost")]:
        r = sig("multiseed", "multitoken_dra", b)
        put("draOver" + nm, r.delta, 4, "reports/revision/tables/significance_r2.csv")
        put("draOver" + nm + "Lo", r.boot_ci_low, 4, "reports/revision/tables/significance_r2.csv", "bci")
        put("draOver" + nm + "Hi", r.boot_ci_high, 4, "reports/revision/tables/significance_r2.csv", "bci")
        put("draOver" + nm + "P", r.wilcoxon_p_seedwise, 4, "reports/revision/tables/significance_r2.csv")
    run("draNoClrna", "dra_noclrna", "multitoken_dra")
    # kriging ranking (exploratory): leak-free shared-Sigma kriging vs leaky DR-A, k=10
    d = arm("10", "randinit", "krig_shared", "rho") - arm("10", "dra", "offset", "rho")
    lo, hi = tci(d)
    S = "reports/novelty/kriging/rows/fu_rows_s*.csv (EXPLORATORY)"
    put("panelKtenSharedKrigRhoOverDra", d.mean(), 4, S, "random init + shared-Sigma kriging minus DR-A + offset")
    put("panelKtenSharedKrigRhoOverDraLo", lo, 4, S, "tci"); put("panelKtenSharedKrigRhoOverDraHi", hi, 4, S, "tci")
    put_int("panelKtenSharedKrigRhoOverDraPos", (d > 0).sum(), S)
    for k, nm in [("10", "Kten"), ("20", "Ktwenty")]:
        dd = arm(k, "randinit", "krig_shared", "rho") - arm(k, "randinit", "offset", "rho")
        put("panel" + nm + "RandSharedRhoGain", dd.mean(), 3, S, "random init: shared-Sigma kriging rho minus offset rho")
        put_int("panel" + nm + "RandSharedRhoGainPos", (dd > 0).sum(), S)
    put("panelEvalRhoRandinit", arm("10", "randinit", "offset", "rho").mean(), 4, S,
        "evaluation-half per-line rho; offset does not reorder, so k-independent")
    put("panelEvalRhoDrugMean", arm("10", "drugmean", "offset", "rho").mean(), 4, S, "as above, per-drug mean")
    # HISTORICAL, submitted-era values (not results of this revision)
    S = "reports/qformer_query_clrna_vs_dra/clrna_vs_dra_qformer_results.json"
    leg = json.load(open(ROOT / S))
    fm = lambda k: float(np.mean([f["r2"] for f in leg[k]["fold_metrics"]]))
    put("legacyRawXgb", fm("raw_features"), 4, S, "HISTORICAL submitted-era; no generating script in this repo")
    put("legacyDraXgb", fm("dra_features"), 4, S, "HISTORICAL submitted-era")
    put("legacyClrnaXgb", fm("clrna_features"), 4, S, "HISTORICAL submitted-era")
    S2 = "reports/5fold_ncc_ablation/ablation_results.json"
    abl = json.load(open(ROOT / S2))
    put("legacyAblationXgb", float(np.mean([f["r2"] for f in abl["xgboost"]["fold_metrics"]])), 4, S2,
        "HISTORICAL; reproduced by run_cv.py --arm xgboost --seed 42")
    put("legacyGainReproducible", REG["legacyDraXgb"]["value"] - REG["legacyAblationXgb"]["value"], 4, S + " & " + S2,
        "HISTORICAL: submitted DR-A frozen-feature R2 minus reproducible XGBoost baseline")
    put("legacyGainSubmitted", REG["legacyDraXgb"]["value"] - REG["legacyRawXgb"]["value"], 4, S, "HISTORICAL (+4.9%)")


    # ============================================================ SUPPLEMENT: systematic per-row registration
    for _, r in MS.iterrows():
        base = "ms" + cam(r.experiment, r.arm, r.regime)
        put_soft(base + "R", r.r2_seed_mean, 4, SRC_MS)
        put_soft(base + "Sd", r.r2_seed_std, 4, SRC_MS, "across-seed SD")
        put_soft(base + "FoldSd", r.r2_std, 4, SRC_MS, "fold-pooled SD")
        for col, mn in [("rmse_mean", "Rmse"), ("mae_mean", "Mae"), ("male_mean", "Male")]:
            put_soft(base + mn, r[col], 4, SRC_MS)
        put_soft(base + "Evapa", 100 * r.evapa_mean, 2, SRC_MS, "%")
    for _, r in ev.iterrows():
        base = "ev" + cam(r.experiment, r.arm, r.regime)
        for col, mn in [("evapa_all", "All"), ("evapa_measured", "Measured"), ("evapa_extrapolated", "Extrap")]:
            put_soft(base + mn, r[col], 2, "reports/revision/tables/evapa.csv", "%")
    for _, r in cs.iterrows():
        base = "cs" + cam(r.arm, r.stratum)
        put_soft(base + "R", r.r2, 4, "reports/revision/tables/censoring_strata.csv")
        put_soft(base + "Rmse", r.rmse, 4, "reports/revision/tables/censoring_strata.csv")
    for st, g in cs.groupby("stratum"):
        put_soft("csVarY" + cam(st), g.var_y.iloc[0], 2, "reports/revision/tables/censoring_strata.csv")
    put("pctExtrapBetweenDrugs", prov["extrapolation_between_drug_pct"], 1, "reports/revision/tables/dataset_provenance.json")
    encp = pd.read_csv(T / "encoder_partitions.csv")
    for _, r in encp.iterrows():
        base = "enc" + cam(r.regime, r.encoder, r.design)
        put_soft(base + "Mean", r["mean"], 4, "reports/revision/tables/encoder_partitions.csv")
        put_soft(base + "Sd", r["sd"], 4, "reports/revision/tables/encoder_partitions.csv")
    for (e_, rg, a_), g in RANK.groupby(["experiment", "regime", "arm"]):
        base = "rk" + cam(e_, a_, rg)
        for col, mn in [("r2_drug_centered", "Dc"), ("per_cellline_spearman_mean", "Rho"), ("top10_recall_mean", "Topten")]:
            put_soft(base + mn, g[col].mean(), 4, "reports/revision/tables/per_cellline_ranking.csv")
            put_soft(base + mn + "Sd", g[col].std(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    for (e_, rg, a_), g in RNA.groupby(["experiment", "regime", "arm"]):
        pv = g.pivot_table(index="seed", columns="has_rna", values="r2")
        if True not in pv or False not in pv:
            continue
        base = "rs" + cam(e_, a_, rg)
        put_soft(base + "No", pv[False].mean(), 4, "reports/revision/tables/rna_subset.csv")
        put_soft(base + "Yes", pv[True].mean(), 4, "reports/revision/tables/rna_subset.csv")
        put_soft(base + "Gain", (pv[True] - pv[False]).mean(), 4, "reports/revision/tables/rna_subset.csv")
    # transfer EVAPA by arm (all / measured) from the per-seed transfer tables
    trt = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(T / "transfer_ncc_seed*.csv"))))
    for (feat, ds), g in trt.groupby(["features", "downstream"]):
        put_soft("tr" + cam(feat, ds) + "Evapa", 100 * g.evapa.mean(), 2, "reports/revision/tables/transfer_ncc_seed*.csv", "%")
    # S9 bin agreement
    ba = facts["bin_agreement"]
    put("binAgreePct", 100 * ba["global_vs_per_drug_rank"], 1, "reports/revision/tables/dataset_facts.json")
    put("binMaxSharePct", 100 * ba["per_drug_rank_max_bin_share"], 1, "reports/revision/tables/dataset_facts.json")
    # S10 capacity parameters
    S = "reports/revision/tables/param_counts.csv"
    for arm_, nm in [("capacity_q8_l2", "Small"), ("capacity_q16_l4", "Mid"), ("capacity_q32_l6", "Shipped")]:
        put("params" + nm + "TotalM", pcc.loc[arm_, "params_total"] / 1e6, 1, S)
        put("params" + nm + "QformerM", pcc.loc[arm_, "params_qformer"] / 1e6, 1, S)
    put("paramsQformerM", pcc.loc["capacity_q32_l6", "params_qformer"] / 1e6, 1, S, "shipped configuration, constructed model")
    put("paramsMlpM", pcc.loc["standalone_mlp", "params_total"] / 1e6, 2, S)
    # S11 objective ablation: arms vs contrastive-only on the seeds all arms share (t-CI)
    mf = pd.read_csv(T / "metrics_full.csv")
    sm = lambda e: mf[(mf.experiment == e) & (mf.arm == "multitoken_dra") & (mf.regime == "ncc")].groupby("seed").r2.mean()
    arms_ = {"Contrastive": "dra_contrastive_only", "Published": "binmode_global", "Wthree": "dra_reconw3p0",
             "Weleven": "dra_reconw11p0", "ReconOnly": "dra_recon_only"}
    ser = {k: sm(v) for k, v in arms_.items()}
    shared = sorted(set.intersection(*[set(v.index) for v in ser.values()]))
    S = f"reports/revision/tables/metrics_full.csv (shared seeds {shared})"
    for k, v in ser.items():
        put("obj" + k + "R", v[shared].mean(), 4, S); put("obj" + k + "Sd", v[shared].std(), 4, S)
        if k != "Contrastive":
            dd = v[shared] - ser["Contrastive"][shared]
            lo, hi = tci(dd)
            put("obj" + k + "D", dd.mean(), 4, S); put("obj" + k + "Lo", lo, 4, S, "tci"); put("obj" + k + "Hi", hi, 4, S, "tci")
    rr = pd.read_csv(ROOT / "reports/recon_investigation/tables/recon_r2.csv")
    S = "reports/recon_investigation/tables/recon_r2.csv"
    pd_ = rr[(rr.checkpoint == "published_w0.5") & (rr.branch == "mask_drug")].iloc[0]
    put("reconVarRatioPct", 100 * pd_.var_ratio, 2, S); put("reconAffineOracle", pd_.r2_affine_oracle, 6, S)
    put("reconEtaCross", pd_.eta2_crossfit, 4, S); put("reconEtaNull", pd_.eta2_null_analytic, 4, S)
    wd = pd.read_csv(ROOT / "reports/recon_investigation/tables/weight_dose_response.csv").set_index("checkpoint")
    S = "reports/recon_investigation/tables/weight_dose_response.csv"
    put("cosWithRecon", wd.loc["published_w0.5", "final_cos_sim"], 4, S); put("cosWithoutRecon", wd.loc["contrastive_only_w0", "final_cos_sim"], 4, S)
    nr = pd.read_csv(ROOT / "reports/recon_investigation/tables/no_rna_paired_tests.csv").iloc[0]
    S = "reports/recon_investigation/tables/no_rna_paired_tests.csv"
    put("didD", nr.mean_diff, 4, S); put("didLo", nr.ci_low, 4, S); put("didHi", nr.ci_high, 4, S); put("didP", nr.wilcoxon_p, 2, S)
    tw = pd.read_csv(ROOT / "reports/recon_investigation/tables/no_rna_paired_tests.csv").iloc[1]
    put("reconTenP", tw.wilcoxon_p, 2, S, "two-term vs one-term, 10 seeds")


    # S13 caption: ChemBERTa-2 span at fixed weights (seed 0) across resampled partitions
    er = pd.read_csv(T / "encoder_partitions_runs.csv")
    x = er[(er.encoder == "chemberta") & (er.regime == "ncd") & (er.design == "partition") & (er.seed == 0)]
    put("encChembertaFixedMin", x.r2.min(), 4, "reports/revision/tables/encoder_partitions_runs.csv")
    put("encChembertaFixedMax", x.r2.max(), 4, "reports/revision/tables/encoder_partitions_runs.csv")
    # S15: CLRNA-initialisation paired difference SD (3 seeds)
    d_c, c_c = paired("dra_noclrna", "multiseed", range(3))
    put("clrnaInitSd", np.std(-d_c, ddof=1), 4, "reports/revision/tables/metrics_full.csv", f"seeds {c_c}")


    # POST-HOC ridge controls: raw expression vs BulkRNABERT (2026-10-02)
    for tag, f in [("Ctl", "ridge_raw_vs_embedding"), ("Full", "ridge_full_transcriptome")]:
        t = pd.read_csv(NOV / f"sweep/tables/{f}_per_seed.csv")
        S = f"reports/novelty/sweep/tables/{f}_per_seed.csv (POST-HOC)"
        put_int("ridge" + tag + "Lines", t.n_lines.iloc[0], S)
        for a, nm in [("E_embedding", "Emb"), ("X_raw697", "Raw"), ("C_context", "Ctx"), ("F_full_transcriptome", "Full")]:
            g = t[t.arm == a]
            if len(g) == 0:
                continue
            put("ridge" + tag + nm + "R", g.r2.mean(), 4, S); put("ridge" + tag + nm + "Sd", g.r2.std(), 4, S)
            put("ridge" + tag + nm + "Dc", g.dc_r2.mean(), 4, S)
        dl = pd.read_csv(NOV / f"sweep/tables/{f}_deltas.csv")
        for _, r in dl[dl.metric == "r2"].iterrows():
            a, b = [x.split("_")[0] for x in r.comparison.split(" - ")]
            nm = "ridge" + tag + {"E": "Emb", "X": "Raw", "C": "Ctx", "F": "Full"}[a] + "Minus" + {"E": "Emb", "X": "Raw", "C": "Ctx", "F": "Full"}[b]
            put(nm, r.delta, 4, S); put(nm + "Lo", r.ci_lo, 4, S, "tci"); put(nm + "Hi", r.ci_hi, 4, S, "tci")


    # POST-HOC drug-side control (2026-10-04): per-cell-line ridge on drug representations
    t = pd.read_csv(NOV / "sweep/tables/ridge_drug_side_per_fold.csv")
    S = "reports/novelty/sweep/tables/ridge_drug_side_per_fold.csv (POST-HOC)"
    pm_ = t.groupby(["regime", "partition", "arm"]).r2.mean().reset_index()
    for rg, rn_ in [("ncd", "Ncd"), ("scaffold", "Scaf")]:
        for a in ["CB", "DS", "MG", "MTR", "LM"]:
            g = pm_[(pm_.regime == rg) & (pm_.arm == a)].r2
            nm = "drugRidge" + rn_ + cam(a.lower())
            put(nm + "R", g.mean(), 4, S); put(nm + "Sd", g.std(), 4, S, "SD across 4 partitions")
            put(nm + "Primary", pm_[(pm_.regime == rg) & (pm_.arm == a) & (pm_.partition == "primary")].r2.iloc[0], 4, S)
    dl = pd.read_csv(NOV / "sweep/tables/ridge_drug_side_deltas.csv")
    for _, r in dl[dl.metric == "r2"].iterrows():
        a, b = r.comparison.split(" - ")
        nm = "drugRidge" + ("Ncd" if r.regime == "ncd" else "Scaf") + cam(a.lower()) + "Minus" + cam(b.lower())
        put(nm, r.delta, 4, S, "fold-paired, 20 folds"); put(nm + "Lo", r.ci_lo, 4, S, "tci"); put(nm + "Hi", r.ci_hi, 4, S, "tci")


    # PRE-REGISTERED downstream case (2026-10-04): drug mechanism from per-drug representations
    S = "reports/novelty/downstream/moa_summary.csv"
    ms_ = pd.read_csv(NOV / "downstream/moa_summary.csv").set_index("arm")
    for a in ms_.index:
        put("moa" + cam(a.lower()) + "Ba", ms_.loc[a, "balanced_accuracy"], 4, S)
        put("moa" + cam(a.lower()) + "Nn", ms_.loc[a, "nn1_retrieval"], 4, S)
    md_ = pd.read_csv(NOV / "downstream/moa_deltas.csv")
    S = "reports/novelty/downstream/moa_deltas.csv"
    for _, r in md_.iterrows():
        a, b = r.comparison.split(" - ")
        nm = "moa" + cam(a.lower()) + "Minus" + cam(b.lower())
        put(nm, r.d_balanced_accuracy, 4, S); put(nm + "Lo", r.ba_lo, 4, S, "bootstrap over drugs")
        put(nm + "Hi", r.ba_hi, 4, S, "bootstrap over drugs")

    # ============================================================ added 2026-10-12 (finalisation list, items 1-5)
    def put_or_same(name, value, dec, source, note=""):
        """Register, or if a hand-named macro already holds this value, assert they agree."""
        if name in REG:
            assert abs(REG[name]["value"] - float(value)) < 1e-12, (name, REG[name]["value"], value)
            return
        put(name, value, dec, source, note)

    # (1) RNA gain of the unpretrained Q-Former, 10 seeds; and every arm on seeds 0-2, for the
    #     paragraph that quotes s0-2 values next to the 10-seed comparator ceiling
    rna_row("Randinit", "multiseed", "qformer_randinit")
    rna_row("Clrna", "multiseed", "qformer_clrna")
    S = "reports/revision/tables/rna_subset.csv (regime==ncc, seeds 0-2)"
    g3 = {a: rn[(rn.experiment == "multiseed") & (rn.arm == a) & rn.seed.isin([0, 1, 2])]
          .pivot_table(index="seed", columns="has_rna", values="r2").pipe(lambda g: (g[True] - g[False]).mean())
          for a in comps}
    put("rnaCompGainMaxSthree", max(g3.values()), 4, S, f"max over comparators on s0-2 ({max(g3, key=g3.get)})")
    put("rnaRandinitGainSthree", g3["qformer_randinit"], 4, S)
    # (2) drug-centred / ranking for Q-Former + CLRNA (Table 2, Table S20)
    g = rank("multiseed", "qformer_clrna")
    for met, mn in [("r2_drug_centered", "Dc"), ("per_cellline_spearman_mean", "Rho"), ("top10_recall_mean", "Topten")]:
        put("rankClrna" + mn, g[met].mean(), 4, "reports/revision/tables/per_cellline_ranking.csv")
        put("rankClrna" + mn + "Sd", g[met].std(), 4, "reports/revision/tables/per_cellline_ranking.csv")
    # (3) random-init Q-Former under leak-free NCD (Discussion, Table S16)
    run("ncdRandinit", "ncd_baselines", "qformer_randinit", regime="ncd")
    run("ncdXgb", "ncd_baselines", "xgboost", regime="ncd")
    put("ncdDraOverRandinit", ms("leakfree_ncd", "multitoken_dra", "ncd").r2_seed_mean
        - ms("ncd_baselines", "qformer_randinit", "ncd").r2_seed_mean, 4, SRC_MS)
    # (4) Table S8(b): RMSE, MAE, MALE (+ across-seed SD of MALE from metrics_full)
    S = SRC_MS + " (MALE SD: across-seed, metrics_full.csv)"
    for nm, e_, a_, rg in [("Dra", "multiseed", "multitoken_dra", "ncc"), ("Randinit", "multiseed", "qformer_randinit", "ncc"),
                           ("Xgb", "multiseed", "xgboost", "ncc"), ("Mlp", "multiseed", "standalone_mlp", "ncc"),
                           ("Ind", "leakfree_strict_ncc", "multitoken_dra", "strict_ncc"),
                           ("DrugMean", "mean_baselines", "drug_mean", "ncc")]:
        r = ms(e_, a_, rg)
        put("err" + nm + "Rmse", r.rmse_mean, 4, S); put("err" + nm + "Mae", r.mae_mean, 4, S)
        put("err" + nm + "Male", r.male_mean, 4, S)
        ps_ = mf[(mf.experiment == e_) & (mf.arm == a_) & (mf.regime == rg)].groupby("seed").male.mean()
        if len(ps_) > 1:
            put("err" + nm + "MaleSd", ps_.std(), 4, S, f"across-seed SD, {len(ps_)} seeds")
    # (5) Table S22 per-k values: offset arms, kriging alone, placebo, matrix factorisation
    S = "reports/novelty/kriging/rows/fu_rows_s*.csv (EXPLORATORY)"
    for k, nm in [("5", "Kfive"), ("10", "Kten"), ("20", "Ktwenty"), ("all", "Half")]:
        put_or_same("panel" + nm + "DrugMeanOffR", arm(k, "drugmean", "offset").mean(), 4, S)
        put_or_same("panel" + nm + "RandOffR", arm(k, "randinit", "offset").mean(), 4, S)
        put_or_same("panel" + nm + "DraOffR", arm(k, "dra", "offset").mean(), 4, S)
        put_or_same("panel" + nm + "PureKrigR", arm(k, "drugmean", "krig_shared").mean(), 4, S, "kriging alone")
        put_or_same("panel" + nm + "RandKrigR", arm(k, "randinit", "krig_own").mean(), 4, S, "own Sigma")
        put_or_same("panel" + nm + "DraKrigR", arm(k, "dra", "krig_own").mean(), 4, S, "own Sigma")
    dr = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(NOV / "kriging/rows/dra_rows_s*.csv"))))
    dr = dr[(dr.stratum == "all") & (dr.arm == "dra_transductive+lowrank(placebo)")]
    plc = dr.groupby(["seed", "k"]).r2.mean().groupby("k").mean()
    S = "reports/novelty/kriging/rows/dra_rows_s*.csv (EXPLORATORY; placebo = permuted drug labels in shared Sigma)"
    for k, nm in [("5", "Kfive"), ("10", "Kten"), ("20", "Ktwenty"), ("all", "Half")]:
        put("panel" + nm + "PlaceboDraR", plc[k], 4, S)
    put("panelPlaceboDraMin", plc.min(), 3, S, "min over k"); put("panelPlaceboDraMax", plc.max(), 3, S, "max over k")
    pps = pd.read_csv(NOV / "sweep/tables/panel_per_seed.csv", dtype={"k": str})
    pps = pps[pps.stratum == "all"].pivot_table(index=["k", "seed"], columns="arm", values="r2")
    S = "reports/novelty/sweep/tables/panel_per_seed.csv (P1 biased ALS)"
    for k, nm in [("5", "Kfive"), ("10", "Kten"), ("20", "Ktwenty"), ("all", "Half")]:
        x = pps.loc[k]
        put("panel" + nm + "MfR", x["P1_mf_als"].mean(), 4, S)
        dd = x["P1_mf_als"] - x["drugmean+krig_shared"]
        assert np.allclose(x["drugmean+krig_shared"].mean(), REG["panel" + nm + "PureKrigR"]["value"]), k
        lo, hi = tci(dd)
        put("panel" + nm + "MfMinusPureKrig", dd.mean(), 4, S, "seed-paired"); put_int("panel" + nm + "MfMinusPureKrigPos", (dd > 0).sum(), S)
        put("panel" + nm + "MfMinusPureKrigLo", lo, 4, S, "tci"); put("panel" + nm + "MfMinusPureKrigHi", hi, 4, S, "tci")

    # ============================================================ added 2026-10-16 (Table 2: absolute values for every row)
    # steps 2/3 as absolute R2 (per-seed mean of fold R2, across-seed SD), and the seed-0-2 random-init
    # reference those runs are paired against (0.7602, not the 10-seed 0.7593)
    for step, nm in [("a1", "stepTwo"), ("aux", "stepThree")]:
        f = pd.read_csv(NOV / f"tables/score_pass_{step}_fold_pairs.csv")
        S = f"reports/novelty/tables/score_pass_{step}_fold_pairs.csv"
        ps_ = f.groupby("seed")[["model", "randinit"]].mean()
        assert len(ps_) == 3, step
        put(nm + "R", ps_.model.mean(), 4, S, "seeds 0-2"); put(nm + "Sd", ps_.model.std(), 4, S, "across-seed SD")
        put_or_same("randinitSthreeR", ps_.randinit.mean(), 4, S, "random init, seeds 0-2 (paired reference)")
        put_or_same("randinitSthreeSd", ps_.randinit.std(), 4, S, "across-seed SD")
    # drug-centred R2 for the bin-mode rows (Table S9 / Table 2), seeds 0-2
    for nm, e_ in [("binRandom", "binmode_random"), ("binPerDrugRank", "binmode_per_drug_rank"),
                   ("binPerDrugZ", "binmode_per_drug_z"), ("binGlobal", "binmode_global")]:
        g = rank(e_, "multitoken_dra")
        assert g.seed.nunique() == 3, e_
        put(nm + "Dc", g.r2_drug_centered.mean(), 4, "reports/revision/tables/per_cellline_ranking.csv", "seeds 0-2")
        put(nm + "DcSd", g.r2_drug_centered.std(), 4, "reports/revision/tables/per_cellline_ranking.csv")

    # ============================================================ write
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(REG, indent=1, ensure_ascii=False))
    import datetime, hashlib, subprocess
    digest = hashlib.sha256((OUT / "numbers.json").read_bytes()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    stamp = f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {len(REG)} values | numbers.json sha256 {digest} | repo {commit}"
    lines = ["% AUTO-GENERATED by scripts/numbers/build_numbers.py -- do not edit by hand.",
             "% Every value is computed from reports/; see numbers.json for source and note.",
             f"% VERSION: {stamp}",
             f"\\newcommand{{\\numbersVersion}}{{{digest}}}", ""]
    for k, v in REG.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v['printed']}}}  % {v['source']}" + (f" | {v['note']}" if v["note"] else ""))
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(REG)} values -> {OUT}/numbers.{{json,tex}}")


if __name__ == "__main__":
    main()
