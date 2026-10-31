#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文19 · MICE 敏感性分析（多重插补 + Rubin 合并）

目的：回应"complete-case 可能选择偏倚"。在**有血脂测量（blood-assay）的 11,334 人**内，
      对缺失的基线 CES-D、协变量、结局（dep）与中介（stroke_early）做多重插补，
      重估总效应、路径 a/b/c' 与 NIE，并与 complete-case 结果对照。

边界（必须写进正文/补充）：
  - 暴露 MetS 仍用实测/自报组分（TG/HDL 仅在 blood-assay 子样本可测），**不对暴露做插补**；
  - 结局与中介被插补 ⇒ 本分析是 MAR 假设下的敏感性分析，不是主分析；
  - 20 套插补、10 个 MICE 循环、每套 100 次 bootstrap（可调）。

用法：
  python mice_sensitivity.py [--m 20] [--cycles 10] [--boot 100] [--seed 42]
输出：_output/mice_sensitivity.json、_output/mice_imputations.csv
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "_output")
sys.path.insert(0, HERE)
from run_mets_v2 import build  # noqa: E402


def add_const(df, cols):
    return sm.add_constant(df[cols].astype(float), has_constant="add")


def fit_pair(df, cov):
    """Fit a: M ~ A + cov; b: Y ~ A + M + cov."""
    xa = ["mets"] + cov
    xb = ["mets", "stroke_early"] + cov
    ma = sm.Logit(df["stroke_early"].astype(float), add_const(df, xa)).fit(disp=0)
    mb = sm.Logit(df["dep"].astype(float), add_const(df, xb)).fit(disp=0)
    return ma, mb


def mediate_point(df, cov):
    ma, mb = fit_pair(df, cov)

    def pred(model, cols, d):
        return model.predict(add_const(d, cols))

    xa = ["mets"] + cov
    xb = ["mets", "stroke_early"] + cov
    d = df.copy()
    p1 = pred(ma, xa, d.assign(mets=1.0))
    p0 = pred(ma, xa, d.assign(mets=0.0))
    y1m1 = pred(mb, xb, d.assign(mets=1.0, stroke_early=1.0))
    y1m0 = pred(mb, xb, d.assign(mets=1.0, stroke_early=0.0))
    y0m1 = pred(mb, xb, d.assign(mets=0.0, stroke_early=1.0))
    y0m0 = pred(mb, xb, d.assign(mets=0.0, stroke_early=0.0))
    ey1m1 = np.mean(p1 * y1m1 + (1 - p1) * y1m0)
    ey1m0 = np.mean(p0 * y1m1 + (1 - p0) * y1m0)
    ey0m0 = np.mean(p0 * y0m1 + (1 - p0) * y0m0)
    return float(ey1m0 - ey0m0), float(ey1m1 - ey1m0), float(ey1m1 - ey0m0)


def fit_total(df, cov):
    return sm.Logit(df["dep"].astype(float), add_const(df, ["mets"] + cov)).fit(disp=0)


def pooled(est, var, label="", df=None):
    est = np.asarray(est, float)
    var = np.asarray(var, float)
    if len(est) == 0:
        return {"label": label, "m": 0, "mean": float("nan"), "se": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "within_var": float("nan"), "between_var": float("nan"), "p": float("nan")}
    m = len(est)
    mean = float(np.mean(est))
    within = float(np.mean(var))
    between = float(np.var(est, ddof=1)) if m > 1 else 0.0
    total = within + (1 + 1.0 / m) * between
    se = float(np.sqrt(total))
    z = stats.norm.ppf(0.975)
    return {"label": label, "m": m, "mean": mean, "se": se,
            "ci_lo": mean - z * se, "ci_hi": mean + z * se,
            "within_var": within, "between_var": between,
            "p": float(2 * (1 - stats.norm.cdf(abs(mean / se)))) if se > 0 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    warnings.filterwarnings("ignore")

    h_path = os.path.join(DATA, r"Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta")
    print("reading stata...", flush=True)
    h = pd.read_stata(h_path, convert_categoricals=False)
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    d = build(d, h)
    print("full eligible cohort n =", len(d), flush=True)

    cov = ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]
    model_vars = ["mets", "stroke_early", "dep"] + cov
    aux = [c for c in ["rural", "bmi", "dyslip", "stroke_w1", "waist"] if c in d.columns]
    cols = model_vars + aux
    df = d[cols].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    miss = {c: int(df[c].isna().sum()) for c in cols}
    print("missingness:", {k: v for k, v in miss.items() if v}, flush=True)
    print("analysis n per complete case:", int(df[model_vars].dropna().shape[0]), flush=True)

    rows = []
    nie_est, nie_var, te_est, te_var = [], [], [], []
    nde_est, nde_var = [], []
    logor_total, se_total = [], []
    logor_a, se_a = [], []
    logor_b, se_b = [], []
    logor_c, se_c = [], []

    for i in range(args.m):
        md = MICEData(df.copy(), k_pmm=20, rng=np.random.default_rng(args.seed + i))
        for _ in range(args.cycles):
            md.update_all()
        imp = md.data.copy()
        # full-data fits
        rt = fit_total(imp, cov)
        ra, rb = fit_pair(imp, cov)
        rows.append({"imputation": i + 1,
                     "OR_total": float(np.exp(rt.params["mets"])),
                     "OR_a": float(np.exp(ra.params["mets"])),
                     "OR_b": float(np.exp(rb.params["stroke_early"])),
                     "OR_cprime": float(np.exp(rb.params["mets"]))})
        logor_total.append(float(rt.params["mets"])); se_total.append(float(rt.bse["mets"]))
        logor_a.append(float(ra.params["mets"])); se_a.append(float(ra.bse["mets"]))
        logor_b.append(float(rb.params["stroke_early"])); se_b.append(float(rb.bse["stroke_early"]))
        logor_c.append(float(rb.params["mets"])); se_c.append(float(rb.bse["mets"]))
        # bootstrap within imputation for mediation
        nd, ni, te = [], [], []
        for b in range(args.boot):
            s = imp.sample(len(imp), replace=True, random_state=int(rng.integers(1e9)))
            try:
                n0, n1, t0 = mediate_point(s, cov)
                nd.append(n0); ni.append(n1); te.append(t0)
            except Exception:
                continue
        if len(ni) >= 10:
            nde_est.append(float(np.mean(nd))); nde_var.append(float(np.var(nd, ddof=1)))
            nie_est.append(float(np.mean(ni))); nie_var.append(float(np.var(ni, ddof=1)))
            te_est.append(float(np.mean(te))); te_var.append(float(np.var(te, ddof=1)))
        print("imputation", i + 1, "done; NIE", float(np.mean(ni)) if ni else None, flush=True)

    res = {
        "n_full_cohort": int(len(d)),
        "n_complete_case_model_vars": int(df[model_vars].dropna().shape[0]),
        "missingness": miss,
        "m": args.m, "cycles": args.cycles, "bootstrap_per_imputation": args.boot,
        "pooled": {
            "total_effect_mets": {"OR": float(np.exp(np.mean(logor_total))),
                                  "logOR": pooled(logor_total, np.square(se_total), "logOR")},
            "path_a_mets_stroke": {"OR": float(np.exp(np.mean(logor_a))),
                                   "logOR": pooled(logor_a, np.square(se_a), "logOR")},
            "path_b_stroke_dep": {"OR": float(np.exp(np.mean(logor_b))),
                                  "logOR": pooled(logor_b, np.square(se_b), "logOR")},
            "path_cprime_mets_dep": {"OR": float(np.exp(np.mean(logor_c))),
                                     "logOR": pooled(logor_c, np.square(se_c), "logOR")},
            "mediation": {
                "NDE": pooled(nde_est, nde_var, "NDE"),
                "NIE": pooled(nie_est, nie_var, "NIE"),
                "TE": pooled(te_est, te_var, "TE"),
            },
        },
        "per_imputation": rows,
    }
    # complete-case reference
    cc_path = os.path.join(OUT, "mets_v2_standard.json")
    if os.path.exists(cc_path):
        cc = json.load(open(cc_path))
        res["complete_case_reference"] = cc.get("main", {})
    json.dump(res, open(os.path.join(OUT, "mice_sensitivity.json"), "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mice_imputations.csv"), index=False)
    print("\n=== MICE pooled ===")
    for k in ["total_effect_mets", "path_a_mets_stroke", "path_b_stroke_dep", "path_cprime_mets_dep"]:
        print(k, res["pooled"][k]["OR"], res["pooled"][k]["logOR"]["ci_lo"], res["pooled"][k]["logOR"]["ci_hi"])
    med = res["pooled"]["mediation"]
    print("NIE", med["NIE"]["mean"], med["NIE"]["ci_lo"], med["NIE"]["ci_hi"])
    print("NDE", med["NDE"]["mean"], med["NDE"]["ci_lo"], med["NDE"]["ci_hi"])
    print("TE", med["TE"]["mean"], med["TE"]["ci_lo"], med["TE"]["ci_hi"])
    print("wrote", os.path.join(OUT, "mice_sensitivity.json"))


if __name__ == "__main__":
    main()
