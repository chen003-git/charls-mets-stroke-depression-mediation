#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文19 · 分层×暴露交互（总效应）与每层 n/事件/近似人年"""
import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "_output")
sys.path.insert(0, HERE)
from run_mets_v2 import build  # noqa: E402


def fit(df, cols, outcome="dep"):
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    return sm.Logit(df[outcome].astype(float), X).fit(disp=0)


def orci(res, term):
    b = float(res.params[term]); se = float(res.bse[term])
    return float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se)), float(res.pvalues[term])


def main():
    h = pd.read_stata(os.path.join(DATA, r"Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta"), convert_categoricals=False)
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    d = build(d, h)
    cov = ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]
    dd = d.dropna(subset=["mets", "stroke_early", "dep"] + cov).copy()
    med_age = float(d["age_w1"].median())
    dd["age_ge"] = (dd["age_w1"] >= med_age).astype(int)
    print("analytic n", len(dd), "events", int(dd["dep"].sum()), "stroke", int(dd["stroke_early"].sum()))

    strata = {
        "sex": ("female", {0: "male", 1: "female"}),
        "age": ("age_ge", {0: f"<{med_age:.0f}", 1: f">={med_age:.0f}"}),
        "residence": ("rural", {0: "urban", 1: "rural"}),
    }
    rows = []
    for name, (var, levels) in strata.items():
        if var not in dd.columns:
            continue
        adj = [c for c in cov if c != var]
        # interaction model
        di = dd.copy()
        di["mets_x"] = di["mets"] * di[var]
        di = di.dropna(subset=["mets", var, "mets_x", "dep"] + adj).copy()
        ri = fit(di, ["mets", var, "mets_x"] + adj)
        i_or, i_lo, i_hi, i_p = orci(ri, "mets_x")
        for val, label in levels.items():
            sub = dd[dd[var] == val].dropna(subset=["mets", "dep"] + adj).copy()
            try:
                r = fit(sub, ["mets"] + adj)
                o, lo, hi, p = orci(r, "mets")
            except Exception:
                o = lo = hi = p = float("nan")
            rows.append({
                "stratifier": name, "level": label, "n": int(len(sub)),
                "events_dep": int(sub["dep"].sum()), "exposed": int(sub["mets"].sum()),
                "exposed_events": int(((sub["mets"] == 1) & (sub["dep"] == 1)).sum()),
                "stroke_events": int(sub["stroke_early"].sum()),
                "person_years_max7": int(len(sub) * 7),
                "OR_mets": o, "lo": lo, "hi": hi, "P": p,
                "interaction_OR": i_or, "interaction_lo": i_lo,
                "interaction_hi": i_hi, "interaction_P": i_p,
            })
    res = {
        "analytic_n": int(len(dd)), "events_dep": int(dd["dep"].sum()),
        "stroke_events": int(dd["stroke_early"].sum()),
        "age_median": med_age,
        "person_years_note": "maximum 7-year follow-up (2011-2018); approximated as n x 7, no censoring/death adjustment",
        "rows": rows,
    }
    json.dump(res, open(os.path.join(OUT, "stratified_exposure.json"), "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "stratified_exposure.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
