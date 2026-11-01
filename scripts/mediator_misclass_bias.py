#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probabilistic bias analysis for mediator (self-reported stroke) misclassification.

Companion to scripts/run_review_extras.py, which performs a different and simpler
sensitivity analysis: it flips k mediator events from 1 to 0 at random and reports
how the indirect effect moves (_output/misclass_fragility.json). That analysis is
one-directional (it only removes events) and says nothing about false-positive
mediator reports.

This script instead varies the two operating characteristics of self-reported
stroke, sensitivity (Se) and specificity (Sp), and reclassifies every participant,
so it moves events in both directions.

Method
------
For a given (Se, Sp) pair the true prevalence pi implied by the observed
prevalence pi* follows from  pi* = Se*pi + (1-Sp)(1-pi),  i.e.

    pi = (pi* - (1 - Sp)) / (Se + Sp - 1)

Each participant's mediator status is then redrawn as Bernoulli with

    P(M = 1 | M* = 1) = Se*pi / (Se*pi + (1-Sp)(1-pi))
    P(M = 1 | M* = 0) = (1-Se)*pi / ((1-Se)*pi + Sp(1-pi))

the counterfactual mediation model is re-fitted, and the procedure is repeated
R times per combination. Combinations whose implied true prevalence falls outside
1.5%-3.5% are excluded, because they are incompatible with population estimates
for Chinese adults aged 45 years and over.

Output: _output/mediator_misclass_bias.json
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.environ.setdefault("PYTHONUTF8", "1")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA = os.environ.get("HL_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_output")

R_NUM = 80
SEED = 20261101
SE_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
SP_GRID = [0.98, 0.99, 0.995, 1.00]
PREV_LO, PREV_HI = 0.015, 0.035


def mm(df, cols):
    return sm.add_constant(df[cols].astype(float), has_constant="add")


def mediation_point(d, A, M, Y, covars):
    """Counterfactual mediation, statsmodels chain (same implementation as the main text)."""
    XmA, Xy = [A] + covars, [A, M] + covars
    m_fit = sm.Logit(d[M].astype(float), mm(d, XmA)).fit(disp=0)
    y_fit = sm.Logit(d[Y].astype(float), mm(d, Xy)).fit(disp=0)
    p1 = m_fit.predict(mm(d.assign(**{A: 1.0}), XmA))
    p0 = m_fit.predict(mm(d.assign(**{A: 0.0}), XmA))
    y1m1 = y_fit.predict(mm(d.assign(**{A: 1.0, M: 1.0}), Xy))
    y1m0 = y_fit.predict(mm(d.assign(**{A: 1.0, M: 0.0}), Xy))
    y0m1 = y_fit.predict(mm(d.assign(**{A: 0.0, M: 1.0}), Xy))
    y0m0 = y_fit.predict(mm(d.assign(**{A: 0.0, M: 0.0}), Xy))
    ey1m1 = np.mean(p1 * y1m1 + (1 - p1) * y1m0)
    ey1m0 = np.mean(p0 * y1m1 + (1 - p0) * y1m0)
    ey0m0 = np.mean(p0 * y0m1 + (1 - p0) * y0m0)
    return float(ey1m0 - ey0m0), float(ey1m1 - ey1m0)      # NDE, NIE


def implied_true_prev(obs, se, sp):
    denom = se + sp - 1.0
    return np.nan if abs(denom) < 1e-9 else (obs - (1.0 - sp)) / denom


def reclassify(mstar, se, sp, pi, rng):
    p_pos = se * pi / (se * pi + (1 - sp) * (1 - pi))
    p_neg = (1 - se) * pi / ((1 - se) * pi + sp * (1 - pi))
    prob = np.where(mstar == 1, p_pos, p_neg)
    return (rng.random(len(mstar)) < prob).astype(int)


def build(d, h, h_comm):
    """Identical to the main analysis; reproduced here so the script is self-contained."""
    h2 = h.copy()
    h2["IDr"] = h2["ID"].astype(str).str.strip()
    d["IDr"] = d["ID"].astype(str).str.strip().str.zfill(12)
    d["waist"] = d["IDr"].map(dict(zip(h2["IDr"], pd.to_numeric(h["r1mwaist"], errors="coerce"))))
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d["c_tg"] = (d["newtg"] >= 150).astype(float)
    d["c_hdl"] = (d["newhdl"] < 40).astype(float)
    d["c_bp"] = d["hibp"].astype(float)
    d["c_glu"] = d["diab"].astype(float)
    d["mets_n"] = d[["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]].fillna(0).sum(axis=1)
    d["mets"] = (d["mets_n"] >= 3).astype(int)
    early = ((pd.to_numeric(h["r2stroke"], errors="coerce") == 1) |
             (pd.to_numeric(h["r3stroke"], errors="coerce") == 1)).astype(float)
    h2["stroke_early"] = early
    d = d.merge(h2[["IDr", "stroke_early"]], on="IDr", how="left")
    return d


def main():
    dta = os.path.join(DATA, "Regular Waves", "H_CHARLS_D_Data", "H_CHARLS_D_Data.dta")
    print("[1/3] rebuilding the analysis variables")
    h = pd.read_stata(dta, columns=["ID", "r1mwaist", "r2stroke", "r3stroke"],
                      convert_categoricals=False)
    d0 = build(pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False), h, None)

    covars = [c for c in ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]
              if c in d0.columns and d0[c].notna().sum() > 1000]
    A, M, Y = "mets", "stroke_early", "dep"
    d = d0[[A, M, Y] + covars].dropna().copy()
    obs_prev = float(d[M].mean())
    nde_obs, nie_obs = mediation_point(d, A, M, Y, covars)
    print("      n=%d | observed mediator prevalence=%.4f (%d events) | NIE_obs=%+.6f"
          % (len(d), obs_prev, int(d[M].sum()), nie_obs))

    print("[2/3] bias grid (%d reclassifications per cell)" % R_NUM)
    rng = np.random.default_rng(SEED)
    grid = []
    for se in SE_GRID:
        for sp in SP_GRID:
            pi = implied_true_prev(obs_prev, se, sp)
            if not np.isfinite(pi) or pi <= 0 or pi >= 1:
                continue
            compat = "yes" if PREV_LO <= pi <= PREV_HI else ("marginal" if 0.005 <= pi <= 0.06 else "no")
            nies = []
            for _ in range(R_NUM):
                dd = d.copy()
                dd[M] = reclassify(d[M].values, se, sp, pi, rng)
                try:
                    nies.append(mediation_point(dd, A, M, Y, covars)[1])
                except Exception:
                    pass
            if not nies:
                continue
            a = np.array(nies)
            grid.append(dict(se=se, sp=sp, pi_true=float(pi), compatible=compat,
                             nie_med=float(np.median(a)),
                             nie_lo=float(np.percentile(a, 2.5)),
                             nie_hi=float(np.percentile(a, 97.5)),
                             frac_positive=float((a > 0).mean()), n_ok=len(a)))

    print("[3/3] summarising")
    comp = [g for g in grid if g["compatible"] == "yes"]
    summary = dict(
        n=len(d), obs_prev=obs_prev, nie_obs=nie_obs, nde_obs=nde_obs,
        R=R_NUM, seed=SEED, prev_window=[PREV_LO, PREV_HI],
        n_cells=len(grid), n_compatible=len(comp),
        compatible_retention_min=min((g["nie_med"] / nie_obs for g in comp), default=None),
        compatible_retention_max=max((g["nie_med"] / nie_obs for g in comp), default=None),
        compatible_excluding_zero=sum(1 for g in comp if g["nie_lo"] > 0),
        sign_flips=sum(1 for g in grid if g["nie_med"] <= 0),
        grid=grid)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "mediator_misclass_bias.json")
    json.dump(summary, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("      compatible cells %d/%d | retention %.0f%%-%.0f%% | exclude zero %d/%d | sign flips %d"
          % (len(comp), len(grid), 100 * summary["compatible_retention_min"],
             100 * summary["compatible_retention_max"], summary["compatible_excluding_zero"],
             len(comp), summary["sign_flips"]))
    print("-> %s" % p)


if __name__ == "__main__":
    main()
