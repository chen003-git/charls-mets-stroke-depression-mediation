#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文19 整群 bootstrap 敏感性分析（对应 §2.5 末句与 §3.2 聚类句）

CHARLS 为多阶段分层整群抽样，样本在社区（communityID）层面成簇。
主分析按【个人】重抽样；本脚本按【社区】重抽样，检验推断是否依赖"个人独立"这一假设。

输出：_output/cluster_bootstrap.json
"""
import os, sys, json, time
import numpy as np, pandas as pd
import statsmodels.api as sm
import warnings as _w

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_output")


def _mm(df, cols):
    return sm.add_constant(df[cols].astype(float), has_constant="add")


def mediation_boot(d, A, M, Y, covars, n_boot=800, seed=42, cluster=None, label=""):
    """cluster=None → 个体重抽样；cluster='列名' → 整群重抽样（保留簇内全部个体）。"""
    d = d[[A, M, Y] + covars + ([cluster] if cluster else [])].dropna().copy()
    XmA = [A] + covars; Xy = [A, M] + covars

    def fit(dm):
        return (sm.Logit(dm[M].astype(float), _mm(dm, XmA)).fit(disp=0),
                sm.Logit(dm[Y].astype(float), _mm(dm, Xy)).fit(disp=0))

    def eff(dm, mm, my):
        p1 = mm.predict(_mm(dm.assign(**{A: 1.0}), XmA)); p0 = mm.predict(_mm(dm.assign(**{A: 0.0}), XmA))
        y1_m1 = my.predict(_mm(dm.assign(**{A: 1.0, M: 1.0}), Xy)); y1_m0 = my.predict(_mm(dm.assign(**{A: 1.0, M: 0.0}), Xy))
        y0_m1 = my.predict(_mm(dm.assign(**{A: 0.0, M: 1.0}), Xy)); y0_m0 = my.predict(_mm(dm.assign(**{A: 0.0, M: 0.0}), Xy))
        ey1m1 = np.mean(p1*y1_m1+(1-p1)*y1_m0)
        ey1m0 = np.mean(p0*y1_m1+(1-p0)*y1_m0)
        ey0m0 = np.mean(p0*y0_m1+(1-p0)*y0_m0)
        return ey1m0 - ey0m0, ey1m1 - ey1m0

    mm, my = fit(d); nde, nie = eff(d, mm, my); te = nde + nie
    rng = np.random.default_rng(seed)
    bo = []; n_fail = 0; fail_msg = None; n_conv = 0

    if cluster:
        cids = d[cluster].unique()
        idx_by_c = {c: np.flatnonzero(d[cluster].values == c) for c in cids}
        n_cl = len(cids)
    t0 = time.time()
    for i in range(n_boot):
        if cluster:
            pick = rng.choice(cids, size=n_cl, replace=True)
            s = d.iloc[np.concatenate([idx_by_c[c] for c in pick])]
        else:
            s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        try:
            with _w.catch_warnings(record=True) as _wl:
                _w.simplefilter("always")
                m2, y2 = fit(s); n2, e2 = eff(s, m2, y2); bo.append((n2, e2, n2 + e2))
            n_conv += sum(1 for _x in _wl if "Convergence" in _x.category.__name__)
        except Exception as _e:
            n_fail += 1
            if fail_msg is None: fail_msg = str(_e)[:80]
        if (i + 1) % 200 == 0:
            print("    %s %d/%d  %.0fs" % (label, i + 1, n_boot, time.time() - t0), flush=True)

    b = np.array(bo); pct = lambda a, q: float(np.percentile(a, q)) if len(a) else np.nan
    te_boot = b[:, 2]
    te_p = 2 * min((te_boot <= 0).mean(), (te_boot >= 0).mean()) if len(te_boot) else np.nan
    return {"method": label, "cluster": cluster,
            "n_units": int(n_cl) if cluster else int(len(d)),
            "NDE": nde, "NDE_lo": pct(b[:, 0], 2.5), "NDE_hi": pct(b[:, 0], 97.5),
            "NIE": nie, "NIE_lo": pct(b[:, 1], 2.5), "NIE_hi": pct(b[:, 1], 97.5),
            "TE": te, "TE_lo": pct(te_boot, 2.5), "TE_hi": pct(te_boot, 97.5), "TE_p": te_p,
            "n_boot_requested": int(n_boot), "n_boot_ok": int(len(b)),
            "n_boot_failed": int(n_fail), "n_boot_conv_warn": int(n_conv),
            "n": int(len(d)), "n_exposed": int(d[A].sum()),
            "n_mediator": int(d[M].sum()), "n_outcome": int(d[Y].sum()),
            "PM": nie / te if te else np.nan}


def build(d, h):
    """与 run_mets_v2.build() 完全一致，另加 community。"""
    h2 = h.copy(); h2["IDr"] = h2["ID"].astype(str).str.strip()
    d["IDr"] = d["ID"].astype(str).str.strip().str.zfill(12)
    d["waist"] = d["IDr"].map(dict(zip(h2["IDr"], pd.to_numeric(h["r1mwaist"], errors="coerce"))))
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d["c_tg"]  = (d["newtg"] >= 150).astype(float)
    d["c_hdl"] = (d["newhdl"] < 40).astype(float)
    d["c_bp"]  = d["hibp"].astype(float)
    d["c_glu"] = d["diab"].astype(float)
    d["mets_n"] = d[["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]].fillna(0).sum(axis=1)
    d["mets"]  = (d["mets_n"] >= 3).astype(int)
    early = ((pd.to_numeric(h["r2stroke"], errors="coerce") == 1) |
             (pd.to_numeric(h["r3stroke"], errors="coerce") == 1)).astype(float)
    h2["stroke_early"] = early
    d = d.merge(h2[["IDr", "stroke_early"]], on="IDr", how="left")
    d["community"] = d["IDr"].map(dict(zip(h2["IDr"], pd.to_numeric(h["communityID"], errors="coerce"))))
    return d


def main():
    dta = os.path.join(DATA, "Regular Waves", "H_CHARLS_D_Data", "H_CHARLS_D_Data.dta")
    want = ["ID", "communityID", "r1mwaist", "r2stroke", "r3stroke"]
    print("[1/3] 读 Harmonized 所需列")
    h = pd.read_stata(dta, columns=want, convert_categoricals=False)
    d = build(pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False), h)
    print("      n=%d | 社区数=%d" % (len(d), d["community"].dropna().nunique()))

    cov = [c for c in ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]
           if c in d.columns and d[c].notna().sum() > 1000]
    A, M, Y = "mets", "stroke_early", "dep"

    print("[2/3] 个体 bootstrap（主分析口径，seed=42, n=800）")
    ra = mediation_boot(d, A, M, Y, cov, 800, 42, None, "个体")
    print("[3/3] 整群 bootstrap（communityID，seed=42, n=800）")
    rb = mediation_boot(d, A, M, Y, cov, 800, 42, "community", "整群")

    print("\n%-8s %-9s %-24s %-24s" % ("方法", "NIE", "NIE 95%CI", "TE 95%CI"))
    for r in (ra, rb):
        print("%-8s %-9.5f [%8.5f,%8.5f]   [%8.5f,%8.5f]"
              % (r["method"], r["NIE"], r["NIE_lo"], r["NIE_hi"], r["TE_lo"], r["TE_hi"]))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "cluster_bootstrap.json")
    json.dump({"individual": ra, "cluster": rb}, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n-> %s" % p)


if __name__ == "__main__":
    main()
