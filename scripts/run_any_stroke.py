#!/usr/bin/env python3
"""论文19 时序对比 · any-stroke 中介（补 Fig3 硬编码 0.0086/1.689 的生产脚本）。

主分析用「早期卒中」（r2/r3，2013/15）作中介，结局是 2018 抑郁 —— 时间有序。
时序对比用「任何卒中」（r2/r3/r4，含 2018，与结局同期）作中介，
展示"同期暴露"如何把 NIE 从 ~0.0018 抬到 ~0.0086（约 5-6 倍）。
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm


# ── GBK 控制台守卫（阶段4 / 审稿人B §1.5b）──────────────────────────────
# 本机 pwsh 实为 PowerShell 5.1，控制台默认 GBK；脚本里 print 非 ASCII（如 kg/m² 的 '²'、
# '≥'、中文）会抛 UnicodeEncodeError，且**崩在写文件之前**，让复现者误判为数据问题。
import os as _os, sys as _sys
_os.environ.setdefault("PYTHONUTF8", "1")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────────────

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_output")

def _mm(df, cols):
    return sm.add_constant(df[cols].astype(float), has_constant="add")

def mediation_full(d, A, M, Y, covars, n_boot=800, seed=42):
    d = d[[A, M, Y] + covars].dropna().copy()
    XmA = [A] + covars; Xy = [A, M] + covars
    def fit(dm):
        return (sm.Logit(dm[M].astype(float), _mm(dm, XmA)).fit(disp=0),
                sm.Logit(dm[Y].astype(float), _mm(dm, Xy)).fit(disp=0))
    def eff(dm, mm, my):
        p1 = mm.predict(_mm(dm.assign(**{A: 1.0}), XmA)); p0 = mm.predict(_mm(dm.assign(**{A: 0.0}), XmA))
        y1_m1 = my.predict(_mm(dm.assign(**{A: 1.0, M: 1.0}), Xy)); y1_m0 = my.predict(_mm(dm.assign(**{A: 1.0, M: 0.0}), Xy))
        y0_m1 = my.predict(_mm(dm.assign(**{A: 0.0, M: 1.0}), Xy)); y0_m0 = my.predict(_mm(dm.assign(**{A: 0.0, M: 0.0}), Xy))
        ey1m1 = np.mean(p1 * y1_m1 + (1 - p1) * y1_m0); ey1m0 = np.mean(p0 * y1_m1 + (1 - p0) * y1_m0)
        ey0m0 = np.mean(p0 * y0_m1 + (1 - p0) * y0_m0)
        return ey1m0 - ey0m0, ey1m1 - ey1m0
    mm, my = fit(d); nde, nie = eff(d, mm, my); te = nde + nie
    rng = np.random.default_rng(seed); bo = []
    for _ in range(n_boot):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        try:
            m2, y2 = fit(s); n2, e2 = eff(s, m2, y2); bo.append((n2, e2, n2 + e2))
        except Exception:
            pass
    b = np.array(bo)
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else np.nan
    return {"NIE": nie, "NIE_lo": pct(b[:, 1], 2.5), "NIE_hi": pct(b[:, 1], 97.5),
            "TE": te, "PM": nie / te if te else np.nan, "n": len(d)}

def main():
    h = pd.read_stata(os.path.join(DATA, r"Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta"), convert_categoricals=False)
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    d["IDr"] = d["ID"].astype(str).str.strip().str.zfill(12)
    h2 = h.copy(); h2["IDr"] = h2["ID"].astype(str).str.strip()

    # build：计算 MetS（与 run_mets_v2.py 一致的 ≥3/5 含腰围定义）
    d["waist"] = d["IDr"].map(dict(zip(h2["IDr"], pd.to_numeric(h["r1mwaist"], errors="coerce"))))
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d["c_tg"] = (d["newtg"] >= 150).astype(float)
    d["c_hdl"] = (d["newhdl"] < 40).astype(float)
    d["c_bp"] = d["hibp"].astype(float)
    d["c_glu"] = d["diab"].astype(float)
    d["mets_n"] = d[["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]].fillna(0).sum(axis=1)
    d["mets"] = (d["mets_n"] >= 3).astype(int)

    # 任何卒中（r2/r3/r4 任一，含 2018 同期）
    st_cols = [c for c in ["r2stroke", "r3stroke", "r4stroke"] if c in h.columns]
    h2["stroke_any"] = (h[st_cols].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1).astype(float)
    d = d.merge(h2[["IDr", "stroke_any"]], on="IDr", how="left")

    cov = [c for c in ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"] if c in d.columns and d[c].notna().sum() > 1000]
    dd = d.dropna(subset=["mets", "stroke_any", "dep"] + cov).copy()
    print(f"any-stroke 分析样本 n={len(dd)} | MetS={int(dd['mets'].sum())} | any-stroke={int(dd['stroke_any'].sum())} | 抑郁={int(dd['dep'].sum())}", flush=True)

    r = mediation_full(dd, "mets", "stroke_any", "dep", cov, 800, seed=42)
    print(f"\nany-stroke: NIE={r['NIE']:.6f} [{r['NIE_lo']:.6f}, {r['NIE_hi']:.6f}] | TE={r['TE']:.6f} | PM={r['PM']:.5f}", flush=True)
    json.dump(r, open(os.path.join(OUT, "mets_any_stroke.json"), "w"), indent=2, default=float)
    print("已写 mets_any_stroke.json", flush=True)

if __name__ == "__main__":
    main()
