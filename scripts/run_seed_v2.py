#!/usr/bin/env python3
"""种子稳健性：现行规格下 9 个 bootstrap 种子，输出 _output/seed_sensitivity.json
如实报告全部种子的 NIE 与 95% 百分位 CI 下界（含为负者）
"""
import os, json
import numpy as np
import pandas as pd


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

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "_output")
HARM = os.path.join(DATA, "Regular Waves", "H_CHARLS_D_Data", "H_CHARLS_D_Data.dta")
SEEDS = [42, 0, 1, 2, 3, 7, 123, 999, 2024]
NBOOT = 500

def num(s): return pd.to_numeric(s, errors="coerce")
def to_raw(x):
    x = str(x).strip()
    return int(x[:10] + str(int(x[10:]))) if len(x) == 12 else None

def logit(X, y, maxit=80, tol=1e-9):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X@b, -30, 30); p = 1/(1+np.exp(-eta))
        W = np.maximum(p*(1-p), 1e-10); z = eta + (y-p)/W
        try: bn = np.linalg.solve((X.T*W)@X, (X.T*W)@z)
        except np.linalg.LinAlgError: return None
        if np.max(np.abs(bn-b)) < tol: return bn
        b = bn
    return b

def build():
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"))
    h = pd.read_stata(HARM, convert_categoricals=False)
    hh = pd.DataFrame({"IDr": h["ID"].map(to_raw), "waist": num(h["r1mwaist"]),
                       "r2": num(h["r2stroke"]), "r3": num(h["r3stroke"])}).dropna(subset=["IDr"])
    d = d.merge(hh, on="IDr", how="left")
    d["stroke_early"] = np.where(d["r2"].notna() | d["r3"].notna(),
                                 ((d["r2"] == 1) | (d["r3"] == 1)).astype(float), np.nan)
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d.loc[d["waist"].isna(), "c_waist"] = np.nan
    d["c_tg"] = (d["newtg"] >= 150).astype(float); d.loc[d["newtg"].isna(), "c_tg"] = np.nan
    d["c_hdl"] = (d["newhdl"] < 40).astype(float); d.loc[d["newhdl"].isna(), "c_hdl"] = np.nan
    d["c_bp"] = d["hibp"].astype(float); d["c_glu"] = d["diab"].astype(float)
    comps = ["c_waist","c_tg","c_hdl","c_bp","c_glu"]
    d["mets"] = (d[comps].fillna(0).sum(axis=1) >= 3).astype(float)
    cov = [c for c in ["age_w1","female","edu","smoke","drink","cesd_w1"] if c in d.columns]
    return d, cov

def mediation(df, cov):
    s = df.dropna(subset=["mets","stroke_early","dep"] + cov)
    n = len(s)
    if n < 200: return None
    Xy = np.column_stack([np.ones(n), s["mets"]] + [s[c] for c in cov] + [s["stroke_early"]])
    Xm = np.column_stack([np.ones(n), s["mets"]] + [s[c] for c in cov])
    by = logit(Xy, s["dep"].values); bm = logit(Xm, s["stroke_early"].values)
    if by is None or bm is None: return None
    cc = np.column_stack([s[c] for c in cov])
    def pr(b, av, mv): return 1/(1+np.exp(-np.clip(np.column_stack([np.ones(n), np.full(n,av), cc, np.full(n,mv)])@b, -30, 30)))
    def pm(b, av): return 1/(1+np.exp(-np.clip(np.column_stack([np.ones(n), np.full(n,av), cc])@b, -30, 30)))
    m1, m0 = pm(bm,1.0), pm(bm,0.0)
    y11, y10 = pr(by,1.0,m1), pr(by,1.0,m0)
    return float(np.mean(y11-y10))

def main():
    d, cov = build()
    rows = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd); nies = []
        for _ in range(NBOOT):
            r = mediation(d.iloc[rng.integers(0, len(d), len(d))], cov)
            if r is not None: nies.append(r)
        if not nies: continue
        lo, hi = np.percentile(nies, 2.5), np.percentile(nies, 97.5)
        rows.append(dict(seed=int(sd), n_boot=len(nies), NIE=float(np.mean(nies)),
                         NIE_lo=float(lo), NIE_hi=float(hi), crosses_zero=bool(lo <= 0)))
        print(f"  seed {sd:>5}: NIE={np.mean(nies):+.6f}  [{lo:+.6f}, {hi:+.6f}]  {'CI 跨零' if lo <= 0 else 'CI>0'}")
    n_neg = sum(1 for r in rows if r["crosses_zero"])
    print(f"\n  合计 {len(rows)} 个种子，CI 跨零 {n_neg} 个")
    if rows:
        print(f"  最小下界 {min(r['NIE_lo'] for r in rows):+.6f}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(seeds=rows, n_seeds=len(rows), n_crossing_zero=n_neg,
                   min_lo=(min(r["NIE_lo"] for r in rows) if rows else None),
                   spec="primary (>=3/5, waist+TG+HDL+self-reported BP/GLU, BMI excluded)"),
              open(os.path.join(OUT, "seed_sensitivity.json"), "w"), ensure_ascii=False, indent=2)
    print(f"已写 {os.path.join(OUT, 'seed_sensitivity.json')}")

if __name__ == "__main__":
    main()
