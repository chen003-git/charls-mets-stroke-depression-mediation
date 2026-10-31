#!/usr/bin/env python3
"""路径模型：MetS -> 早期卒中 -> 抑郁 的三条路径 OR
输出 _output/path_model.json（Figure 1 由此读取，不再依赖任何外部文件）
口径与 run_mets_v2.py 一致：MetS = 腰围(男>=90/女>=85) + TG>=150 + HDL<40 + 自报高血压 + 自报糖尿病，>=3/5
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

def num(s): return pd.to_numeric(s, errors="coerce")
def to_raw(x):
    x = str(x).strip()
    return int(x[:10] + str(int(x[10:]))) if len(x) == 12 else None

def logit(X, y, maxit=100, tol=1e-10):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30); p = 1/(1+np.exp(-eta))
        W = np.maximum(p*(1-p), 1e-10); z = eta + (y-p)/W
        try: bn = np.linalg.solve((X.T*W)@X, (X.T*W)@z)
        except np.linalg.LinAlgError: return None, None
        if np.max(np.abs(bn-b)) < tol: break
        b = bn
    p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); W = p*(1-p)
    se = np.sqrt(np.diag(np.linalg.pinv((X.T*W)@X)))
    return b, se

def fit(df, y, xs):
    s = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(s))] + [s[c].values.astype(float) for c in xs])
    b, se = logit(X, s[y].values)
    if b is None: return None
    import math
    z = b[1]/se[1]
    return dict(n=int(len(s)), OR=round(float(np.exp(b[1])), 4),
                lo=round(float(np.exp(b[1]-1.96*se[1])), 4),
                hi=round(float(np.exp(b[1]+1.96*se[1])), 4),
                P=round(float(2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))), 4))

def main():
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"))
    h = pd.read_stata(HARM, convert_categoricals=False)
    hh = pd.DataFrame({"IDr": h["ID"].map(to_raw), "waist": num(h["r1mwaist"]),
                       "r2": num(h["r2stroke"]), "r3": num(h["r3stroke"])}).dropna(subset=["IDr"])
    d = d.merge(hh, on="IDr", how="left")
    # 主口径：中介缺失记为阴性（与 run_mets_v2.py 一致）
    d["stroke_early"] = ((d["r2"].fillna(0) == 1) | (d["r3"].fillna(0) == 1)).astype(float)
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d.loc[d["waist"].isna(), "c_waist"] = np.nan
    for src, nm, cut, op in [("newtg","c_tg",150,"ge"), ("newhdl","c_hdl",40,"lt")]:
        v = d[src]
        d[nm] = (v >= cut).astype(float) if op == "ge" else (v < cut).astype(float)
        d.loc[v.isna(), nm] = np.nan
    d["c_bp"] = d["hibp"].astype(float); d["c_glu"] = d["diab"].astype(float)
    comps = ["c_waist","c_tg","c_hdl","c_bp","c_glu"]
    d["mets"] = (d[comps].fillna(0).sum(axis=1) >= 3).astype(float)
    cov = [c for c in ["age_w1","female","edu","smoke","drink","cesd_w1"] if c in d.columns]

    # 统一分析样本：与主分析 run_mets_v2.py 完全一致 ——
    # 中介缺失记为阴性（fillna(0)），结局/协变量缺失剔除，n = 11,314。
    # 路径 a / b / c' / 未中介总效应 全部在这一个样本上估计，避免同一投稿里出现多个样本量。
    d = d.dropna(subset=["dep"] + cov).copy()
    print(f"  统一分析样本 n={len(d)}（与主分析一致）")

    out = {}
    for lab, y, xs, key in [
        ("a: MetS -> 早期卒中", "stroke_early", ["mets"] + cov, "path_a_mets_to_stroke"),
        ("b: 早期卒中 -> 抑郁", "dep", ["stroke_early"] + cov, "path_b_stroke_to_dep"),
        ("c': 直接效应", "dep", ["mets","stroke_early"] + cov, "path_c_direct"),
        ("未中介总效应", "dep", ["mets"] + cov, "total_unmediated"),
    ]:
        # c' 与 total 的第一项都是 mets，但 c' 含 stroke_early
        r = fit(d, y, xs)
        if r:
            out[key] = r
            print(f"  {lab:<22} n={r['n']:>6}  OR={r['OR']:.4f} [{r['lo']}, {r['hi']}]  P={r['P']}")
    out["_sample"] = {"n": int(len(d)), "spec": "mediator missing coded as negative; outcome and covariates complete",
                      "consistent_with": "run_mets_v2.py primary analysis"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "path_model.json"), "w"), ensure_ascii=False, indent=2)
    print(f"已写 {os.path.join(OUT, 'path_model.json')}")

if __name__ == "__main__":
    main()
