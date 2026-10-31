#!/usr/bin/env python3
"""论文19 Table 1 v2：标准 MetS（腰围 + ≥3/5，与 run_mets_v2.py 一致）+ 修正 n=9821。"""
import os
import numpy as np, pandas as pd
from scipy import stats


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
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_output")

h = pd.read_stata(os.path.join(DATA, r"Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta"), convert_categoricals=False)
d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)

# build（与 run_mets_v2.py 一致）
h2 = h.copy(); h2["IDr"] = h2["ID"].astype(str).str.strip()
d["IDr"] = d["ID"].astype(str).str.strip().str.zfill(12)
d["waist"] = d["IDr"].map(dict(zip(h2["IDr"], pd.to_numeric(h["r1mwaist"], errors="coerce"))))
d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
d["c_tg"] = (d["newtg"] >= 150).astype(float)
d["c_hdl"] = (d["newhdl"] < 40).astype(float)
d["c_bp"] = d["hibp"].astype(float)
d["c_glu"] = d["diab"].astype(float)
d["mets_n"] = d[["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]].fillna(0).sum(axis=1)
d["mets"] = (d["mets_n"] >= 3).astype(int)

early = ((pd.to_numeric(h["r2stroke"], errors="coerce") == 1) | (pd.to_numeric(h["r3stroke"], errors="coerce") == 1)).astype(float)
h2["stroke_early"] = early
d = d.merge(h2[["IDr", "stroke_early"]], on="IDr", how="left")

cov = ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]  # 移除 bmi（与腰围共线）
t = d.dropna(subset=["mets", "stroke_early", "dep"] + cov).copy()
print(f"分析样本 n={len(t)} | MetS={int(t['mets'].sum())} ({100*t['mets'].mean():.1f}%)", flush=True)

def fmt(name, unit="mean", cat=False):
    m1 = t[t['mets'] == 1]; m0 = t[t['mets'] == 0]
    if unit == "pct":
        v1 = 100 * m1[name].mean(); v0 = 100 * m0[name].mean(); v = 100 * t[name].mean()
    else:
        v1 = m1[name].mean(); v0 = m0[name].mean(); v = t[name].mean()
    if cat:
        tab = pd.crosstab(t['mets'], t[name])
        _, p, _, _ = stats.chi2_contingency(tab)
    else:
        _, p = stats.ttest_ind(m1[name].dropna(), m0[name].dropna())
    return v, v1, v0, p

_rows = []
rows = [
    ("Age, years (mean)", "age_w1", "mean", False),
    ("Female, %", "female", "pct", True),
    ("BMI, kg/m² (mean)", "bmi", "mean", False),
    ("Education level, 1-3 (mean)", "edu", "mean", False),
    ("Current smoker, %", "smoke", "pct", True),
    ("Current drinker, %", "drink", "pct", True),
    ("Baseline CES-D-10 (mean)", "cesd_w1", "mean", False),
    ("Central obesity (waist ≥90/85), %", "c_waist", "pct", True),
    ("Elevated triglycerides (≥150 mg/dL), %", "c_tg", "pct", True),
    ("Low HDL-cholesterol (<40 mg/dL), %", "c_hdl", "pct", True),
    ("Hypertension, %", "c_bp", "pct", True),
    ("Diabetes, %", "c_glu", "pct", True),
    ("Early incident stroke (2013/15), %", "stroke_early", "pct", True),
    ("Depression (CES-D ≥10, 2018), %", "dep", "pct", True),
]

print("\n| Characteristic | Overall (n=%d) | MetS (n=%d) | Non-MetS (n=%d) | P |" % (len(t), int(t['mets'].sum()), int((t['mets'] == 0).sum())))
print("|---|---|---|---|---|")
for label, name, unit, cat in rows:
    v, v1, v0, p = fmt(name, unit, cat)
    pstr = f"{p:.3f}" if p >= 0.001 else f"{p:.2e}"
    _rows.append({"characteristic": label, "overall": round(float(v), 2),
                  "mets": round(float(v1), 2), "non_mets": round(float(v0), 2), "p": pstr})
    print(f"| {label} | {v:.1f} | {v1:.1f} | {v0:.1f} | {pstr} |")

# ---- R4: Table 1 落盘，便于与稿件逐项对账 ----
if __name__ == "__main__":
    pass

    # ---- R5: Table 1 落盘，便于与稿件逐项对账 ----
    import json as _json
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_output")
    os.makedirs(_out, exist_ok=True)
    # 修复（阶段4 / 审稿人C-M7）：原实现手写 "{},{},{},{},{}" 拼接，
    # 而 characteristic 里含逗号（如 "Age, years (mean)"），导致 13/15 行字段数错误、
    # csv.reader 读不开。改用 csv.writer（自动为含逗号的字段加引号）。
    import csv as _csv
    with open(os.path.join(_out, "table1.csv"), "w", encoding="utf-8", newline="") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["characteristic", "overall", "mets", "non_mets", "p"])
        for _r in _rows:
            _w.writerow([_r["characteristic"], _r["overall"], _r["mets"],
                         _r["non_mets"], _r["p"]])
    _json.dump({"n_total": int(len(t)), "n_mets": int(t["mets"].sum()),
                "n_non_mets": int((1 - t["mets"]).sum()),
                "mets_pct": round(100 * float(t["mets"].mean()), 2),
                "rows": _rows},
               open(os.path.join(_out, "table1.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n已写 _output/table1.csv 与 table1.json（{} 行）".format(len(_rows)))
