#!/usr/bin/env python3
"""信号链简单验证：三高(上游)→炎症CRP(中游)→卒中/抑郁(下游)
数据：analysis_harmonized.csv(协变量+结局) + Blood_20140429.dta(血检血糖/CRP/糖化)
"""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

DATA = r"D:\SCI论文\19_CHARLS疾病链\data"

# ── 加载 ──
d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"))
b = pd.read_stata(os.path.join(DATA, "2011全国", "Blood_20140429", "Blood_20140429.dta"), convert_categoricals=False)
# ID 对齐：b 的 11 位带前导0 → 去前导0变 10 位；d 的 IDr 是 10 位 int → str
b["IDr"] = b["ID"].astype(str).str.strip().str.lstrip("0")
d["IDr_str"] = d["IDr"].astype(str).str.strip()

# 合并血糖/CRP/糖化/肾功能（按去前导0后的 ID）
for c in ["newglu", "newcrp", "newhba1c", "newcrea", "newua", "cystatinc"]:
    d[c] = d["IDr_str"].map(dict(zip(b["IDr"], pd.to_numeric(b[c], errors="coerce"))))

# ── 构造信号 ──
d["TyG"] = np.log(d["newtg"].clip(lower=1) * d["newglu"].clip(lower=1) / 2)  # 胰岛素抵抗代理
print(f"合并后样本 n={len(d)}")
print(f"血糖 newglu 有效 {d['newglu'].notna().sum()} | CRP newcrp 有效 {d['newcrp'].notna().sum()} | 糖化 newhba1c 有效 {d['newhba1c'].notna().sum()}")
print(f"新发卒中 stroke_fu={d['stroke_fu'].sum()} | 抑郁 dep={d['dep'].sum()}")

UP = ["newglu", "newtg", "newldl", "newhba1c", "TyG"]   # 上游信号
MID = ["newcrp"]                                          # 中游信号

print("\n" + "="*70)
print("【验证1】上游信号 → 中游炎症(CRP) 的 Spearman 相关（信号链是否\"亮\"）")
print("="*70)
crp = d["newcrp"]
for s in UP:
    sub = d[[s, "newcrp"]].dropna()
    if len(sub) > 50:
        r, p = stats.spearmanr(sub[s], sub["newcrp"])
        print(f"  {s:10s} ↔ CRP: ρ={r:+.3f}  P={p:.3g}  (n={len(sub)})")

print("\n" + "="*70)
print("【验证2】上游信号 → 下游卒中(stroke_fu) 单因素逻辑回归 OR")
print("="*70)
for s in UP + MID:
    sub = d[[s, "stroke_fu"]].dropna()
    if len(sub) > 50 and sub["stroke_fu"].sum() > 20:
        X = sm.add_constant(sub[s].astype(float))
        m = sm.Logit(sub["stroke_fu"], X).fit(disp=0)
        orr = np.exp(m.params.iloc[1]); p = m.pvalues.iloc[1]
        print(f"  {s:10s} → 卒中: OR={orr:.3f}  P={p:.3g}  (n={len(sub)}, 事件={int(sub['stroke_fu'].sum())})")

print("\n" + "="*70)
print("【验证3】上游信号 → 下游抑郁(dep) 单因素逻辑回归 OR")
print("="*70)
for s in UP + MID:
    sub = d[[s, "dep"]].dropna()
    if len(sub) > 50 and sub["dep"].sum() > 20:
        X = sm.add_constant(sub[s].astype(float))
        m = sm.Logit(sub["dep"], X).fit(disp=0)
        orr = np.exp(m.params.iloc[1]); p = m.pvalues.iloc[1]
        print(f"  {s:10s} → 抑郁: OR={orr:.3f}  P={p:.3g}  (n={len(sub)}, 事件={int(sub['dep'].sum())})")

print("\n" + "="*70)
print("【验证4】中游炎症(CRP) → 下游结局（中游是否\"传导\"到下游）")
print("="*70)
for outc in ["stroke_fu", "dep"]:
    sub = d[["newcrp", outc]].dropna()
    if len(sub) > 50 and sub[outc].sum() > 20:
        X = sm.add_constant(sub["newcrp"].astype(float))
        m = sm.Logit(sub[outc], X).fit(disp=0)
        orr = np.exp(m.params.iloc[1]); p = m.pvalues.iloc[1]
        print(f"  CRP → {outc}: OR={orr:.3f}  P={p:.3g}  (n={len(sub)}, 事件={int(sub[outc].sum())})")

print("\n" + "="*70)
print("【验证5】多信号联合 → 卒中（看三高+炎症组合的预测力）")
print("="*70)
cols = ["newglu", "newtg", "newcrp", "hibp", "age_w1", "female", "bmi", "stroke_fu"]
sub = d[cols].dropna()
if len(sub) > 100 and sub["stroke_fu"].sum() > 20:
    X = sm.add_constant(sub[["newglu", "newtg", "newcrp", "hibp", "age_w1", "female", "bmi"]].astype(float))
    m = sm.Logit(sub["stroke_fu"], X).fit(disp=0)
    print(f"  n={len(sub)}, 事件={int(sub['stroke_fu'].sum())}")
    for k in ["newglu", "newtg", "newcrp", "hibp", "age_w1", "female", "bmi"]:
        orr = np.exp(m.params[k]); p = m.pvalues[k]
        print(f"    {k:10s}: OR={orr:.3f}  P={p:.3g}")
