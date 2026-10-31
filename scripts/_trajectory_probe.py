#!/usr/bin/env python3
"""纵向轨迹验证：信号变化率（RUL 逻辑）vs 信号绝对值 对后续卒中的预测力
核心问题：信号的趋势(Δ) 是否比信号的绝对值(基线) 更能预测未来事件？
数据：Harmonized r1/r2/r3mbmi、r1/r2/r3mwaist（3波实测）+ 卒中 r1-r4
时间对齐：Δ(2011→2013) → 后续卒中(2015/2018)
"""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import numpy as np, pandas as pd
import statsmodels.api as sm

HARM = r"D:\SCI论文\19_CHARLS疾病链\data\Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta"
h = pd.read_stata(HARM, convert_categoricals=False)

d = pd.DataFrame()
d["ID"] = h["ID"].astype(str)
d["age"] = pd.to_numeric(h["r1agey"], errors="coerce")
d["female"] = (pd.to_numeric(h["ragender"], errors="coerce") == 2).astype(int)
d["hibp"] = (pd.to_numeric(h["r1hibpe"], errors="coerce") == 1).astype(int)
d["diab"] = (pd.to_numeric(h["r1diabe"], errors="coerce") == 1).astype(int)

# 纵向信号（3 波实测）
d["bmi1"] = pd.to_numeric(h["r1mbmi"], errors="coerce")
d["bmi2"] = pd.to_numeric(h["r2mbmi"], errors="coerce")
d["bmi3"] = pd.to_numeric(h["r3mbmi"], errors="coerce")
d["waist1"] = pd.to_numeric(h["r1mwaist"], errors="coerce")
d["waist2"] = pd.to_numeric(h["r2mwaist"], errors="coerce")
d["waist3"] = pd.to_numeric(h["r3mwaist"], errors="coerce")

# 卒中（4 波）
d["s1"] = (pd.to_numeric(h["r1stroke"], errors="coerce") == 1).astype(int)
d["s3"] = (pd.to_numeric(h["r3stroke"], errors="coerce") == 1).astype(int)
d["s4"] = (pd.to_numeric(h["r4stroke"], errors="coerce") == 1).astype(int)

# 排除基线卒中
d = d[d["s1"] != 1].copy()

# ── 变化率（RUL 信号：趋势）──
d["dbmi12"] = d["bmi2"] - d["bmi1"]        # 2011→2013 BMI 变化
d["dwaist12"] = d["waist2"] - d["waist1"]  # 2011→2013 腰围变化
# 后续卒中（2015 或 2018，在轨迹之后）
d["stroke_after"] = ((d["s3"] == 1) | (d["s4"] == 1)).astype(int)

print(f"排除基线卒中后 n={len(d)}")
print(f"BMI 3波全有效: {d[['bmi1','bmi2','bmi3']].notna().all(axis=1).sum()}")
print(f"腰围 3波全有效: {d[['waist1','waist2','waist3']].notna().all(axis=1).sum()}")
print(f"轨迹后新发卒中: {d['stroke_after'].sum()}")

def logit_report(x, y, label):
    sub = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(sub) < 100 or sub["y"].sum() < 20:
        print(f"  {label}: 样本不足 (n={len(sub)}, 事件={sub['y'].sum()})")
        return None
    X = sm.add_constant(sub["x"].astype(float))
    m = sm.Logit(sub["y"], X).fit(disp=0)
    return (label, np.exp(m.params.iloc[1]), m.pvalues.iloc[1], len(sub), int(sub["y"].sum()))

print("\n" + "="*72)
print("【核心对比】信号变化率(Δ) vs 信号绝对值(基线) 对后续卒中的预测力")
print("="*72)
rows = []
for lab, x in [("基线 BMI (bmi1)", d["bmi1"]),
               ("BMI 变化率 Δ(2011→13)", d["dbmi12"]),
               ("基线 腰围 (waist1)", d["waist1"]),
               ("腰围变化率 Δ(2011→13)", d["dwaist12"])]:
    r = logit_report(x, d["stroke_after"], lab)
    if r:
        rows.append(r)
        print(f"  {r[0]:28s} OR={r[1]:.4f}  P={r[2]:.3g}  (n={r[3]}, 事件={r[4]})")

print("\n" + "="*72)
print("【趋势方向拆分】BMI 变化率分三组：明显下降/稳定/明显上升 → 卒中")
print("="*72)
sub = d[["dbmi12", "stroke_after", "age", "female"]].dropna()
if len(sub) > 100:
    q = sub["dbmi12"].quantile([0.25, 0.75])
    sub["grp"] = np.where(sub["dbmi12"] < q.iloc[0], "下降(<-" + f"{q.iloc[0]:.1f})",
                 np.where(sub["dbmi12"] > q.iloc[1], "上升(>" + f"{q.iloc[1]:.1f})", "稳定"))
    g = sub.groupby("grp")["stroke_after"].agg(["count", "mean"])
    print(g.to_string())
    # 下降组 vs 稳定组 OR
    sub["down"] = (sub["grp"] == "下降(<-" + f"{q.iloc[0]:.1f})").astype(int)
    X = sm.add_constant(sub[["down", "age", "female"]].astype(float))
    m = sm.Logit(sub["stroke_after"], X).fit(disp=0)
    print(f"\n  BMI 明显下降 vs 其余（调整年龄/性别）: OR={np.exp(m.params['down']):.3f}  P={m.pvalues['down']:.3g}")

print("\n" + "="*72)
print("【多因素】基线BMI + BMI变化率 联合 → 后续卒中（变化率是否有独立增量）")
print("="*72)
sub = d[["bmi1", "dbmi12", "stroke_after", "age", "female", "hibp", "diab"]].dropna()
if len(sub) > 100 and sub["stroke_after"].sum() > 20:
    X = sm.add_constant(sub[["bmi1", "dbmi12", "age", "female", "hibp", "diab"]].astype(float))
    m = sm.Logit(sub["stroke_after"], X).fit(disp=0)
    print(f"  n={len(sub)}, 事件={int(sub['stroke_after'].sum())}")
    for k in ["bmi1", "dbmi12", "age", "female", "hibp", "diab"]:
        print(f"    {k:10s}: OR={np.exp(m.params[k]):.4f}  P={m.pvalues[k]:.3g}")
