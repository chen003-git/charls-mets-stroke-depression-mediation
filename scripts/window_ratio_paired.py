# -*- coding: utf-8 -*-
"""论文19 · 配对 bootstrap：给「窗口倍数」与「窗口差值」各自的 95% CI

为什么需要这个脚本（回答"为什么原来没有 CI"）：
  - 严格窗口（stroke_early, r2/r3）的 800 次 bootstrap 在 run_mets_v2.py
  - 宽松窗口（stroke_any, r2/r3/r4）的 800 次 bootstrap 在 run_any_stroke.py
  - 两个脚本各自 resample、各自出 CI；比值 5.60 是**两次独立运行的删失点估计相除**
  - 没有任何一处同时持有两个 bootstrap 分布 → 比值的 CI 从未被计算

正确做法：配对 bootstrap —— 每一次重抽样都在**同一个样本**上同时算两个定义。
这样才捕捉到两者（同一队列、any 是 strict 的超集）之间的强相关。

输出：_output/window_ratio_paired.json
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_mets_v2 import build as build_mets          # noqa: E402
from run_mets_v2 import mediation_full as med_full  # noqa: E402

DATA = os.environ.get("P19_DATA") or os.path.join(os.path.dirname(HERE), "data")
OUT = os.path.join(os.path.dirname(HERE), "_output")
NB = int(os.environ.get("NB", "400"))
SEED = 42


def eff_both(dm, A, M1, M2, Y, covars):
    """在同一个数据集上拟合 strict 与 any 两个中介模型，返回两个 NIE。"""
    Xa1 = [A] + covars; Xb1 = [A, M1] + covars
    Xa2 = [A] + covars; Xb2 = [A, M2] + covars

    def addc(df, cols):
        return sm.add_constant(df[cols].astype(float), has_constant="add")

    ma1 = sm.Logit(dm[M1].astype(float), addc(dm, Xa1)).fit(disp=0)
    mb1 = sm.Logit(dm[Y].astype(float),  addc(dm, Xb1)).fit(disp=0)
    ma2 = sm.Logit(dm[M2].astype(float), addc(dm, Xa2)).fit(disp=0)
    mb2 = sm.Logit(dm[Y].astype(float),  addc(dm, Xb2)).fit(disp=0)

    def nie(ma, mb, M):
        p1 = ma.predict(addc(dm.assign(**{A: 1.0}), Xa1))
        p0 = ma.predict(addc(dm.assign(**{A: 0.0}), Xa1))
        y11 = mb.predict(addc(dm.assign(**{A: 1.0, M: 1.0}), Xb1 if M == M1 else Xb2))
        y10 = mb.predict(addc(dm.assign(**{A: 1.0, M: 0.0}), Xb1 if M == M1 else Xb2))
        return float(np.mean(p1 * y11 + (1 - p1) * y10) - np.mean(p0 * y11 + (1 - p0) * y10))

    return nie(ma1, mb1, M1), nie(ma2, mb2, M2)


def main():
    h = pd.read_stata(os.path.join(DATA, r"Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta"),
                      convert_categoricals=False)
    d0 = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    d = build_mets(d0, h)          # build(d, h)：算出 mets / mets2 / stroke_early 并合并
    # any = r2/r3/r4 任一（与 run_any_stroke.py 一致）
    h2 = h[["ID", "r2stroke", "r3stroke", "r4stroke"]].copy()
    h2["IDr"] = h2["ID"].astype(str).str.strip()
    st = h2[["r2stroke", "r3stroke", "r4stroke"]].apply(pd.to_numeric, errors="coerce")
    h2["stroke_any"] = ((st == 1).any(axis=1)).astype(float)
    d["IDr"] = d["ID"].astype(str).str.strip().str.zfill(12)
    h2["IDr"] = h2["IDr"].str.zfill(12)
    d = d.merge(h2[["IDr", "stroke_any"]], on="IDr", how="left")

    cov = [c for c in ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]
           if c in d.columns and d[c].notna().sum() > 1000]
    dd = d.dropna(subset=["mets", "stroke_early", "stroke_any", "dep"] + cov).copy()
    A, M1, M2, Y = "mets", "stroke_early", "stroke_any", "dep"
    print(f"n={len(dd)} | strict 事件={int(dd[M1].sum())} | any 事件={int(dd[M2].sum())} | 抑郁={int(dd[Y].sum())}", flush=True)

    nie1, nie2 = eff_both(dd, A, M1, M2, Y, cov)
    print(f"点估计: NIE_strict={nie1:.6f}  NIE_any={nie2:.6f}  差={nie2-nie1:.6f}  比值={nie2/nie1:.3f}", flush=True)

    rng = np.random.default_rng(SEED)
    rec = []
    t0 = time.time()
    for i in range(NB):
        s = dd.sample(len(dd), replace=True, random_state=int(rng.integers(1e9)))
        try:
            a, b = eff_both(s, A, M1, M2, Y, cov)
            rec.append((a, b))
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{NB}  {time.time()-t0:.0f}s", flush=True)
    arr = np.array(rec)
    n1, n2 = arr[:, 0], arr[:, 1]
    diff = n2 - n1
    ratio = np.where(np.abs(n1) > 1e-9, n2 / np.where(np.abs(n1) > 1e-9, n1, np.nan), np.nan)
    pct = lambda x, q: float(np.nanpercentile(x, q))
    out = {
        "n": int(len(dd)), "n_strict_events": int(dd[M1].sum()), "n_any_events": int(dd[M2].sum()),
        "n_boot_requested": NB, "n_boot_ok": int(len(arr)),
        "NIE_strict": nie1, "NIE_strict_lo": pct(n1, 2.5), "NIE_strict_hi": pct(n1, 97.5),
        "NIE_any": nie2, "NIE_any_lo": pct(n2, 2.5), "NIE_any_hi": pct(n2, 97.5),
        "difference": float(nie2 - nie1),
        "difference_lo": pct(diff, 2.5), "difference_hi": pct(diff, 97.5),
        "ratio": float(nie2 / nie1),
        "ratio_lo": pct(ratio, 2.5), "ratio_hi": pct(ratio, 97.5),
        "ratio_p_gt1": float(np.nanmean(ratio > 1)),
        "ratio_p_gt5": float(np.nanmean(ratio > 5)),
        "corr_boot": float(np.corrcoef(n1, n2)[0, 1]),
        "method": "paired bootstrap (same resample for both mediator definitions), percentile CI",
        "note": "Paper19 previously reported ratio=5.60 from two INDEPENDENT bootstrap runs; this script pairs them.",
    }
    json.dump(out, open(os.path.join(OUT, "window_ratio_paired.json"), "w"), indent=2)
    print("\n=== 配对 bootstrap 结果 ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}" if abs(v) < 100 else f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print("\n已写 window_ratio_paired.json")


if __name__ == "__main__":
    main()
