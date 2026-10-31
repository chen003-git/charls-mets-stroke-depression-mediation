#!/usr/bin/env python3
"""论文19 阶段4 新增分析（回应三位审稿人的 Critical/Major）

输出到 _output/：
  suppression_joint.json    A-C1：抑制模式（NDE<0 且 NIE>0）自身的 bootstrap 概率
  mediator_window.json      A §5.5 / B-C1：把"仅 2018 波首次报告"的卒中单独作为中介口径分解
  misclass_fragility.json   A-M5：把 k 个中介事件从 1 改 0，看 NIE 是否仍排除 0

口径与 run_mets_v2.py 的主分析完全一致（statsmodels Logit；协变量 age/sex/edu/smoke/drink/baseline CES-D；
BMI 排除；中介缺失记为阴性）。
"""
import os, sys, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "_output")
N_JOINT = 500      # §3.2 抑制模式联合概率的重抽样次数
N_WINDOW = 200     # §3.3 三个中介口径的 CI 重抽样次数
N_FRAG = 80        # §3.2 中介错分脆弱性每个 k 的重抽样次数

HARM = os.path.join(DATA, "Regular Waves", "H_CHARLS_D_Data", "H_CHARLS_D_Data.dta")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def to_raw(x):
    x = str(x).strip()
    return int(x[:10] + str(int(x[10:]))) if len(x) == 12 else None


def _mm(df, cols):
    return sm.add_constant(df[cols].astype(float), has_constant="add")


def effects(dm, A, M, Y, covars, mm, my):
    """返回 (NDE, NIE) —— 与 run_mets_v2.py 的 eff() 同一算法。"""
    XmA = [A] + covars; Xy = [A, M] + covars
    p1 = mm.predict(_mm(dm.assign(**{A: 1.0}), XmA)); p0 = mm.predict(_mm(dm.assign(**{A: 0.0}), XmA))
    y1_m1 = my.predict(_mm(dm.assign(**{A: 1.0, M: 1.0}), Xy)); y1_m0 = my.predict(_mm(dm.assign(**{A: 1.0, M: 0.0}), Xy))
    y0_m1 = my.predict(_mm(dm.assign(**{A: 0.0, M: 1.0}), Xy)); y0_m0 = my.predict(_mm(dm.assign(**{A: 0.0, M: 0.0}), Xy))
    ey1m1 = np.mean(p1 * y1_m1 + (1 - p1) * y1_m0)
    ey1m0 = np.mean(p0 * y1_m1 + (1 - p0) * y1_m0)
    ey0m0 = np.mean(p0 * y0_m1 + (1 - p0) * y0_m0)
    return ey1m0 - ey0m0, ey1m1 - ey1m0


def fit_pair(dm, A, M, Y, covars):
    XmA = [A] + covars; Xy = [A, M] + covars
    mm = sm.Logit(dm[M].astype(float), _mm(dm, XmA)).fit(disp=0)
    my = sm.Logit(dm[Y].astype(float), _mm(dm, Xy)).fit(disp=0)
    return mm, my


def build():
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    h = pd.read_stata(HARM, convert_categoricals=False)
    hh = pd.DataFrame({"IDr": h["ID"].map(to_raw), "waist": num(h["r1mwaist"]),
                       "r2": num(h["r2stroke"]), "r3": num(h["r3stroke"]),
                       "r4": num(h["r4stroke"])}).dropna(subset=["IDr"])
    # 合并口径必须与 run_mets_v2.py / run_flow_missing.py / run_sensitivity_pack.py 完全一致：
    # `analysis_harmonized.csv` 的 IDr 已是 to_raw() 产出的整数，直接用整数键 merge。
    # （曾误用 zfill(12) 字符串去匹配 to_raw 的整数字符串 ⇒ 映射全 NaN ⇒ stroke 全 0 ⇒ Logit 奇异。
    #   下方自检即为防此复发。）
    d = d.merge(hh, on="IDr", how="left")
    cov = [c for c in ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"] if c in d.columns]
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d["c_tg"] = (d["newtg"] >= 150).astype(float)
    d["c_hdl"] = (d["newhdl"] < 40).astype(float)
    d["c_bp"] = d["hibp"].astype(float)
    d["c_glu"] = d["diab"].astype(float)
    d["mets"] = (d[["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]].fillna(0).sum(axis=1) >= 3).astype(int)
    d["stroke_early"] = ((d["r2"].fillna(0) == 1) | (d["r3"].fillna(0) == 1)).astype(float)
    d["stroke_only2018"] = ((d["r2"].fillna(0) != 1) & (d["r3"].fillna(0) != 1) & (d["r4"].fillna(0) == 1)).astype(float)
    d["stroke_any"] = (d[["r2", "r3", "r4"]].fillna(0) == 1).any(axis=1).astype(float)
    dd = d.dropna(subset=["mets", "stroke_early", "dep"] + cov).copy()
    n_med = int(dd["stroke_early"].sum()); n_any = int(dd["stroke_any"].sum())
    if len(dd) < 10000 or not (150 <= n_med <= 200) or not (700 <= n_any <= 900):
        raise SystemExit(f"[中止] 自检未过：n={len(dd)} 早期卒中={n_med}（期望 172）"
                         f" 任意卒中={n_any}（期望 796）—— 很可能是 ID 合并不上")
    print(f"  [自检通过] n={len(dd)} 早期卒中={n_med} 任意卒中={n_any}")
    return d, dd, cov


def main():
    print("=" * 60); print("论文19 阶段4 新增分析（回应审稿人 A/B）"); print("=" * 60)
    d, dd, cov = build()
    print(f"  分析样本 n = {len(dd)} | MetS {int(dd['mets'].sum())} | 早期卒中 {int(dd['stroke_early'].sum())}"
          f" | 仅2018首报 {int(dd['stroke_only2018'].sum())} | 任意含2018 {int(dd['stroke_any'].sum())}")
    res = {}

    # ---------- 1) A-C1：抑制模式的联合概率 ----------
    print("\n[1] suppression_joint —— 成对 bootstrap，P(NDE<0 且 NIE>0)")
    A, M, Y = "mets", "stroke_early", "dep"
    mm, my = fit_pair(dd, A, M, Y, cov)
    nde0, nie0 = effects(dd, A, M, Y, cov, mm, my)
    NB = N_JOINT
    rng = np.random.default_rng(42)
    nde_b, nie_b = [], []
    for i in range(NB):
        s = dd.sample(len(dd), replace=True, random_state=int(rng.integers(1e9)))
        try:
            m2, y2 = fit_pair(s, A, M, Y, cov)
            n_, e_ = effects(s, A, M, Y, cov, m2, y2)
            nde_b.append(n_); nie_b.append(e_)
        except Exception:
            pass
    nde_b = np.array(nde_b); nie_b = np.array(nie_b)
    joint = dict(
        n_boot_requested=NB, n_boot_ok=int(len(nde_b)), seed=42,
        NDE_point=float(nde0), NIE_point=float(nie0),
        P_NIE_gt0=float((nie_b > 0).mean()),
        P_NDE_lt0=float((nde_b < 0).mean()),
        P_suppression_NDElt0_and_NIEgt0=float(((nde_b < 0) & (nie_b > 0)).mean()),
        P_NIE_le0=float((nie_b <= 0).mean()),
        corr_NDE_NIE=float(np.corrcoef(nde_b, nie_b)[0, 1]),
        NIE_lo=float(np.percentile(nie_b, 2.5)), NIE_hi=float(np.percentile(nie_b, 97.5)),
        NDE_lo=float(np.percentile(nde_b, 2.5)), NDE_hi=float(np.percentile(nde_b, 97.5)),
        note=("inconsistent/suppression mediation 的定义是 a*b 与 c' 异号；"
              "本文件给出该联合事件的重抽样概率，用于判断 'suppression' 这一措辞能否作为结论。"),
    )
    print(f"  P(NIE>0)={joint['P_NIE_gt0']:.3f}  P(NDE<0)={joint['P_NDE_lt0']:.3f}"
          f"  P(NDE<0 且 NIE>0)={joint['P_suppression_NDElt0_and_NIEgt0']:.3f}"
          f"  corr={joint['corr_NDE_NIE']:.3f}")
    res["suppression_joint"] = joint

    # ---------- 2) A §5.5 / B-C1：三个中介口径 + 仅 2018 首报 ----------
    print("\n[2] mediator_window —— 三个中介口径的 NDE/NIE/TE（800 次 bootstrap）")
    mw = {}
    for lab, col in [("early_2013_or_2015", "stroke_early"),
                     ("only_2018_first_reported", "stroke_only2018"),
                     ("any_incl_2018", "stroke_any")]:
        dsub = dd.dropna(subset=[col]).copy()
        m_, y_ = fit_pair(dsub, "mets", col, "dep", cov)
        n0, e0 = effects(dsub, "mets", col, "dep", cov, m_, y_)
        rng2 = np.random.default_rng(42); nb_, eb_ = [], []
        for i in range(N_WINDOW):
            s = dsub.sample(len(dsub), replace=True, random_state=int(rng2.integers(1e9)))
            try:
                m2, y2 = fit_pair(s, "mets", col, "dep", cov)
                nn, ee = effects(s, "mets", col, "dep", cov, m2, y2)
                nb_.append(nn); eb_.append(ee)
            except Exception:
                pass
        nb_ = np.array(nb_); eb_ = np.array(eb_)
        # 路径 OR
        orA = sm.Logit(dsub[col].astype(float), _mm(dsub, ["mets"] + cov)).fit(disp=0)
        orB = sm.Logit(dsub["dep"].astype(float), _mm(dsub, ["mets", col] + cov)).fit(disp=0)
        mw[lab] = dict(
            n=int(len(dsub)), n_mediator=int(dsub[col].sum()),
            NDE=float(n0), NIE=float(e0), TE=float(n0 + e0),
            NIE_lo=float(np.percentile(eb_, 2.5)), NIE_hi=float(np.percentile(eb_, 97.5)),
            NDE_lo=float(np.percentile(nb_, 2.5)), NDE_hi=float(np.percentile(nb_, 97.5)),
            path_a_OR=float(np.exp(orA.params["mets"])),
            path_a_lo=float(np.exp(orA.conf_int().loc["mets", 0])), path_a_hi=float(np.exp(orA.conf_int().loc["mets", 1])),
            path_b_OR=float(np.exp(orB.params[col])),
            path_b_lo=float(np.exp(orB.conf_int().loc[col, 0])), path_b_hi=float(np.exp(orB.conf_int().loc[col, 1])),
            n_boot_requested=N_WINDOW, n_boot_ok=int(len(eb_)),
        )
        print(f"  {lab:<26} n={len(dsub):<6} n_med={int(dsub[col].sum()):<5} NIE={e0:+.6f}"
              f" [{mw[lab]['NIE_lo']:+.6f},{mw[lab]['NIE_hi']:+.6f}]  a_OR={mw[lab]['path_a_OR']:.3f}  b_OR={mw[lab]['path_b_OR']:.3f}")
    a_, b_, c_ = mw["early_2013_or_2015"], mw["only_2018_first_reported"], mw["any_incl_2018"]
    mw["fold_only2018_vs_early"] = round(b_["NIE"] / a_["NIE"], 2)
    mw["fold_any_vs_early"] = round(c_["NIE"] / a_["NIE"], 2)
    mw["note"] = ("`only_2018_first_reported` 是 r2/r3 均为 0 且 r4==1 的人 —— 其卒中发生时间落在 2015 末至 2018 访问日之间，"
                  "属时序合法的中介；`any_incl_2018` 额外含与结局同波次报告者。")
    print(f"  ⇒ 仅2018/早期 = {mw['fold_only2018_vs_early']} 倍；任意含2018/早期 = {mw['fold_any_vs_early']} 倍")
    res["mediator_window"] = mw

    # ---------- 3) A-M5：中介错分脆弱性 ----------
    print("\n[3] misclass_fragility —— 把 k 个中介事件从 1 改 0 后 NIE 是否仍排除 0")
    pos = dd.index[dd["stroke_early"] == 1].to_numpy()
    frag = []
    for k in [0, 3, 12, 30]:
        dd2 = dd.copy()
        if k:
            drop = np.random.default_rng(7).choice(pos, size=k, replace=False)
            dd2.loc[drop, "stroke_early"] = 0.0
        m_, y_ = fit_pair(dd2, "mets", "stroke_early", "dep", cov)
        n0, e0 = effects(dd2, "mets", "stroke_early", "dep", cov, m_, y_)
        rng3 = np.random.default_rng(42); eb2 = []
        for i in range(N_FRAG):
            s = dd2.sample(len(dd2), replace=True, random_state=int(rng3.integers(1e9)))
            try:
                m2, y2 = fit_pair(s, "mets", "stroke_early", "dep", cov)
                _, ee = effects(s, "mets", "stroke_early", "dep", cov, m2, y2)
                eb2.append(ee)
            except Exception:
                pass
        eb2 = np.array(eb2)
        lo = float(np.percentile(eb2, 2.5))
        frag.append(dict(k_reclassified=int(k), NIE=float(e0), NIE_lo=lo,
                         NIE_hi=float(np.percentile(eb2, 97.5)), excludes_zero=bool(lo > 0),
                         n_boot=N_FRAG))
        print(f"  k={k:<3} NIE={e0:+.6f}  95% CI 下界={lo:+.6f}  排除0={'是' if lo > 0 else '否'}")
    res["misclass_fragility"] = dict(
        rows=frag, seed_reclassify=7, seed_boot=42, n_mediator=int((dd["stroke_early"] == 1).sum()),
        note=("把 k 个中介事件从 1 改 0（随机、seed=7）后重跑整套分解；"
              "检验的是**数据层面**的中介错分敏感性，比只改 bootstrap 种子更有信息量。"))

    for k, v in res.items():
        json.dump(v, open(os.path.join(OUT, k + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)
        print(f"  OK 已写 {k}.json")


if __name__ == "__main__":
    main()
