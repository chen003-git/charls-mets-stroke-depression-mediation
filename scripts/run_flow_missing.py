#!/usr/bin/env python3
"""论文19：样本流转 / 缺失率 / 缺失预测因子（补齐此前无生成脚本的三个产物）

输出（全部到 _output/）：
  flow_counts.json              样本流转计数（正文 §3.1、Figure S1）
  missingness.json              组分缺失率，**键名显式区分分母**（全队列 / 分析样本）
  missingness_predictors.json   预测"2018 结局缺失"的协变量 OR（正文 §4.5 (iii)）

修复记录（阶段4）：
  · 本脚本此前**不存在** —— 这三个产物曾由一次性手工操作落盘，无法由 run_all.py 复现。
  · `missingness.json` 原键名 `*_pct_analytic` 装载的其实是**全队列**值（54.84/54.82），
    与正文 §3.1 的 27.3%/27.2%（分析样本）冲突 —— 现显式拆成 `*_full_cohort` / `*_analytic`。
  · `missingness_predictors.json` 原键名 `early_stroke` 装载的其实是
    **any stroke（含 2018 波）** 的 OR（0.584），**不是**稿件定义的早期卒中（r2|r3）；
    而正文 §4.5(iii) 据它写"early incident stroke … in the opposite direction (OR 0.584, P<0.001)"，
    与真实值**方向相反且不显著**。现拆成 `early_stroke_r2r3` 与 `any_stroke_incl2018` 两个键。

口径与 clean_harmonized.py 一致：
  · 第 1 层 全队列（排除基线卒中后）= data/analysis_harmonized.csv 的行数（n = 25,100）
  · 第 2 层 协变量完整（age/sex/edu/smoke/drink/baseline CES-D-10 均非缺失）
  · 第 3 层 结局非缺失 = 主分析样本
  · 缺失预测因子 = 在第 2 层内对「2018 结局缺失」做**逐变量单变量 logistic**
    （此规格由 age/edu/smoke 三个正值对照 1.0590 / 1.3343 / 1.1402 反推确认）
  · 组分可测性：waist 来自 H_CHARLS_D_Data.dta 的 r1mwaist（**不在 analysis_harmonized.csv 里**）
"""
import os, sys, json, math
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "_output")
HARM = os.path.join(DATA, "Regular Waves", "H_CHARLS_D_Data", "H_CHARLS_D_Data.dta")
COV = ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"]


def num(s): return pd.to_numeric(s, errors="coerce")


def to_raw(x):
    x = str(x).strip()
    return int(x[:10] + str(int(x[10:]))) if len(x) == 12 else None


def logit(X, y, maxit=100, tol=1e-10):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30); p = 1 / (1 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-10); z = eta + (y - p) / W
        try:
            bn = np.linalg.solve((X.T * W) @ X, (X.T * W) @ z)
        except np.linalg.LinAlgError:
            return None, None
        if np.max(np.abs(bn - b)) < tol:
            b = bn; break
        b = bn
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = np.maximum(p * (1 - p), 1e-12)
    se = np.sqrt(np.diag(np.linalg.pinv((X.T * W) @ X)))
    return b, se


def p2(z): return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def fit(df, y, xs):
    s = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(s))] + [s[c].values.astype(float) for c in xs])
    b, se = logit(X, s[y].values)
    if b is None:
        return None
    z = b[1] / se[1]
    return dict(n=int(len(s)), OR=round(float(np.exp(b[1])), 4),
                lo=round(float(np.exp(b[1] - 1.96 * se[1])), 4),
                hi=round(float(np.exp(b[1] + 1.96 * se[1])), 4),
                P=float(p2(z)))


def main():
    print("=" * 58); print("论文19 · 样本流转 / 缺失率 / 缺失预测因子"); print("=" * 58)

    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    h = pd.read_stata(HARM, convert_categoricals=False)

    raw_respondents = int(h.shape[0])
    stroke_w1 = num(h["r1stroke"]) if "r1stroke" in h.columns else pd.Series(0, index=h.index)
    excluded_prevalent_stroke = int((stroke_w1 == 1).sum())
    after = len(d)

    cov_complete = d.dropna(subset=COV)
    covariate_complete = int(len(cov_complete))
    excluded_missing_covariates = after - covariate_complete
    analytic_sample = int(len(cov_complete.dropna(subset=["dep"])))
    excluded_missing_outcome = covariate_complete - analytic_sample

    hh = pd.DataFrame({"IDr": h["ID"].map(to_raw), "waist": num(h["r1mwaist"]),
                       "r2": num(h["r2stroke"]), "r3": num(h["r3stroke"]),
                       "r4": num(h["r4stroke"])}).dropna(subset=["IDr"])
    d = d.merge(hh, on="IDr", how="left")
    an = d.dropna(subset=COV + ["dep"]).copy()

    mediator_unascertainable = int((an["r2"].isna() & an["r3"].isna()).sum())

    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d.loc[d["waist"].isna(), "c_waist"] = np.nan
    d["c_tg"] = (d["newtg"] >= 150).astype(float); d.loc[d["newtg"].isna(), "c_tg"] = np.nan
    d["c_hdl"] = (d["newhdl"] < 40).astype(float); d.loc[d["newhdl"].isna(), "c_hdl"] = np.nan
    d["c_bp"] = d["hibp"].astype(float)
    d["c_glu"] = d["diab"].astype(float)
    CUR = ["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]
    full = d.copy()
    ana = d.dropna(subset=COV + ["dep"]).copy()
    union = int(ana[CUR].isna().any(axis=1).sum())
    pct = round(union / len(ana) * 100, 2)

    # 仅在 2018 波首次报告的卒中（正文 §3.3 的 624）
    n_first_2018 = int(((ana["r2"].fillna(0) != 1) & (ana["r3"].fillna(0) != 1) & (ana["r4"].fillna(0) == 1)).sum())

    flow = {
        "raw_respondents": raw_respondents,
        "excluded_prevalent_stroke": excluded_prevalent_stroke,
        "after_stroke_exclusion": after,
        "covariate_complete": covariate_complete,
        "excluded_missing_covariates": excluded_missing_covariates,
        "excluded_missing_outcome": excluded_missing_outcome,
        "analytic_sample": analytic_sample,
        "mediator_unascertainable": mediator_unascertainable,
        "mediator_missing_dropped_n": analytic_sample - mediator_unascertainable,
        "component_missing_union": union,
        "component_missing_pct_of_analytic": pct,
        "n_stroke_first_reported_2018": n_first_2018,
        "n_stroke_early_r2r3": int((((ana["r2"].fillna(0) == 1) | (ana["r3"].fillna(0) == 1))).sum()),
        "n_stroke_any_incl2018": int((ana[["r2", "r3", "r4"]].fillna(0) == 1).any(axis=1).sum()),
        "component_missing_scope": ("analytic sample (n=%s); union over waist(r1mwaist), TG, HDL, "
                                    "self-reported hypertension, self-reported diabetes" % f"{analytic_sample:,}"),
        "mediator_unascertainable_note": "r2stroke and r3stroke both missing; counted as free of early stroke in primary",
        "component_missing_note": ("waist (%s missing) comes from H_CHARLS_D_Data.dta and is not a column in "
                                   "analysis_harmonized.csv; a recomputation from that CSV alone is wrong."
                                   % f"{int(ana['waist'].isna().sum()):,}"),
    }
    print(f"  全队列 {raw_respondents:,} -> 排除基线卒中 {excluded_prevalent_stroke} -> {after:,}")
    print(f"  协变量完整 {covariate_complete:,}（剔除 {excluded_missing_covariates:,}）"
          f"-> 结局完整 {analytic_sample:,}（剔除 {excluded_missing_outcome:,}）")
    print(f"  中介不可判定 {mediator_unascertainable} | 组分并集缺失 {union:,} ({pct}%)")
    print(f"  早期卒中(r2|r3) {flow['n_stroke_early_r2r3']} | 仅 2018 首报 {n_first_2018} | 任意含2018 {flow['n_stroke_any_incl2018']}")

    def miss(frame):
        return {c: round(float(frame[c].isna().mean() * 100), 2) for c in CUR}
    mf, ma = miss(full), miss(ana)
    missingness = {
        "n_full_cohort": int(len(full)),
        "n_analytic": int(len(ana)),
        "note": ("全队列 = 排除基线卒中后的全样本（n=%s）；分析样本 = 主分析样本（n=%s）。"
                 "正文 §3.1 的 27.3%% / 27.2%% / 12.7%% 用 analytic 口径。"
                 % (f"{len(full):,}", f"{len(ana):,}")),
        "tg_missing_pct_full_cohort": mf["c_tg"],
        "hdl_missing_pct_full_cohort": mf["c_hdl"],
        "waist_missing_pct_full_cohort": mf["c_waist"],
        "tg_missing_pct_analytic": ma["c_tg"],
        "hdl_missing_pct_analytic": ma["c_hdl"],
        "waist_missing_pct_analytic": ma["c_waist"],
        "waist_missing_n_analytic": int(ana["waist"].isna().sum()),
    }
    print(f"  TG 缺失 全队列 {mf['c_tg']}% / 分析样本 {ma['c_tg']}%"
          f" | HDL {mf['c_hdl']}% / {ma['c_hdl']}% | waist {mf['c_waist']}% / {ma['c_waist']}%")

    # ── 缺失预测因子：协变量完整层内，逐变量单变量 logistic，因变量 = 2018 结局缺失 ──
    base = d.dropna(subset=COV).copy()
    base["outcome_missing"] = base["dep"].isna().astype(float)
    base["mets"] = (base[CUR].fillna(0).sum(axis=1) >= 3).astype(float)
    base["early_stroke_r2r3"] = ((base["r2"].fillna(0) == 1) | (base["r3"].fillna(0) == 1)).astype(float)
    base["any_stroke_incl2018"] = (base[["r2", "r3", "r4"]].fillna(0) == 1).any(axis=1).astype(float)
    preds = {"spec": "univariable logistic within the covariate-complete layer (n=15,496); "
                     "outcome = 2018 CES-D-10 missing",
             "note": "键 early_stroke_r2r3 用稿件定义的早期卒中（r2|r3）；键 any_stroke_incl2018 用含 2018 波的任意卒中。"}
    for lab, col in [("age", "age_w1"), ("female", "female"), ("edu", "edu"),
                     ("smoke", "smoke"), ("drink", "drink"), ("mets", "mets"),
                     ("early_stroke_r2r3", "early_stroke_r2r3"),
                     ("any_stroke_incl2018", "any_stroke_incl2018")]:
        r = fit(base, "outcome_missing", [col])
        if r:
            preds[lab] = r
            print(f"  {lab:<22} n={r['n']:,}  OR={r['OR']:.4f} [{r['lo']:.4f},{r['hi']:.4f}]  P={r['P']:.6f}")

    os.makedirs(OUT, exist_ok=True)
    for name, obj in [("flow_counts", flow), ("missingness", missingness),
                      ("missingness_predictors", preds)]:
        json.dump(obj, open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=float)
        print(f"  OK 已写 {name}.json")


if __name__ == "__main__":
    main()
