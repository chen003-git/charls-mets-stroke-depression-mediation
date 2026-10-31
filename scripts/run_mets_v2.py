#!/usr/bin/env python3
"""论文19 重做：标准 MetS(≥3/5, 含腰围) 中介分析 + 亚组 + 阴性对照 + 种子敏感性

审稿①：改用腰围(男≥90/女≥85)替代 BMI，MetS 用标准 ≥3/5（IDF 2005 / NCEP ATP III 修订版）
审稿②：TE 报正确 CI/P（不再用 NDE 凑数）
审稿③：固定 seed + 报告多种子 NIE_lo 敏感性
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

def _mm(df, cols): return sm.add_constant(df[cols].astype(float), has_constant="add")

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
        ey1m1 = np.mean(p1*y1_m1+(1-p1)*y1_m0); ey1m0 = np.mean(p0*y1_m1+(1-p0)*y1_m0); ey0m0 = np.mean(p0*y0_m1+(1-p0)*y0_m0)
        return ey1m0 - ey0m0, ey1m1 - ey1m0
    mm, my = fit(d); nde, nie = eff(d, mm, my); te = nde + nie
    rng = np.random.default_rng(seed); bo = []; n_fail = 0; fail_msg = None; n_conv = 0
    # 修复（阶段4 / 审稿人B+C）：原实现只捕异常，不捕 ConvergenceWarning，
    # 故 "n_boot_failed": 0 是死代码 —— 它只说明"没有异常抛出"，不代表"全部收敛"。
    # 现在显式统计收敛警告。
    import warnings as _w
    for _ in range(n_boot):
        s = d.sample(len(d), replace=True, random_state=int(rng.integers(1e9)))
        try:
            with _w.catch_warnings(record=True) as _wl:
                _w.simplefilter("always")
                m2, y2 = fit(s); n2, e2 = eff(s, m2, y2); bo.append((n2, e2, n2 + e2))
            n_conv += sum(1 for _x in _wl if "Convergence" in _x.category.__name__)
        except Exception as _e:
            n_fail += 1
            if fail_msg is None: fail_msg = str(_e)[:80]
    b = np.array(bo); pct = lambda a, q: float(np.percentile(a, q)) if len(a) else np.nan
    te_boot = b[:, 2]
    te_p = 2 * min((te_boot <= 0).mean(), (te_boot >= 0).mean()) if len(te_boot) else np.nan
    if n_fail:
        print(f"  注意：{n_boot} 次重抽样中 {n_fail} 次未收敛（{n_fail/n_boot*100:.2f}%）；"
              f"置信区间基于成功收敛的 {len(b)} 次。首个错误：{fail_msg}")
    return {"NDE": nde, "NDE_lo": pct(b[:, 0], 2.5), "NDE_hi": pct(b[:, 0], 97.5),
            "NIE": nie, "NIE_lo": pct(b[:, 1], 2.5), "NIE_hi": pct(b[:, 1], 97.5),
            "TE": te, "TE_lo": pct(te_boot, 2.5), "TE_hi": pct(te_boot, 97.5), "TE_p": te_p,
            "n_boot_requested": int(n_boot), "n_boot_ok": int(len(b)), "n_boot_failed": int(n_fail),
            "n_boot_convergence_warnings": int(n_conv),
            "PM": nie / te if te else np.nan,
            "n": len(d), "n_exposed": int(d[A].sum()), "n_mediator": int(d[M].sum()), "n_outcome": int(d[Y].sum())}

def build(d, h):
    h2 = h.copy(); h2["IDr"] = h2["ID"].astype(str).str.strip()
    d["IDr"] = d["ID"].astype(str).str.strip().str.zfill(12)
    d["waist"] = d["IDr"].map(dict(zip(h2["IDr"], pd.to_numeric(h["r1mwaist"], errors="coerce"))))
    d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
    d["c_tg"] = (d["newtg"] >= 150).astype(float)
    d["c_hdl"] = (d["newhdl"] < 40).astype(float)
    d["c_bp"] = d["hibp"].astype(float)
    d["c_glu"] = d["diab"].astype(float)
    d["mets_n"] = d[["c_waist", "c_tg", "c_hdl", "c_bp", "c_glu"]].fillna(0).sum(axis=1)
    d["mets"] = (d["mets_n"] >= 3).astype(int)   # 主：标准 ≥3/5
    d["mets2"] = (d["mets_n"] >= 2).astype(int)  # 敏感性 ≥2/5
    early = ((pd.to_numeric(h["r2stroke"], errors="coerce") == 1) | (pd.to_numeric(h["r3stroke"], errors="coerce") == 1)).astype(float)
    h2["stroke_early"] = early
    d = d.merge(h2[["IDr", "stroke_early"]], on="IDr", how="left")
    return d

def main():
    h = pd.read_stata(os.path.join(DATA, r"Regular Waves\H_CHARLS_D_Data\H_CHARLS_D_Data.dta"), convert_categoricals=False)
    d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"), low_memory=False)
    d = build(d, h)
    cov = [c for c in ["age_w1", "female", "edu", "smoke", "drink", "cesd_w1"] if c in d.columns and d[c].notna().sum() > 1000]  # BMI 移除：与腰围(MetS组分)共线，over-adjustment

    dd = d.dropna(subset=["mets", "stroke_early", "dep"] + cov).copy()
    print(f"分析样本 n={len(dd)} | 标准MetS(≥3/5)={dd['mets'].mean()*100:.1f}% (n={int(dd['mets'].sum())}) | 早期卒中={int(dd['stroke_early'].sum())} | 抑郁={int(dd['dep'].sum())}")

    # 主分析（标准 ≥3/5）
    r = mediation_full(dd, "mets", "stroke_early", "dep", cov, 800, seed=42)
    print("\n=== 主分析（标准 MetS ≥3/5, seed=42, 800 boot）===")
    for k in ["n", "n_exposed", "n_mediator", "n_outcome", "NDE", "NDE_lo", "NDE_hi", "NIE", "NIE_lo", "NIE_hi", "TE", "TE_lo", "TE_hi", "TE_p", "PM"]:
        print(f"  {k}: {r[k]}")

    # 敏感性 ≥2/5
    dd2 = d.dropna(subset=["mets2", "stroke_early", "dep"] + cov).copy()
    r2 = mediation_full(dd2, "mets2", "stroke_early", "dep", cov, 800, seed=42)
    print("\n=== 敏感性（≥2/5, seed=42）===")
    print(f"  MetS(≥2/5)={dd2['mets2'].mean()*100:.1f}% | NIE={r2['NIE']:.5f} [{r2['NIE_lo']:.5f},{r2['NIE_hi']:.5f}] | TE_p={r2['TE_p']:.3f} | PM={r2['PM']:.3f}")

    # 种子敏感性（主分析）
    print("\n=== 种子敏感性（主分析, 500 boot/种子）===")
    seed_rows = []
    for seed in [42, 0, 1, 2, 3, 7]:
        rs = mediation_full(dd, "mets", "stroke_early", "dep", cov, 500, seed=seed)
        seed_rows.append({"seed": seed, "NIE": rs["NIE"], "NIE_lo": rs["NIE_lo"], "NIE_hi": rs["NIE_hi"]})
        print(f"  seed={seed:>3}: NIE={rs['NIE']:.6f} [{rs['NIE_lo']:+.6f},{rs['NIE_hi']:+.6f}]")

    # 亚组
    print("\n=== 亚组（标准 MetS）===")
    rows = []
    strata = {"sex": {"male": d["female"] == 0, "female": d["female"] == 1}}
    med = d["age_w1"].median()
    strata["age"] = {f"<{int(med)}": d["age_w1"] < med, f">={int(med)}": d["age_w1"] >= med}
    if "rural" in d.columns: strata["residence"] = {"urban": d["rural"] == 0, "rural": d["rural"] == 1}
    for grp, gs in strata.items():
        for lvl, mask in gs.items():
            sub = d[mask].dropna(subset=["mets", "stroke_early", "dep"])
            cv = [c for c in cov if c in sub.columns and sub[c].nunique() > 1 and not (grp == "sex" and c == "female")]
            try:
                rr = mediation_full(sub, "mets", "stroke_early", "dep", cv, 300)
                rows.append({"stratum": grp, "level": lvl, "n": rr["n"], "n_med": rr["n_mediator"],
                             "NIE": round(rr["NIE"], 5), "NIE_lo": round(rr["NIE_lo"], 5), "NIE_hi": round(rr["NIE_hi"], 5),
                             "PM": round(rr["PM"], 3), "n_boot": 300})
                print(f"  {grp}/{lvl}: n={rr['n']} NIE={rr['NIE']:.5f}[{rr['NIE_lo']:.5f},{rr['NIE_hi']:.5f}]")
            except Exception as e: print(f"  {grp}/{lvl} 失败 {str(e)[:40]}")

    # 阴性对照
    print("\n=== 阴性对照（标准 MetS）===")
    neg = {"arthritis": "r4arthre", "asthma": "r4asthmae", "lung": "r4lunge"}
    h2 = h[["ID"]].copy()
    for nm, col in neg.items():
        if col in h.columns: h2[col] = pd.to_numeric(h[col], errors="coerce")
    h2["IDr"] = h2["ID"].astype(str).str.strip()
    d2 = d.merge(h2[["IDr"] + [c for c in neg.values() if c in h.columns]], on="IDr", how="left")
    neg_rows = []
    for nm, col in neg.items():
        if col not in d2.columns: continue
        d2[col] = (d2[col] == 1).astype(float)
        ddn = d2.dropna(subset=["mets", "stroke_early", col] + cov).copy()
        rn = mediation_full(ddn, "mets", "stroke_early", col, cov, 300)
        neg_rows.append({"outcome": nm, "NIE": rn["NIE"], "NIE_lo": rn["NIE_lo"], "NIE_hi": rn["NIE_hi"],
                         "n": rn["n"], "n_boot": 300})
        print(f"  {nm}: n={rn['n']} NIE={rn['NIE']:.5f}[{rn['NIE_lo']:.5f},{rn['NIE_hi']:.5f}]")

    json.dump({"main": r, "sens2": r2, "seed_sensitivity": seed_rows,
               "subgroup": rows, "negcontrol": neg_rows},
              open(os.path.join(OUT, "mets_v2_standard.json"), "w"), indent=2, default=float)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mets_v2_subgroup.csv"), index=False)
    print("\n✓ 已写 mets_v2_standard.json + mets_v2_subgroup.csv")

if __name__ == "__main__":
    main()
