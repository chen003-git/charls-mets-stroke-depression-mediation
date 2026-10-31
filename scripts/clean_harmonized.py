#!/usr/bin/env python3
"""
论文19：CHARLS Harmonized 数据清洗（干净版）
曲线：Wave1(2011) 血脂异常 → Wave2-4(2013/2015/2018) 新发卒中 → Wave4(2018) 抑郁
数据：Regular Waves/H_CHARLS_D_Data/H_CHARLS_D_Data.dta
输出：data/analysis.csv
"""
import os
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

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HARM = os.path.join(DATA, "Regular Waves", "H_CHARLS_D_Data", "H_CHARLS_D_Data.dta")

def main():
    print("="*58, "\n论文19 · Harmonized 清洗\n", "="*58)
    h = pd.read_stata(HARM, convert_categoricals=False)
    print(f"Harmonized: {h.shape}")

    d = pd.DataFrame()
    d["ID"] = h["ID"].astype(str) if "ID" in h.columns else h.iloc[:,0].astype(str)

    # ── 协变量（基线）──
    for src, nm in [("ragender","female"),("r1agey","age_w1"),("raeducl","edu"),
                    ("r1mbmi","bmi"),("r1hibpe","hibp"),("r1diabe","diab"),
                    ("r1smokev","smoke"),("r1drinkev","drink"),("r1rural2","rural")]:
        if src in h.columns: d[nm] = pd.to_numeric(h[src], errors="coerce")
    if "female" in d: d["female"] = (d["female"] == 2).astype(int)  # 1=male,2=female

    # ── 暴露：基线血脂异常（自我报告）──
    d["dyslip"] = (pd.to_numeric(h["r1dyslipe"], errors="coerce") == 1).astype(int) if "r1dyslipe" in h.columns else np.nan

    # ── 基线卒中（排除用）──
    d["stroke_w1"] = (pd.to_numeric(h["r1stroke"], errors="coerce") == 1).astype(int) if "r1stroke" in h.columns else 0

    # ── 基线抑郁（协变量）──
    d["cesd_w1"] = pd.to_numeric(h["r1cesd10"], errors="coerce") if "r1cesd10" in h.columns else np.nan

    # ── 中介：随访新发卒中（2/3/4 波任一）──
    st_cols = [c for c in ["r2stroke","r3stroke","r4stroke"] if c in h.columns]
    d["stroke_fu"] = (h[st_cols].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1).astype(int) if st_cols else 0

    # ── 结局：随访抑郁（Wave4 CES-D≥10）──
    for w in ["r4cesd10","r3cesd10","r2cesd10"]:
        if w in h.columns:
            d["cesd_end"] = pd.to_numeric(h[w], errors="coerce"); d["cesd_wave"] = w; break
    # 结局：随访抑郁（Wave4 CES-D≥10）；保留缺失（NaN），不静默记为"非抑郁"
    d["dep"] = (d["cesd_end"] >= 10).astype("float").where(d["cesd_end"].notna())

    # ── 合并 2011 血检血脂（原始数据，更客观）──
    bp = os.path.join(DATA, r"2011全国\Blood_20140429\Blood_20140429.dta")
    if os.path.exists(bp):
        b = pd.read_stata(bp, convert_categoricals=False)
        b["IDr"] = b["ID"].astype(str).str.strip()
        def to_raw(x):
            x=str(x).strip()
            return x[:10]+str(int(x[10:])) if len(x)==12 else x
        d["IDr"] = d["ID"].map(to_raw)
        for c in ["newcho","newtg","newhdl","newldl"]:
            if c in b.columns: d[c] = d["IDr"].map(dict(zip(b["IDr"], pd.to_numeric(b[c], errors="coerce"))))
        print(f"血检合并: TC 有效 {d['newcho'].notna().sum() if 'newcho' in d else 0}")
        # 血检定义血脂异常（TC>5.2 或 TG>1.7 或 LDL>3.4 或 HDL<1.0 mmol/L）
        if "newcho" in d:
            import numpy as _np
            # 三项血脂全缺失 → dyslip_blood = NaN（不误判为正常）；否则按切点
            avail = d[["newcho","newtg","newldl","newhdl"]].notna().any(axis=1)
            defn = ((d["newcho"]>200)|(d["newtg"]>150)|(d["newldl"]>130)|(d["newhdl"]<40))  # unit = mg/dL
            d["dyslip_blood"] = _np.where(avail, defn.astype(float), _np.nan)

    # ── 排除基线卒中 ──
    n0 = len(d); d = d[d["stroke_w1"] != 1].copy()
    print(f"排除基线卒中: {n0} → {len(d)}")

    outp = os.path.join(DATA, "analysis_harmonized.csv")
    d.to_csv(outp, index=False)
    print(f"\n✓ {outp}: {d.shape}")
    print(f"  新发卒中 {d['stroke_fu'].sum()} | 抑郁(≥10) {d['dep'].sum()} | 血脂异常 {d['dyslip'].sum()}")
    print(f"  CES-D 均值(W1) {d['cesd_w1'].mean():.2f} | 结局波次 {d.get('cesd_wave','?')}")
    print("  列:", list(d.columns))

if __name__ == "__main__":
    main()
