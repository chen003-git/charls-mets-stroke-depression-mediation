#!/usr/bin/env python3
"""论文19 §3.1 后半 + §3.4 全部新增分析的统一产物生成器
输出（全部到 _output/）：
  complete_phenotype.json   五组分全实测子集 + 标准定义（实测血压/血糖、性别特异 HDL）
  missing_conventions.json  四种缺失约定
  reverse_causation.json    基线抑郁 -> 早期卒中
  evalue_interaction.json   E-value（RR 近似）+ 暴露-中介交互
  intermediate_temporal.json 三个中介口径
不依赖 statsmodels（numpy 自实现 IRLS）
"""
import os, json, math
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
BLOOD = os.path.join(DATA, "2011全国", "Blood_20140429", "Blood_20140429.dta")
os.makedirs(OUT, exist_ok=True)

def num(s): return pd.to_numeric(s, errors="coerce")
def to_raw(x):
    x = str(x).strip()
    return int(x[:10] + str(int(x[10:]))) if len(x) == 12 else None

def logit(X, y, maxit=100, tol=1e-10):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = np.clip(X@b, -30, 30); p = 1/(1+np.exp(-eta))
        W = np.maximum(p*(1-p), 1e-10); z = eta + (y-p)/W
        try: bn = np.linalg.solve((X.T*W)@X, (X.T*W)@z)
        except np.linalg.LinAlgError: return None, None
        if np.max(np.abs(bn-b)) < tol: break
        b = bn
    p = 1/(1+np.exp(-np.clip(X@b, -30, 30))); W = np.maximum(p*(1-p), 1e-12)
    se = np.sqrt(np.diag(np.linalg.pinv((X.T*W)@X)))
    return b, se

def p2(z): return 2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))

def fit(df, y, xs):
    s = df.dropna(subset=[y]+xs)
    if len(s) < 200: return None
    X = np.column_stack([np.ones(len(s))] + [s[c].values.astype(float) for c in xs])
    b, se = logit(X, s[y].values)
    if b is None: return None
    z = b[1]/se[1]
    return dict(n=int(len(s)), OR=float(np.exp(b[1])), lo=float(np.exp(b[1]-1.96*se[1])),
                hi=float(np.exp(b[1]+1.96*se[1])), P=float(p2(z)))

def mediation(df, cov, A="mets", M="stroke_early", Y="dep"):
    s = df.dropna(subset=[A, M, Y] + cov); n = len(s)
    if n < 200: return None
    Xy = np.column_stack([np.ones(n), s[A]] + [s[c] for c in cov] + [s[M]])
    Xm = np.column_stack([np.ones(n), s[A]] + [s[c] for c in cov])
    by, _ = logit(Xy, s[Y].values); bm, _ = logit(Xm, s[M].values)
    if by is None or bm is None: return None
    cc = np.column_stack([s[c] for c in cov])
    def pr(b, av, mv): return 1/(1+np.exp(-np.clip(np.column_stack([np.ones(n), np.full(n,av), cc, np.full(n,mv)])@b, -30, 30)))
    def pm(b, av): return 1/(1+np.exp(-np.clip(np.column_stack([np.ones(n), np.full(n,av), cc])@b, -30, 30)))
    m1, m0 = pm(bm,1.0), pm(bm,0.0)
    y11, y10, y00 = pr(by,1.0,m1), pr(by,1.0,m0), pr(by,0.0,m0)
    return dict(n=int(n), n_mets=int(s[A].sum()), mets_prev_sample=float(s[A].mean()),
                NDE=float(np.mean(y10-y00)), NIE=float(np.mean(y11-y10)), TE=float(np.mean(y11-y00)))

# ================= 构建数据 =================
print("读取数据 ...")
d = pd.read_csv(os.path.join(DATA, "analysis_harmonized.csv"))
h = pd.read_stata(HARM, convert_categoricals=False)
b = pd.read_stata(BLOOD, convert_categoricals=False)

hh = pd.DataFrame({
    "IDr": h["ID"].map(to_raw),
    "waist": num(h["r1mwaist"]),
    "sbp1": num(h["r1systo1"]), "sbp2": num(h["r1systo2"]),
    "dbp1": num(h["r1diasto1"]) if "r1diasto1" in h.columns else np.nan,
    "dbp2": num(h["r1diasto2"]) if "r1diasto2" in h.columns else np.nan,
    "r2": num(h["r2stroke"]), "r3": num(h["r3stroke"]), "r4": num(h["r4stroke"]),
    "cesd2": num(h["r2cesd10"]) if "r2cesd10" in h.columns else np.nan,
    "cesd3": num(h["r3cesd10"]) if "r3cesd10" in h.columns else np.nan,
}).dropna(subset=["IDr"])
d = d.merge(hh, on="IDr", how="left")
b2 = pd.DataFrame({"IDr": b["ID"].astype(str).str.strip().str.zfill(11), "glu": num(b["newglu"])})
d["_k"] = d["IDr"].astype(str).str.zfill(11)
d = d.merge(b2.rename(columns={"IDr": "_k"}), on="_k", how="left")

# 中介口径
# 主口径（与 run_mets_v2.py 一致）：中介缺失记为阴性
d["stroke_early"] = (d["r2"].fillna(0) == 1) | (d["r3"].fillna(0) == 1)
d["stroke_early"] = d["stroke_early"].astype(float)
# 敏感性另列：中介缺失保留 NaN
d["stroke_early_nan"] = np.where(d["r2"].notna() | d["r3"].notna(),
                                  ((d["r2"] == 1) | (d["r3"] == 1)).astype(float), np.nan)
d["stroke_any"] = np.where(d[["r2","r3","r4"]].notna().any(axis=1),
                           (d[["r2","r3","r4"]] == 1).any(axis=1).astype(float), np.nan)
d["stroke_2015"] = np.where(d["r3"].notna(), (d["r3"] == 1).astype(float), np.nan)
d["stroke_2018"] = np.where(d["r4"].notna(), (d["r4"] == 1).astype(float), np.nan)

# MetS 组分：现行规格
d["c_waist"] = np.where(d["female"] == 1, d["waist"] >= 85, d["waist"] >= 90).astype(float)
d.loc[d["waist"].isna(), "c_waist"] = np.nan
d["c_tg"]  = (d["newtg"] >= 150).astype(float);  d.loc[d["newtg"].isna(),  "c_tg"]  = np.nan
d["c_hdl"] = (d["newhdl"] < 40).astype(float);   d.loc[d["newhdl"].isna(), "c_hdl"] = np.nan
d["c_bp"]  = d["hibp"].astype(float)
d["c_glu"] = d["diab"].astype(float)
CUR = ["c_waist","c_tg","c_hdl","c_bp","c_glu"]

# 标准定义：性别特异 HDL + 实测血压 + 实测血糖
d["s_hdl"] = np.where(d["female"] == 1, d["newhdl"] < 50, d["newhdl"] < 40).astype(float)
d.loc[d["newhdl"].isna(), "s_hdl"] = np.nan
sbp = d[["sbp1","sbp2"]].mean(axis=1, skipna=True); dbp = d[["dbp1","dbp2"]].mean(axis=1, skipna=True)
d["s_bp"] = ((sbp >= 130) | (dbp >= 85)).astype(float); d.loc[sbp.isna() & dbp.isna(), "s_bp"] = np.nan
d["s_glu"] = (d["glu"] >= 100).astype(float); d.loc[d["glu"].isna(), "s_glu"] = np.nan
STD = ["c_waist","c_tg","s_hdl","s_bp","s_glu"]

cov = [c for c in ["age_w1","female","edu","smoke","drink","cesd_w1"] if c in d.columns]
N_TOTAL = len(d)
print(f"全队列 n = {N_TOTAL}")

def set_mets(df, cols, how="nan"):
    cnt = df[cols].sum(axis=1, min_count=1) if how == "nan" else df[cols].fillna(0).sum(axis=1)
    m = (cnt >= 3).astype(float)
    if how == "nan": m.loc[cnt.isna()] = np.nan
    return m

res = {}

# ============ 1. 完整表型 + 标准定义 ============
print("\n[1] complete_phenotype")
cp = {}
d["mets_zero"] = set_mets(d, CUR, "zero")
d["mets_nan"]  = set_mets(d, CUR, "nan")
d["mets_std"]  = set_mets(d, STD, "nan")
main = mediation(d, cov, A="mets_zero")
cp["primary"] = dict(mets_prev=float(d["mets_zero"].mean()), **main)
# 敏感性：中介缺失保留 NaN 并剔除
rnan = mediation(d, cov, A="mets_zero", M="stroke_early_nan")
if rnan: cp["mediator_missing_dropped"] = dict(**rnan)
# 五组分全实测子集（现行规格定义）
sub = d[d[CUR].notna().all(axis=1)].copy()
sub["mets"] = (sub[CUR].sum(axis=1) >= 3).astype(float)
r = mediation(sub, cov); cp["fully_measured_components"] = dict(n_subset=int(len(sub)), mets_prev=float(sub["mets"].mean()), **r)
# 标准定义（全样本）
r2 = mediation(d, cov, A="mets_std"); cp["measured_definition"] = dict(mets_prev=float(d["mets_std"].mean()), **r2)
# 标准定义 + 五组分全实测
sub2 = d[d[STD].notna().all(axis=1)].copy()
sub2["mets"] = (sub2[STD].sum(axis=1) >= 3).astype(float)
r3 = mediation(sub2, cov); cp["measured_definition_fully_measured"] = dict(n_subset=int(len(sub2)), mets_prev=float(sub2["mets"].mean()), **r3)
res["complete_phenotype"] = cp
for k, v in cp.items():
    if v: print(f"  {k:<38} n={v.get('n'):>6} MetS={v.get('mets_prev_sample',0)*100:5.2f}% ({v.get('n_mets')})  NIE={v['NIE']:+.6f}  TE={v['TE']:+.6f}")

# ============ 2. 四种缺失约定 ============
print("\n[2] missing_conventions")
mc = {}
# 真正的"缺失即剔除"：先把组分缺失者整行去掉，再算 MetS
d_drop = d[d[CUR].notna().all(axis=1)].copy()
d_drop["_m"] = (d_drop[CUR].sum(axis=1) >= 3).astype(float)
for lab, how in [("missing_as_negative", "zero"), ("missing_as_positive", "pos")]:
    cnt = d[CUR].fillna(1 if how == "pos" else 0).sum(axis=1)
    _tmp = d.copy(); _tmp["_m"] = (cnt >= 3).astype(float)
    r = mediation(_tmp, cov, A="_m")
    if r:
        mc[lab] = r
        print("  {:<24} n={}  NIE={:+.6f}  TE={:+.6f}".format(lab, r["n"], r["NIE"], r["TE"]))
r = mediation(d_drop, cov, A="_m")
if r:
    # 注意：missing_dropped 与 complete_phenotype.fully_measured_components 是同一个子集，
    # 不再写进 missing_conventions，避免同一分析在同一字典里出现三次。
    mc["missing_dropped"] = r
    print(f"  {'missing_dropped':<24} n={r['n']}  NIE={r['NIE']:+.6f}  TE={r['TE']:+.6f}")
res["missing_conventions"] = mc
print(f"  不同 NIE 取值数: {len(set(round(v['NIE'], 6) for v in mc.values() if isinstance(v, dict) and 'NIE' in v))} / {len(mc)}")
lo = min(v["NIE"] for v in mc.values() if v); hi = max(v["NIE"] for v in mc.values() if v)
print(f"  ⇒ NIE 范围 [{lo:+.6f}, {hi:+.6f}]  共 {len(mc)} 种约定")

# ============ 3. 反向因果 ============
print("\n[3] reverse_causation")
d["cesd_hi"] = (d["cesd_w1"] >= 10).astype(float); d.loc[d["cesd_w1"].isna(), "cesd_hi"] = np.nan
rc = {}
for lab, expo in [("baseline_cesd_binary", "cesd_hi"), ("baseline_cesd_continuous", "cesd_w1")]:
    r = fit(d, "stroke_early", [expo] + ["age_w1","female","edu","smoke","drink","mets_zero"])
    if r: rc[lab] = r; print(f"  {lab:<28} n={r['n']}  OR={r['OR']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]  P={r['P']:.6f}")
# 反向：2018 抑郁 -> 2018 首次报告卒中（A 指出缺失的那条）
r = fit(d, "stroke_2018", ["dep"] + ["age_w1","female","edu","smoke","drink","mets_zero"])
if r: rc["dep2018_to_stroke2018"] = r; print(f"  {'dep2018_to_stroke2018':<28} n={r['n']}  OR={r['OR']:.4f}  P={r['P']:.6f}")
# 排除基线抑郁者的敏感性（A 指出的第 3 项）
d2 = d[d["cesd_w1"] < 10].copy()
r = mediation(d2, cov, A="mets_nan")
if r: rc["excluding_baseline_depressed"] = r; print(f"  {'excluding_baseline_depressed':<28} n={r['n']}  NIE={r['NIE']:+.6f}  TE={r['TE']:+.6f}")
res["reverse_causation"] = rc

# ============ 4. E-value + 交互 ============
print("\n[4] evalue_interaction")
ei = {}
pb = fit(d, "dep", ["stroke_early"] + cov)
if pb:
    orv = pb["OR"]; p0 = float(d["dep"].mean())
    rr = orv/(1 - p0 + p0*orv)                     # VanderWeele & Ding RR 近似
    ev_rr = rr + math.sqrt(rr*(rr-1))
    ev_or = orv + math.sqrt(orv*(orv-1))           # 罕见结局捷径（仅对照）
    ei["path_b"] = dict(OR=round(orv,4), prevalence=round(p0,4), RR_approx=round(rr,4),
                        E_value_RR=round(ev_rr,3), E_value_rare_outcome=round(ev_or,3))
    print(f"  中介->结局 OR={orv:.4f}  结局流行率={p0:.4f}")
    print(f"  RR 近似={rr:.4f}  ⇒  E-value={ev_rr:.3f}  （稿件用的罕见结局捷径={ev_or:.3f}）")
te = fit(d, "dep", ["mets_zero"] + cov)
if te:
    orv = te["OR"]; p0 = float(d["dep"].mean()); rr = orv/(1-p0+p0*orv)
    ei["total_effect"] = dict(OR=round(orv,4), E_value_RR=round(rr+math.sqrt(rr*(rr-1)),3) if rr>1 else 1.0)
    print(f"  总效应 OR={orv:.4f}  E-value={ei['total_effect']['E_value_RR']:.3f}")
s = d.dropna(subset=["dep","mets_nan","stroke_early"]+cov).copy()
s["inter"] = s["mets_nan"]*s["stroke_early"]
ri = fit(s, "dep", ["mets_nan","stroke_early","inter"]+cov)
if ri:
    ei["interaction"] = ri
    ei["double_exposed_n"] = int(((s["mets_nan"]==1)&(s["stroke_early"]==1)).sum())
    print(f"  交互项 OR={ri['OR']:.4f} [{ri['lo']:.4f}, {ri['hi']:.4f}]  P={ri['P']:.4f}  双层格 n={ei['double_exposed_n']}")
res["evalue_interaction"] = ei

# ============ 5. 时序三口径 ============
print("\n[5] intermediate_temporal")
it = {}
for lab, mcol in [("strict_2013_or_2015", "stroke_early"), ("only_2015", "stroke_2015"),
                  ("unrestricted_incl_2018", "stroke_any")]:
    r = mediation(d, cov, A="mets_nan", M=mcol)
    if r: it[lab] = r; print(f"  {lab:<26} n={r['n']}  NIE={r['NIE']:+.6f}")
a = it.get("strict_2013_or_2015"); b_ = it.get("only_2015"); c_ = it.get("unrestricted_incl_2018")
if a and c_:
    it["fold_increase_vs_strict"] = round(c_["NIE"]/a["NIE"], 2)
    print(f"  ⇒ 无限制 / 严格 = {c_['NIE']/a['NIE']:.2f} 倍")
res["intermediate_temporal"] = it

# ================= 落盘 =================
for k, v in res.items():
    json.dump(v, open(os.path.join(OUT, f"{k}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
print(f"\n已写 {len(res)} 个产物到 {OUT}")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
