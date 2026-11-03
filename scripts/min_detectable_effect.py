# -*- coding: utf-8 -*-
"""论文19 · 中介分析最小可检测效应（MDE）与功效
两条路：(A) delta 法解析解  (B) 无协变量蒙特卡洛复核
估计量与正文一致：NIE = mean_i[(pY(1,1)-pY(1,0)) * (pM(1)-pM(0))]
"""
import numpy as np, json
from scipy import stats, optimize

# ---- 观测参数（来自 path_model.json / table1.json）----
N           = 11314
OR_A        = 1.9134;  LO_A=1.2936; HI_A=2.8303    # MetS -> early stroke
OR_B        = 1.5682;  LO_B=1.1252; HI_B=2.1857    # early stroke -> depression
P_METS      = 0.1143
P_STROKE    = 172/11314
P_STROKE_1  = 0.0247   # Table 1
P_STROKE_0  = 0.0140
P_DEP       = 0.3869
NIE_OBS     = 0.0012
Z975        = stats.norm.ppf(0.975); Z80 = stats.norm.ppf(0.80)

il = lambda z: 1/(1+np.exp(-z))

# ---- (A) delta 法 ----
se_la = (np.log(HI_A)-np.log(LO_A))/(2*Z975)     # SE(log OR_a)
se_lb = (np.log(HI_B)-np.log(LO_B))/(2*Z975)     # SE(log OR_b)
la, lb = np.log(OR_A), np.log(OR_B)
# NIE ≈ k * la * lb  →  k 由观测 NIE 反解
k = NIE_OBS/(la*lb)
se_nie = k*np.sqrt(lb**2*se_la**2 + la**2*se_lb**2)
mde80  = (Z975+Z80)*se_nie
power_obs = stats.norm.cdf(abs(NIE_OBS)/se_nie - Z975) + stats.norm.cdf(-abs(NIE_OBS)/se_nie - Z975)
A = dict(se_logOR_a=se_la, se_logOR_b=se_lb, k=k, se_NIE=se_nie,
         MDE80=mde80, power_at_observed=power_obs,
         MDE80_relative_to_obs=mde80/NIE_OBS)
print("=== (A) delta 法 ===")
for kk,v in A.items(): print(f"   {kk:26s} = {v:.6g}")

# ---- (B) 无协变量蒙特卡洛 ----
def solve_p(baseline_p, OR, prev):
    """给定 OR 与总体发生率 prev，解对照组的 p0 再算 p1"""
    f = lambda p0: (1-prev_m)*p0 + prev_m*(OR*p0/(1-p0+OR*p0)) - prev
    return None
# 分别解：卒中率由 MetS 分布与 OR_A 决定
def solve_two_exp(x1, OR, prev):
    """P(M=1|X=1) 与 P(M=1|X=0) 满足 odds 比 OR，且边缘率 prev"""
    def g(p0):
        p1 = OR*p0/(1-p0+OR*p0)
        return (1-x1)*p0 + x1*p1 - prev, p1
    p0 = optimize.brentq(lambda p0: g(p0)[0], 1e-8, 0.5)
    return p0, g(p0)[1]
pm0, pm1 = solve_two_exp(P_METS, OR_A, P_STROKE)
# 结局：P(Y=1|X,M)，用无交互 logistic：logit = b0 + lb*M + bc*X
def outcome_fit(b0, bc, pm0, pm1):
    def marg(pm):
        p = il(b0 + bc)  # X=1
        q = il(b0)       # X=0
        return pm*il(b0+lb+bc) + (1-pm)*il(b0+bc), pm*il(b0+lb) + (1-pm)*il(b0)
    m1 = marg(pm1)[0]; m0 = marg(pm0)[1]
    return P_METS*m1 + (1-P_METS)*m0
b0sol = optimize.brentq(lambda b0: outcome_fit(b0,0.0,pm0,pm1)-P_DEP, -3, 3)
def nie_from(b0, bc):
    def pY(x,m): return il(b0 + lb*m + bc*x)
    return ((pm1-pm0)*0 + ((pm1*0)))  # placeholder
def nie_exact(b0, bc, pm1v, pm0v):
    y11,y10 = il(b0+lb+bc), il(b0+bc)
    ey1m1 = pm1v*y11 + (1-pm1v)*y10
    ey1m0 = pm0v*y11 + (1-pm0v)*y10
    return ey1m1-ey1m0
print("\n=== (B) 无协变量结构模型（真值标定）===")
print(f"   P(卒中|MetS=1) = {pm1:.4f}  (Table 1: 0.0247)")
print(f"   P(卒中|MetS=0) = {pm0:.4f}  (Table 1: 0.0140)")
print(f"   NIE(真值, b0={b0sol:.4f}, c'=0) = {nie_exact(b0sol,0.0,pm1,pm0):.6f}")

rng = np.random.default_rng(20261102)
B = 400; NIE=[]
for _ in range(B):
    x = (rng.random(N) < P_METS).astype(float)
    m = (rng.random(N) < np.where(x==1, pm1, pm0)).astype(float)
    y = (rng.random(N) < il(b0sol + lb*m)).astype(float)   # c'=0
    # MLE 拟合（无协变量）
    # logit M ~ x
    Xa = np.column_stack([np.ones(N), x]); Xb = np.column_stack([np.ones(N), x, m])
    def fit(X,yy):
        b=np.zeros(X.shape[1]); b[0]=np.log(yy.mean()/(1-yy.mean()))
        for _ in range(30):
            p=il(np.clip(X@b,-30,30)); w=np.clip(p*(1-p),1e-10,None)
            try: nb=np.linalg.solve((X.T*w)@X,(X.T*w)@(X@b+(yy-p)/w))
            except np.linalg.LinAlgError: break
            if np.max(np.abs(nb-b))<1e-10: b=nb; break
            b=nb
        return b
    ba=fit(Xa,m); bb=fit(Xb,y)
    p1=il(ba[0]+ba[1]); p0=il(ba[0])
    y11=il(bb[0]+bb[1]+bb[2]); y10=il(bb[0]+bb[1])
    NIE.append((p1*y11+(1-p1)*y10)-(p0*y11+(1-p0)*y10))
NIE=np.array(NIE)
print(f"   MC: NIE 均值 {NIE.mean():.6f} | SD {NIE.std(ddof=1):.6f} | 2.5-97.5% {np.percentile(NIE,2.5):.6f} ~ {np.percentile(NIE,97.5):.6f}")
print(f"   观测 bootstrap SE ≈ 0.00066（正文 CI 0.0002–0.0028 反推）")

out = dict(observed=dict(n=N, mediator_events=172, OR_a=OR_A, OR_b=OR_B, NIE=NIE_OBS,
                         bootstrap_SE=0.00066, bootstrap_CI=[0.0002,0.0028]),
           delta_method=A, mc_no_covariate=dict(mean=float(NIE.mean()), sd=float(NIE.std(ddof=1)),
                         ci95=[float(np.percentile(NIE,2.5)), float(np.percentile(NIE,97.5))], B=B),
           calibration=dict(P_stroke_given_MetS=float(pm1), P_stroke_given_nonMetS=float(pm0)))
json.dump(out, open("power_mde.json","w"), indent=1, ensure_ascii=False)
print("\n已写 power_mde.json")
