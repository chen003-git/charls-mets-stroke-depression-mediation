#!/usr/bin/env python3
"""论文19 一键复现入口。

顺序执行本论文的全部分析；任一环节失败即以非零码退出（便于 CI 与外部复现者判断）。

依赖见 scripts/requirements.txt：
    pandas numpy statsmodels scipy matplotlib
其中 run_path_model.py / run_seed_v2.py / run_sensitivity_pack.py 仅需 numpy + pandas。
"""
import os
import subprocess
import sys
import time

# 在 GBK 控制台下 print 非 ASCII 会抛 UnicodeEncodeError 而中断整个管线。
# 显式把 stdout/stderr 重配为 UTF-8，并让子进程继承。
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "_output")
FIG = os.path.join(HERE, "..", "figures")

STEPS = [
    ("数据清洗 (Harmonized)",           "clean_harmonized.py"),
    ("样本流转/缺失/缺失预测因子",        "run_flow_missing.py"),
    ("主分析 MetS 中介 (现行规格)",      "run_mets_v2.py"),
    ("时序对照 (any stroke)",           "run_any_stroke.py"),
    ("路径模型 (自产 path_model.json)",  "run_path_model.py"),
    ("种子稳健性 (9 种子 x 500)",        "run_seed_v2.py"),
    ("敏感性分析包 (§3.1后半 + §3.4)",   "run_sensitivity_pack.py"),
    ("审稿补充分析 (§3.2/§3.3)",         "run_review_extras.py"),
    ("整群 bootstrap 敏感性 (§2.5/§3.2)", "cluster_bootstrap.py"),
    ("中介错分偏倚分析 (§2.5/§3.4 + Table S4)", "mediator_misclass_bias.py"),
    ("Table 1",                         "make_table1_v2.py"),
    ("出图 (v4: 原生 >=2400 px + 自检)","make_figs_mets_v3.py"),
    ("补充图 S1 参与者流程 (flow_counts.json -> 矢量)","make_fig_s1.py"),
]

REQUIRED_IMPORTS = {
    "run_mets_v2.py": ["statsmodels"],
    "run_any_stroke.py": ["statsmodels"],
    "run_review_extras.py": ["statsmodels"],
    "make_table1_v2.py": ["scipy"],
    "make_figs_mets_v3.py": ["matplotlib", "PIL"],
}


def missing_deps(script):
    out = []
    for mod in REQUIRED_IMPORTS.get(script, []):
        try:
            __import__(mod)
        except ImportError:
            out.append(mod)
    return out



# 出图步骤的输入产物：这些 JSON 必须在本次运行中被重新生成，
# 否则会静默用陈旧产物出图（R4 的 Figure 1 事故正是由此产生）
FIG_INPUTS = ["path_model.json", "mets_v2_standard.json", "mets_any_stroke.json"]


def assert_fresh(paths, started_at, label):
    stale = []
    for f in paths:
        p = os.path.join(OUT, f)
        if not os.path.exists(p):
            stale.append((f, "missing"))
        elif os.path.getmtime(p) < started_at:
            age = started_at - os.path.getmtime(p)
            stale.append((f, f"{age/60:.1f} min older than this run"))
    if stale:
        print(f"\n  ✗ {label}：输入产物不新鲜，拒绝出图")
        for f, why in stale:
            print(f"      {f}: {why}")
        return False
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    print("=" * 60)
    print("论文19 CHARLS MetS-卒中-抑郁 中介分析 · 一键复现")
    print("=" * 60)

    # 依赖预检：避免跑到一半才发现缺包
    blocked = [(s, m) for _, s in STEPS if (m := missing_deps(s))]
    if blocked:
        print("\n依赖缺失，无法完整复现：")
        for script, miss in blocked:
            print(f"  {script}: 缺少 {', '.join(miss)}")
        print("\n请先安装：pip install -r requirements.txt")
        print("（run_path_model.py / run_seed_v2.py / run_sensitivity_pack.py 无此依赖，可单独运行）")
        sys.exit(2)

    started_at = time.time()
    failed = []
    for label, script in STEPS:
        path = os.path.join(HERE, script)
        print(f"\n▶ {label}  ({script})")
        if script == "make_figs_mets_v3.py":
            if not assert_fresh(FIG_INPUTS, started_at, label):
                failed.append(script)
                continue
        if not os.path.exists(path):
            print(f"  ✗ 脚本不存在：{script}")
            failed.append(script)
            continue
        r = subprocess.run([sys.executable, path], cwd=HERE, env=os.environ.copy())
        if r.returncode != 0:
            print(f"  ✗ {script} 失败（退出码 {r.returncode}）")
            failed.append(script)

    print("\n" + "=" * 60)
    if failed:
        print(f"复现未完成：{len(failed)}/{len(STEPS)} 个步骤失败")
        for s in failed:
            print(f"  - {s}")
        print("=" * 60)
        sys.exit(1)
    print(f"全部 {len(STEPS)} 步完成。结果见 ../_output/ ，图见 ../figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
