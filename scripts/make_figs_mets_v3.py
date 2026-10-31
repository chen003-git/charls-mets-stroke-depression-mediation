#!/usr/bin/env python3
"""论文19 图 v4（阶段4 草案，在 tmp 中验证）：
  · Fig1 DAG / Fig2 中介分解 / Fig3 时序对比
  · **PNG 由流水线原生渲染到 >= TARGET_PX 宽**（不是事后插值放大）—— 修 C1
  · **不再在图内印标题**（JAD p11: caption 的 brief title 不得显示在图上）—— 修 M10
  · **Fig3 的倍数从产物读取**，不硬编码 —— 修 M8
  · PDF 为主交付（纯矢量），PNG 为后备
"""
import os
import glob, json
import numpy as np, matplotlib

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "_output"); FIG = os.path.join(HERE, "..", "figures")
TARGET_PX = 2400          # JAD 整页 @300dpi 的下限（2244）之上留余量；改这一个常量即可整体抬升
BASE_DPI = 300

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": BASE_DPI,
                     "savefig.bbox": "tight"})
OK = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


def save_fig(fig, base):
    """矢量 PDF（主交付）+ 原生 >= TARGET_PX 宽的 PNG（后备）。

    宽度不足时**重新渲染**（提高 dpi），不是把已有位图插值放大。
    bbox_inches='tight' 下宽度与 dpi 近似成正比，但取整会差 1–2 px，
    故按实测宽度迭代（带 0.5% 余量），最多 6 轮。
    """
    from PIL import Image
    pdf = base + ".pdf"; png = base + ".png"
    fig.savefig(pdf)
    dpi = BASE_DPI
    for _ in range(6):
        fig.savefig(png, dpi=dpi)
        with Image.open(png) as im:
            w = im.size[0]
        if w >= TARGET_PX:
            break
        dpi = dpi * (TARGET_PX * 1.005) / w
    else:
        raise SystemExit(f"出图失败：{os.path.basename(png)} 6 轮迭代后仍未达 {TARGET_PX} px")
    return png, pdf


d = json.load(open(os.path.join(OUT, "mets_v2_standard.json")))
se = d["main"]
any_d = json.load(open(os.path.join(OUT, "mets_any_stroke.json")))
any_nie = any_d["NIE"]; any_lo = any_d["NIE_lo"]; any_hi = any_d["NIE_hi"]; any_pm = any_d["PM"]
pm = json.load(open(os.path.join(OUT, "path_model.json")))
# 倍数从产物读取，不再硬编码（M8）
it = json.load(open(os.path.join(OUT, "intermediate_temporal.json")))
FOLD = it.get("fold_increase_vs_strict")
if FOLD is None:      # 产物缺失时退回两条链相除，仍不写字面量
    FOLD = any_nie / se["NIE"]

# ---------------- Fig1 DAG（无图内标题） ----------------
fig, ax = plt.subplots(figsize=(7, 3.2)); ax.axis("off")
bx = {"A": (0.1, "Metabolic\nsyndrome"), "M": (0.5, "Early incident\nstroke (2013/15)"), "Y": (0.88, "Depression\n(2018)")}
for k, (x, t) in bx.items():
    ax.add_patch(FancyBboxPatch((x - 0.10, 0.40), 0.20, 0.20, boxstyle="round,pad=0.02",
                 fc="#FDEDEC" if k == "M" else "#EAF2F8", ec="#D55E00" if k == "M" else "#0072B2", lw=1.5))
    ax.text(x, 0.50, t, ha="center", va="center", fontsize=8.5)
ax.add_patch(FancyArrowPatch((0.20, 0.50), (0.40, 0.50), arrowstyle="-|>", mutation_scale=18, color=OK[3], lw=2))
ax.add_patch(FancyArrowPatch((0.60, 0.50), (0.78, 0.50), arrowstyle="-|>", mutation_scale=18, color=OK[3], lw=2))
ax.add_patch(FancyArrowPatch((0.13, 0.61), (0.85, 0.61), arrowstyle="-|>", mutation_scale=18, color="#999", lw=1.5, connectionstyle="arc3,rad=-0.35"))
ax.text(0.30, 0.66, f"a: OR {pm['path_a_mets_to_stroke']['OR']:.2f} (P={pm['path_a_mets_to_stroke']['P']:.3f})", fontsize=8.5, fontweight="bold", color=OK[3])
ax.text(0.66, 0.66, f"b: OR {pm['path_b_stroke_to_dep']['OR']:.2f} (P={pm['path_b_stroke_to_dep']['P']:.3f})", fontsize=8.5, fontweight="bold", color=OK[3])
ax.text(0.5, 0.90, "c' direct (NDE): NS", fontsize=8.5, color="#888", ha="center")
ax.text(0.5, 0.28, f"indirect NIE = {se['NIE']:.4f} [{se['NIE_lo']:.4f}, {se['NIE_hi']:.4f}]", fontsize=8.5, color="#444", ha="center")
# 原为 plt.title("Metabolic syndrome -> early stroke -> depression - suppression ...")：已删除（M10）
save_fig(fig, os.path.join(FIG, "Fig1_DAG_mets")); plt.close()
print(f"OK Fig1 (OR {pm['path_a_mets_to_stroke']['OR']:.3f}/{pm['path_b_stroke_to_dep']['OR']:.3f}，从 path_model.json 读，NIE {se['NIE']:.4f})")

# ---------------- Fig2 中介分解 ----------------
fig, ax = plt.subplots(figsize=(5.5, 2.6)); y = [2, 1, 0]
vals = [se["NDE"], se["NIE"], se["TE"]]
los = [se["NDE_lo"], se["NIE_lo"], se["TE_lo"]]
his = [se["NDE_hi"], se["NIE_hi"], se["TE_hi"]]
for i, (v, lo, hi) in enumerate(zip(vals, los, his)):
    ax.errorbar(v, y[i], xerr=[[max(v - lo, 0)], [max(hi - v, 0)]], fmt="o", color=OK[i], capsize=4, markersize=8)
    ax.text(hi, y[i], f" {v:.4f}", va="center", fontsize=8.5, color=OK[i], fontweight="bold")
ax.axvline(0, color="gray", ls="--", alpha=0.5); ax.set_yticks(y)
ax.set_yticklabels(["NDE (direct)", "NIE (via early stroke)", "TE (total)"])
ax.set_xlim(-0.030, 0.030)
ax.set_xlabel("Effect (95% CI)")
# 原 plt.title("Effect decomposition (temporally ordered)")：已删除（M10）
save_fig(fig, os.path.join(FIG, "Fig2_mediation_mets")); plt.close()
print("OK Fig2 (NDE -0.0003 / NIE 0.0012 / TE 0.0008)")

# ---------------- Fig3 时序对比 ----------------
fig, ax = plt.subplots(figsize=(6, 2.8))
labels = ["Early stroke\n(2013/15, primary)", "Any stroke\n(incl. 2018, unrestricted)"]
nies = [se["NIE"], any_nie]; los = [se["NIE_lo"], any_lo]; his = [se["NIE_hi"], any_hi]
xx = [0, 1]
for i in range(2):
    ax.errorbar(xx[i], nies[i], yerr=[[max(nies[i] - los[i], 0)], [max(his[i] - nies[i], 0)]], fmt="o",
                color=OK[3] if i == 0 else OK[4], capsize=5, markersize=10)
    ax.text(xx[i], nies[i] + 0.0008, f"NIE={nies[i]:.4f}", ha="center", fontsize=9, fontweight="bold",
            color=OK[3] if i == 0 else OK[4])
ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Indirect effect (NIE)"); ax.axhline(0, color="gray", ls="--", alpha=0.4)
# 原 plt.title("... ~5.5-fold") 硬编码：已删除；倍数改由产物读取，并放进图注（M8/M10）
save_fig(fig, os.path.join(FIG, "Fig3_timing_mets")); plt.close()
print(f"OK Fig3 (early NIE {se['NIE']:.4f} vs any NIE {any_nie:.4f}; fold={FOLD} 由 intermediate_temporal.json 读)")


# ---------------------------------------------------------------------------
# Elsevier/JAD 图件规范：PNG 不得含透明通道（matplotlib 默认 RGBA）。
# ---------------------------------------------------------------------------
def _flatten_pngs():
    """压平 alpha + 把 dpi 元数据统一成 300。

    px 宽度由渲染 dpi 决定（>= TARGET_PX）；这里只改**元数据**，
    使 2400 px 对应 8 in = 20.3 cm（JAD 整页宽 @300 dpi），并按 GfA 去掉 alpha。
    """
    try:
        from PIL import Image
    except ImportError:
        return
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    for f in sorted(glob.glob(os.path.join(figdir, "*_mets.png"))):
        with Image.open(f) as im:
            if im.mode == "RGB":
                im.save(f, "PNG", dpi=(300, 300))
                continue
            bg = Image.new("RGB", im.size, (255, 255, 255))
            rgba = im.convert("RGBA")
            bg.paste(rgba, mask=rgba.split()[-1])
            bg.save(f, "PNG", dpi=(300, 300))
        print("  RGB 压平 + dpi=300:", os.path.basename(f))


def _verify():
    """出图后自检：宽度达标 + 无 alpha。不达标就以非零码退出，让 run_all 拦住。"""
    from PIL import Image
    figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    bad = []
    for f in sorted(glob.glob(os.path.join(figdir, "*_mets.png"))):
        with Image.open(f) as im:
            w = im.size[0]
            if w < TARGET_PX:
                bad.append((os.path.basename(f), f"宽度 {w} px < {TARGET_PX}"))
            if im.mode != "RGB":
                bad.append((os.path.basename(f), f"模式 {im.mode}（应为 RGB）"))
    for f, why in bad:
        print(f"  X 图件自检未过：{f}: {why}")
    return not bad


_flatten_pngs()
if not _verify():
    raise SystemExit(1)
print("图 -> figures/（PNG 原生 >= %d px 且 RGB；PDF 为矢量主交付）" % TARGET_PX)
