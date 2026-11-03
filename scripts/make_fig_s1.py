#!/usr/bin/env python3
"""论文19 Figure S1: participant flow (CONSORT-style).

Rewritten 2026-11-03 to fix defects found in the previous version:
  * arrows floated free of the boxes and pointed at empty space
  * exclusion text sat on the box border
  * a large vertical gap above the analytic-sample box
  * a code-like expression ("r1stroke == 1") appeared in a published figure
  * n_mets (1,293) and n_outcome (4,377) were hard-coded literals

Every count below is read from a deposited output file; nothing is hand-typed.
Consistency assertions refuse to draw the figure if the inputs disagree.

Inputs:
    _output/flow_counts.json        flow counts
    _output/table1.json             n_mets
    _output/mets_v2_standard.json   main.n_outcome

Output:
    Supplementary/Figure_S1.pdf     vector
    Supplementary/Figure_S1.png     400 dpi preview
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "_output")
SUP = os.path.join(HERE, "..", "Supplementary")
os.makedirs(SUP, exist_ok=True)

flow = json.load(open(os.path.join(OUT, "flow_counts.json"), encoding="utf-8"))
t1 = json.load(open(os.path.join(OUT, "table1.json"), encoding="utf-8"))
mets = json.load(open(os.path.join(OUT, "mets_v2_standard.json"), encoding="utf-8"))

N_BASE = flow["raw_respondents"]
N_STROKE_EXCL = flow["excluded_prevalent_stroke"]
N_ELIGIBLE = flow["after_stroke_exclusion"]
N_COV_EXCL = flow["excluded_missing_covariates"]
N_COV_COMPLETE = flow["covariate_complete"]
N_OUT_EXCL = flow["excluded_missing_outcome"]
N_ANALYTIC = flow["analytic_sample"]
N_METS = t1["n_mets"]
N_NON_METS = t1["n_non_mets"]
N_MEDS = flow["n_stroke_early_r2r3"]
N_DEP = mets["main"]["n_outcome"]
N_METS_PCT = t1["mets_pct"]
N_DEP_PCT = round(100.0 * N_DEP / N_ANALYTIC, 1)

# ---- consistency guards ---------------------------------------------------
assert N_BASE - N_STROKE_EXCL == N_ELIGIBLE, "baseline - stroke excluded != eligible"
assert N_ELIGIBLE - N_COV_EXCL == N_COV_COMPLETE, "eligible - covariate exclusions != complete"
assert N_COV_COMPLETE - N_OUT_EXCL == N_ANALYTIC, "complete - outcome exclusions != analytic"
assert N_METS + N_NON_METS == N_ANALYTIC, "MetS + non-MetS != analytic sample"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

fig, ax = plt.subplots(figsize=(6.6, 7.4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ---- geometry -------------------------------------------------------------
MAIN_L, MAIN_R = 0.02, 0.54
MAIN_CX = (MAIN_L + MAIN_R) / 2
EXCL_L, EXCL_R = 0.62, 0.99
EXCL_CX = (EXCL_L + EXCL_R) / 2

C_MAIN, C_MAIN_E = "#DCE9F5", "#4A7EAD"
C_EXCL, C_EXCL_E = "#F2F2F2", "#8C8C8C"
C_FINAL, C_FINAL_E = "#FBE3D0", "#C0713A"
LW_BOX, LW_ARR, ARROW_C = 1.0, 0.9, "#3A3A3A"

H_MAIN, H_EXCL, H_FINAL = 0.082, 0.078, 0.180
Y = [0.915, 0.720, 0.525, 0.330]
Y_FINAL = 0.150
Y_BRANCH = [Y[i] - H_MAIN / 2 - 0.032 for i in range(3)]


def main_box(yc, title, value):
    ax.add_patch(FancyBboxPatch(
        (MAIN_L, yc - H_MAIN / 2), MAIN_R - MAIN_L, H_MAIN,
        boxstyle="round,pad=0.004,rounding_size=0.016",
        linewidth=LW_BOX, edgecolor=C_MAIN_E, facecolor=C_MAIN, zorder=2))
    ax.text(MAIN_CX, yc + H_MAIN * 0.17, title, ha="center", va="center",
            fontsize=9.2, fontweight="bold", zorder=3)
    ax.text(MAIN_CX, yc - H_MAIN * 0.21, value, ha="center", va="center",
            fontsize=9.2, zorder=3)


def excl_box(yc, title, value):
    ax.add_patch(FancyBboxPatch(
        (EXCL_L, yc - H_EXCL / 2), EXCL_R - EXCL_L, H_EXCL,
        boxstyle="round,pad=0.004,rounding_size=0.016",
        linewidth=LW_BOX, edgecolor=C_EXCL_E, facecolor=C_EXCL, zorder=2))
    ax.text(EXCL_CX, yc + H_EXCL * 0.17, title, ha="center", va="center",
            fontsize=8.6, zorder=3)
    ax.text(EXCL_CX, yc - H_EXCL * 0.26, "n = %s" % value, ha="center", va="center",
            fontsize=8.9, fontweight="bold", zorder=3)


def v_arrow(x, y_from, y_to):
    ax.add_patch(FancyArrowPatch(
        (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=11,
        linewidth=LW_ARR, color=ARROW_C, shrinkA=0, shrinkB=0, zorder=1))


def h_arrow(y, x_from, x_to):
    ax.add_patch(FancyArrowPatch(
        (x_from, y), (x_to, y), arrowstyle="-|>", mutation_scale=11,
        linewidth=LW_ARR, color=ARROW_C, shrinkA=0, shrinkB=0, zorder=1))


# ---- main column ----------------------------------------------------------
main_box(Y[0], "CHARLS 2011 baseline", "n = %s" % f"{N_BASE:,}")
main_box(Y[1], "Eligible cohort", "n = %s" % f"{N_ELIGIBLE:,}")
main_box(Y[2], "Covariate-complete", "n = %s" % f"{N_COV_COMPLETE:,}")
main_box(Y[3], "Analytic sample", "n = %s" % f"{N_ANALYTIC:,}")

# composition box
ax.add_patch(FancyBboxPatch(
    (MAIN_L, Y_FINAL - H_FINAL / 2), MAIN_R - MAIN_L, H_FINAL,
    boxstyle="round,pad=0.004,rounding_size=0.016",
    linewidth=LW_BOX, edgecolor=C_FINAL_E, facecolor=C_FINAL, zorder=2))
ax.text(MAIN_CX, Y_FINAL + H_FINAL * 0.355, "Among the analytic sample",
        ha="center", va="center", fontsize=9.2, fontweight="bold", zorder=3)
ax.text(MAIN_CX, Y_FINAL + H_FINAL * 0.215, "subsets; categories overlap",
        ha="center", va="center", fontsize=7.8, style="italic",
        color="#555555", zorder=3)
comp = (
    "Metabolic syndrome  n = %s" % f"{N_METS:,}",
    "Early incident stroke (2013/15)  n = %s" % f"{N_MEDS:,}",
    "Depressive symptoms (2018)  n = %s" % f"{N_DEP:,}",
)
for i, line in enumerate(comp):
    ax.text(MAIN_CX, Y_FINAL + H_FINAL * (0.045 - 0.155 * i), line,
            ha="center", va="center", fontsize=8.6, zorder=3)

# ---- exclusion boxes ------------------------------------------------------
excl_box(Y[0] - 0.108, "Excluded: prevalent stroke\nat baseline",
         f"{N_STROKE_EXCL:,}")
excl_box(Y[1] - 0.108, "Excluded: missing\ncovariate data",
         f"{N_COV_EXCL:,}")
excl_box(Y[2] - 0.108, "Excluded: missing 2018\nCES-D-10 outcome",
         f"{N_OUT_EXCL:,}")

# ---- arrows ---------------------------------------------------------------
for i in range(len(Y) - 1):
    v_arrow(MAIN_CX, Y[i] - H_MAIN / 2, Y[i + 1] + H_MAIN / 2)
v_arrow(MAIN_CX, Y[3] - H_MAIN / 2, Y_FINAL + H_FINAL / 2)

for i in range(3):
    yb = Y_BRANCH[i]
    h_arrow(yb, MAIN_CX, EXCL_L - 0.006)
    ax.plot([MAIN_CX], [yb], marker="o", markersize=3.2,
            color=ARROW_C, zorder=4)
    y_top = (Y[i] - 0.108) + H_EXCL / 2
    ax.plot([EXCL_CX, EXCL_CX], [yb, y_top], color=ARROW_C,
            linewidth=LW_ARR, solid_capstyle="butt", zorder=1)

# ---- caption-style note ---------------------------------------------------
ax.text(0.5, 0.014,
        "Exclusions applied sequentially, top to bottom. "
        "Harmonized CHARLS, version D.",
        ha="center", va="bottom", fontsize=7.4, color="#555555")

fig.savefig(os.path.join(SUP, "Figure_S1.pdf"))
fig.savefig(os.path.join(SUP, "Figure_S1.png"), dpi=400)
plt.close(fig)

from PIL import Image
_png = os.path.join(SUP, "Figure_S1.png")
im = Image.open(_png)
if im.mode != "RGB":
    _bg = Image.new("RGB", im.size, (255, 255, 255))
    _r = im.convert("RGBA")
    _bg.paste(_r, mask=_r.split()[-1])
    _bg.save(_png, dpi=(400, 400))
    im = Image.open(_png)
raw = open(os.path.join(SUP, "Figure_S1.pdf"), "rb").read()
print("OK Figure_S1.pdf  纯矢量=%s  /CharProcs=%s"
      % (b"/Subtype /Image" not in raw, b"/CharProcs" in raw))
print("OK Figure_S1.png %s %s" % (im.size, im.mode))
print("   源数据: flow_counts.json + table1.json + mets_v2_standard.json")
print("   %s → %s | MetS %s (%.2f%%) | early stroke %s | depressive symptoms %s (%.1f%%)"
      % (f"{N_BASE:,}", f"{N_ANALYTIC:,}", f"{N_METS:,}", N_METS_PCT,
         f"{N_MEDS:,}", f"{N_DEP:,}", N_DEP_PCT))
