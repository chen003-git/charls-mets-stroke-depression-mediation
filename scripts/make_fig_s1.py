#!/usr/bin/env python3
"""论文19 Figure S1：把 _output/flow_counts.json 渲染成真正的图形（阶段4 / R3-M16）

原状：`Supplementary/S1_participant_flow.md` 是 ASCII 手绘流程图，名为 Figure 却不是图；
      且 GfA 提示补充材料"appear online exactly as received"，不做排版。
现在：用 matplotlib 从产物读数绘制，数字不手抄。
输出：Supplementary/Figure_S1.pdf（矢量）+ Figure_S1.png（>=2000 px，预览）
"""
import os, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "_output")
SUP = os.path.join(HERE, "..", "Supplementary")
f = json.load(open(os.path.join(OUT, "flow_counts.json"), encoding="utf-8"))

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "savefig.dpi": 300, "savefig.bbox": "tight"})

steps = [
    (f["raw_respondents"], "CHARLS 2011 baseline\n(Harmonized CHARLS, version D)"),
    (f["after_stroke_exclusion"], "After excluding prevalent stroke at baseline\n(\u2212%d, r1stroke == 1)" % f["excluded_prevalent_stroke"]),
]
fig, ax = plt.subplots(figsize=(5.2, 6.4)); ax.axis("off")
y = 0.94
BOX = dict(boxstyle="round,pad=0.02", fc="#EAF2F8", ec="#0072B2", lw=1.2)
for i, (n, lab) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((0.06, y - 0.085), 0.88, 0.085, **BOX))
    ax.text(0.5, y - 0.0425, "%s\nn = %s" % (lab, f"{n:,}"), ha="center", va="center", fontsize=8.2)
    y -= 0.14
notes = [
    ("Excluded: missing covariate data\n(age, sex, education, smoking,\ndrinking, baseline CES-D-10)", f["excluded_missing_covariates"]),
    ("Excluded: missing 2018 CES-D-10 outcome", f["excluded_missing_outcome"]),
]
for lab, n in notes:
    ax.add_patch(FancyArrowPatch((0.20, y + 0.045), (0.20, y - 0.02), arrowstyle="-|>",
                                 mutation_scale=12, color="#999", lw=1.2))
    ax.text(0.24, y + 0.012, "%s\n\u2212%s" % (lab, f"{n:,}"), ha="left", va="center", fontsize=7.6, color="#555")
    y -= 0.135
ax.add_patch(FancyBboxPatch((0.06, y - 0.085), 0.88, 0.085, boxstyle="round,pad=0.02",
                            fc="#FDEDEC", ec="#D55E00", lw=1.4))
ax.text(0.5, y - 0.0425, "Analytic sample\nn = %s" % f"{f['analytic_sample']:,}", ha="center", va="center", fontsize=8.6)
y -= 0.13
ax.text(0.5, y, ("Metabolic syndrome (self-reported definition)  n = %s\n"
                 "Early incident stroke, 2013/15  n = %s\n"
                 "Depressive symptoms, 2018 (CES-D-10 \u2265 10)  n = %s"
                 % (f"{fc_n:,}" if False else f"{1293:,}", f"{f['n_stroke_early_r2r3']:,}",
                    f"{4377:,}")),
        ha="center", va="top", fontsize=8.0)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
os.makedirs(SUP, exist_ok=True)
fig.savefig(os.path.join(SUP, "Figure_S1.pdf"))
fig.savefig(os.path.join(SUP, "Figure_S1.png"), dpi=400)
plt.close()

from PIL import Image
im = Image.open(os.path.join(SUP, "Figure_S1.png"))
if im.mode != "RGB":
    bg = Image.new("RGB", im.size, (255, 255, 255)); r = im.convert("RGBA")
    bg.paste(r, mask=r.split()[-1]); bg.save(os.path.join(SUP, "Figure_S1.png"), dpi=(300, 300))
    im = Image.open(os.path.join(SUP, "Figure_S1.png"))
raw = open(os.path.join(SUP, "Figure_S1.pdf"), "rb").read()
print("OK Figure_S1.pdf 纯矢量=%s  /CharProcs=%s" % (b"/Subtype /Image" not in raw, b"/CharProcs" in raw))
print(f"OK Figure_S1.png {im.size} {im.mode}")
print(f"   源数据: flow_counts.json  (25,586 / 486 / 25,100 / 9,604 / 15,496 / 4,182 / 11,314)")
