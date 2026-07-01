#!/usr/bin/env python3
"""Qwen-3-8B analog of make_internalisation_probes_fig.py.
Two panels -- Demotion (era-rejected truths, Delta_ET-Delta_ED) and Protection
(era-endorsed falsehoods, Delta_EB-Delta_EF) -- each SFT vs OCT, native vs frozen
probe (Qwen-3-8B, Layer 24 = HF24, gen_prompt=False).

Numbers are the canonical genF (gen_prompt=False) recompute on the Qwen genF acts
(oct-darwin:/probe/genF_marks_disbel_qwen_v3, base/sft/oct), scored by
_score_qwen_genF_sft.py -> /tmp/qwen_genF_sft_oct_cells.json:
  native  = per-organism self Marks probe, diff-in-diff in native-z.
  frozen  = base-model Marks probe (pooled-base-marks convention).
OCT cells reproduce genF_internalisation_recompute.json qwen_HF24 exactly.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

COL = {"SFT": "#788c5d", "OCT": "#8e6c9b"}
ORG = ["SFT", "OCT"]
# (native, frozen, native_pos, frozen_pos) per organism, Qwen-3-8B HF24
PROTECT = {"SFT": (0.0465, 0.0891, "14/15", "14/15"), "OCT": (0.0889, 0.0498, "15/15", "14/15")}
DEMOTE  = {"SFT": (0.0100, -0.0410, "9/15", "2/15"),   "OCT": (0.0359, 0.0429, "10/15", "13/15")}
# Demotion on the left so the legend fits its empty upper-left; Protection on the right
PANELS = [("Demotion of era-rejected truths\n($\\Delta_{ET}-\\Delta_{ED}$)", DEMOTE),
          ("Protection of era-endorsed falsehoods\n($\\Delta_{EB}-\\Delta_{EF}$)", PROTECT)]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
x = np.arange(len(ORG)); w = 0.36
for ax, (title, D) in zip(axes, PANELS):
    nat = [D[o][0] for o in ORG]; frz = [D[o][1] for o in ORG]; cols = [COL[o] for o in ORG]
    ax.bar(x - w/2, nat, w, color=cols, edgecolor="#141413", linewidth=0.5)
    ax.bar(x + w/2, frz, w, color=cols, alpha=0.4, edgecolor="#141413", linewidth=0.5)
    for i, o in enumerate(ORG):
        ax.text(x[i] - w/2, nat[i] + (0.003 if nat[i] >= 0 else -0.006), D[o][2],
                ha="center", va="bottom" if nat[i] >= 0 else "top", fontsize=7.5, color=COL[o])
        ax.text(x[i] + w/2, frz[i] + (0.003 if frz[i] >= 0 else -0.006), D[o][3],
                ha="center", va="bottom" if frz[i] >= 0 else "top", fontsize=7.5, color="#777")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylim(-0.065, 0.105)
    ax.set_xticks(x); ax.set_xticklabels(ORG, fontsize=11)
    ax.tick_params(axis="x", pad=8)
    ax.set_title(title, fontsize=9.5)
axes[0].set_ylabel("Calibrated gap (Marks false/true scale)")
leg = [Patch(facecolor="#555555", label="native (organism's own probe)"),
       Patch(facecolor="#555555", alpha=0.4, label="frozen (base-model probe)")]
axes[0].legend(handles=leg, loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_internalisation_probes_qwen.{ext}", bbox_inches="tight")
print("saved fig_internalisation_probes_qwen.{pdf,png}")
