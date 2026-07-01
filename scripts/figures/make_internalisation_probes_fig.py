#!/usr/bin/env python3
"""Fig: deep character training internalises in the model's own truth code.
Two panels -- Protection (era-endorsed falsehoods, Delta_EB-Delta_EF) and Demotion
(era-rejected truths, Delta_ET-Delta_ED) -- each showing SFT vs OCT, native vs frozen
probe (Llama-3.3-70B, Layer 56, gen_prompt=False). Replaces tab:internalisation_probes."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})

COL = {"SFT": "#788c5d", "OCT": "#8e6c9b"}
ORG = ["SFT", "OCT"]
# (native, frozen, native_pos, frozen_pos) per organism
PROTECT = {"SFT": (0.039, 0.001, "14/15", "8/15"),  "OCT": (0.201, 0.043, "15/15", "14/15")}
DEMOTE  = {"SFT": (-0.029, -0.009, "2/15", "1/15"), "OCT": (0.146, -0.004, "14/15", "4/15")}
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
        ax.text(x[i] - w/2, nat[i] + (0.006 if nat[i] >= 0 else -0.014), D[o][2],
                ha="center", va="bottom" if nat[i] >= 0 else "top", fontsize=7.5, color=COL[o])
        ax.text(x[i] + w/2, frz[i] + (0.006 if frz[i] >= 0 else -0.014), D[o][3],
                ha="center", va="bottom" if frz[i] >= 0 else "top", fontsize=7.5, color="#777")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_ylim(-0.09, 0.225)   # extra room below the negative bars + labels
    ax.set_xticks(x); ax.set_xticklabels(ORG, fontsize=11)
    ax.tick_params(axis="x", pad=8)   # more space between bars and x labels
    ax.set_title(title, fontsize=9.5)
axes[0].set_ylabel("Calibrated gap (0 = false, 1 = true)")
leg = [Patch(facecolor=COL["SFT"], edgecolor="#141413", linewidth=0.5, label="SFT"),
       Patch(facecolor=COL["OCT"], edgecolor="#141413", linewidth=0.5, label="OCT"),
       Patch(facecolor="#999999", edgecolor="#141413", linewidth=0.5, label="solid: native (own probe)"),
       Patch(facecolor="#999999", alpha=0.4, edgecolor="#141413", linewidth=0.5, label="faded: frozen (base probe)")]
axes[0].legend(handles=leg, loc="upper left", frameon=False, fontsize=7.5, handlelength=1.2, labelspacing=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"fig_internalisation_probes.{ext}", bbox_inches="tight")
print("saved fig_internalisation_probes.{pdf,png}")
