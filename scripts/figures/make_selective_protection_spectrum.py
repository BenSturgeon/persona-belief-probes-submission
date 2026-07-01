#!/usr/bin/env python3
"""Behavioral selective-protection spectrum: era_believed vs matched era_false
defend rate across the four induction methods (shallow -> deep), both models.
Data: v3 clean dataset, pooled over 15 historical personas (figures/data_v3 +
persona_blackbox_* summaries). House style: DejaVu, thin title, legend in the
empty upper-left."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

METHODS = ["ICL\n(k=32)", "System\nprompt", "SFT", "OCT"]
# defend % (era_believed, era_false), v3, pooled over 15 personas
LLAMA = {"eb": [0.4, 6.9, 14.2, 59.2], "ef": [0.0, 0.0, 0.5, 13.5]}
QWEN  = {"eb": [0.2, 4.8, 17.3, 42.8], "ef": [0.0, 0.6, 3.7, 16.4]}

# per-method palette (matches Fig 3 + the era-disbelieved demotion figure)
METH_COLORS = ["#3d5a80", "#d97757", "#788c5d", "#8e6c9b"]  # ICL, sysprompt, SFT, OCT
EF_ALPHA = 0.4   # era-false control = faded version of the method colour

from matplotlib.patches import Patch

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
x = np.arange(len(METHODS)); w = 0.38
for ax, (name, d) in zip(axes, [("Llama 3.3 70B", LLAMA), ("Qwen 3 8B", QWEN)]):
    ax.bar(x - w/2, d["eb"], w, color=METH_COLORS)
    ax.bar(x + w/2, d["ef"], w, color=METH_COLORS, alpha=EF_ALPHA)
    for xi, (a, b) in enumerate(zip(d["eb"], d["ef"])):
        ax.text(xi - w/2, a + 1.2, f"{a:.0f}", ha="center", va="bottom", fontsize=8, color=METH_COLORS[xi])
        ax.text(xi + w/2, b + 1.2, f"{b:.0f}", ha="center", va="bottom", fontsize=8, color="#5d6d7e")
    ax.set_xticks(x); ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_title(name, fontsize=10, fontweight="regular")
    ax.set_ylim(0, 70)
axes[0].set_ylabel("Defend rate under challenge (%)")
leg = [Patch(facecolor="#555555", label="era-believed"),
       Patch(facecolor="#555555", alpha=EF_ALPHA, label="era-false (matched control)")]
axes[0].legend(handles=leg, loc="upper left", frameon=False, fontsize=8.5)
fig.suptitle("Behavioral protection is selective and scales with induction depth",
             fontsize=11, fontweight="regular", y=1.02)
for ext in ("pdf", "png"):
    fig.savefig(f"fig_selective_protection_spectrum.{ext}")
print("saved fig_selective_protection_spectrum.{pdf,png}")
