#!/usr/bin/env python3
"""Binned calibration figure: probe-score deciles vs behavioral rates, both arms."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
})

rows = json.load(open("/tmp/perstmt/perstatement_rows.json"))


def zin(rs, xkey):
    """Within-group z-score of xkey."""
    out = {}
    groups = set(r["group"] for r in rs)
    for g in groups:
        v = np.array([r[xkey] for r in rs if r["group"] == g])
        out[g] = (v.mean(), v.std())
    return np.array([(r[xkey] - out[r["group"]][0]) / out[r["group"]][1] for r in rs])


def binned(rs, xkey, ykey, nbins=10):
    sub = [r for r in rs if r[ykey] is not None]
    x = zin(sub, xkey)
    y = np.array([r[ykey] for r in sub], float)
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9
    xs, ys, es = [], [], []
    for i in range(nbins):
        m = (x > qs[i]) & (x <= qs[i + 1])
        if m.sum() == 0:
            continue
        xs.append(x[m].mean())
        p = y[m].mean()
        ys.append(p)
        es.append(np.sqrt(p * (1 - p) / m.sum()))
    return np.array(xs), np.array(ys), np.array(es)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

panels = [
    (axes[0], rows["persona_rows"], "sft_score",
     "Persona-SFT (15 personas, 1800 era-believed statements)",
     "Probe score under SFT (z within persona)"),
    (axes[1], rows["em_rows"], "lift",
     "EM organism (13 categories, 390 propositions)",
     "Calibrated probe lift (z within category)"),
]
colors = {"defend": "#1f77b4", "consistent": "#ff7f0e"}
labels = {"defend": "Defend under challenge", "consistent": "Consistent in generalisation"}

for ax, rs, xkey, title, xlabel in panels:
    for ykey in ("defend", "consistent"):
        xs, ys, es = binned(rs, xkey, ykey)
        ax.errorbar(xs, 100 * ys, yerr=100 * es, marker="o", ms=4.5, lw=1.6,
                    capsize=2.5, color=colors[ykey], label=labels[ykey])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=11.5, fontweight="normal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)

axes[0].set_ylabel("Behavioral rate (%)", fontsize=12)
axes[0].legend(loc="upper left", fontsize=9.5, framealpha=0.9)
axes[1].legend(loc="lower left", fontsize=9.5, framealpha=0.9)
fig.suptitle("Per-statement probe score vs behavioral belief tests (probe-score deciles)",
             fontsize=12.5, fontweight="normal")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.environ.get("FIG_OUT", "perstatement_probe_behavior.png")
fig.savefig(out, dpi=200)
print("saved", out)
