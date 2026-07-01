#!/usr/bin/env python3
"""EM-vs-persona comparison figure, per family, on all three instruments: whitebox
truth-representation lift, blackbox defend rate (challenge), and blackbox consistent
rate (generalisation). Shows the depth spectrum across three fine-tuning interventions
(shallow -> deep): persona SFT, Open Character Training (OCT), and Emergent Misalignment.

Persona-SFT and EM cells are the published paper numbers (unchanged). OCT cells:
  calibrated lift  = self-probe era gap (oct_era_gap_{llama,qwen}_dual.json): Llama +0.149
                     @L30, Qwen +0.033 @L24, 1.96*sd/sqrt(15) CI.
  defend/consistent = oct_blackbox_mm with-persona run, 15-persona mean +/- 1.96*sd/sqrt(15):
                     Llama 62.3/70.1, Qwen 41.6/53.8.

SFT whitebox lift = 15-persona mean era-believed lift (marks probe; Qwen L24, Llama L30).
EM lift = historical-evil mean (marks probe; Qwen L24, Llama L56).

NOTE (flagged for BEN): OCT defend Llama is 62.3 here (oct_blackbox_mm, matches the spectrum
matrix and shares its source with the plotted CI). The v3 era-false-control run in main.tex
sec:behavioral reports OCT Llama defend 59.2 instead; reconcile which is canonical.

Renders locally into persona-belief-paper/figures/ (the dir main.tex \includegraphics from);
no Modal round-trip needed. Usage: python scripts/probes/modal_em_vs_persona_figure.py
"""
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), "../../../persona-belief-paper/figures")
FAMS = ["Qwen3-8B", "Llama-3.3-70B"]
METHODS = ["sft", "oct", "em"]            # shallow -> deep
LABELS = {"sft": "Persona SFT", "oct": "OCT", "em": "Emergent misalignment"}
COLORS = {"sft": "#2c7fb8", "oct": "#7b5aa6", "em": "#c0584f"}   # blue -> purple -> red
TXTCOL = {"sft": "#1a5276", "oct": "#4a316b", "em": "#7a2f28"}
# whitebox lift (0->1)
# OCT lift = genF (gen_prompt=False) era-believed gap_full at primary layer (results/probes/genF_eratopic_projection.json)
WB = {"Qwen3-8B":     {"sft": 0.038, "oct": 0.089, "em": 0.148},
      "Llama-3.3-70B": {"sft": 0.048, "oct": 0.124, "em": 0.282}}
# blackbox defend% (challenge)
DEF = {"Qwen3-8B":     {"sft": 17.3, "oct": 41.6, "em": 48.0},
       "Llama-3.3-70B": {"sft": 14.2, "oct": 62.3, "em": 56.0}}
# blackbox consistent% (generalisation)
CON = {"Qwen3-8B":     {"sft": 26.4, "oct": 53.8, "em": 79.0},
       "Llama-3.3-70B": {"sft": 34.5, "oct": 70.1, "em": 82.0}}
# 95% CI half-widths, same units as the point values above (lift units for WB; pct points
# for DEF/CON, divided by `scale` at plot time). Each entry is (minus, plus) half-widths.
# WB: sft = 1.96*sd/sqrt(15) over per-persona era-believed lift (Qwen L24; Llama L30);
# EM = propagated bootstrap CI over the 2 historical-evil categories; OCT = 1.96*sd/sqrt(15)
# over per-persona self-probe era gap (oct_era_gap_{llama,qwen}_dual.json, L30/L24).
# BB: sft = 1.96*sd/sqrt(15) pooled-persona; EM = Wilson 95% on pooled rates (n=390 defend,
# 388 consistent); OCT = 1.96*sd/sqrt(15) over the 15 per-persona rates (oct_blackbox_mm).
WB_CI = {"Qwen3-8B":     {"sft": (0.0154, 0.0154), "oct": (0.023, 0.023), "em": (0.0124, 0.0124)},
         "Llama-3.3-70B": {"sft": (0.0868, 0.0868), "oct": (0.039, 0.039), "em": (0.0130, 0.0130)}}
DEF_CI = {"Qwen3-8B":     {"sft": (5.47, 5.47), "oct": (9.59, 9.59), "em": (4.91, 4.95)},
          "Llama-3.3-70B": {"sft": (5.77, 5.77), "oct": (11.04, 11.04), "em": (4.96, 4.85)}}
CON_CI = {"Qwen3-8B":     {"sft": (6.53, 6.53), "oct": (6.03, 6.03), "em": (4.32, 3.75)},
          "Llama-3.3-70B": {"sft": (8.40, 8.40), "oct": (5.04, 5.04), "em": (4.13, 3.51)}}


def _yerr(CI, key, scale):
    """(2, n_fam) array of [minus, plus] half-widths for `key`, or None if any family lacks a CI."""
    import numpy as np
    if CI is None or any(CI[f].get(key) is None for f in FAMS):
        return None
    lo = np.array([CI[f][key][0] / scale for f in FAMS])
    hi = np.array([CI[f][key][1] / scale for f in FAMS])
    return np.vstack([np.maximum(lo, 0.0), np.maximum(hi, 0.0)])


def _bars(ax, D, scale, title, ylab, CI=None):
    import numpy as np
    x = np.arange(len(FAMS)); w = 0.26
    offs = {"sft": -w, "oct": 0.0, "em": w}
    eb_kw = dict(ecolor="#444444", capsize=4, error_kw=dict(lw=1.4))
    for m in METHODS:
        vals = [D[f][m] / scale for f in FAMS]
        err = _yerr(CI, m, scale)
        ax.bar(x + offs[m], vals, w, color=COLORS[m], label=LABELS[m], yerr=err, **eb_kw)
        tops = [v + (err[1][i] if err is not None else 0) for i, v in enumerate(vals)]
        for xi, v, t in zip(x + offs[m], vals, tops):
            ax.text(xi, t + 0.02, f"{v:.2f}", ha="center", fontsize=12,
                    color=TXTCOL[m], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(FAMS, fontsize=15.5)
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=16.5, loc="left")
    ax.set_ylabel(ylab, fontsize=15.5)
    ax.tick_params(axis="y", labelsize=16)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def render(outdir=FIG_DIR):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)

    # --- Figure A: whitebox truth-probe calibration (0 = false, 1 = true) ---
    figA, axA = plt.subplots(figsize=(4.8, 3.9))
    _bars(axA, WB, 1.0, "Truth probe:\ntruth-representation lift",
          "Probe calibration:\n0 = false, 1 = true", CI=WB_CI)
    # EM first in the legend order
    LEG_ORDER = [LABELS["em"], LABELS["oct"], LABELS["sft"]]
    hA, lA = axA.get_legend_handles_labels()
    oA = [lA.index(t) for t in LEG_ORDER]
    axA.legend([hA[i] for i in oA], [lA[i] for i in oA], loc="upper left", frameon=False, fontsize=13)
    figA.tight_layout()
    for ext in ("png", "pdf"):
        figA.savefig(f"{outdir}/em_vs_persona_whitebox.{ext}", dpi=200, bbox_inches="tight")

    # --- Figure B: black-box behaviour (0 = never, 1 = always) ---
    figB, axesB = plt.subplots(1, 2, figsize=(7.7, 3.9))
    _bars(axesB[0], DEF, 100.0, "Defend under challenge", "Rate: 0 = never, 1 = always", CI=DEF_CI)
    _bars(axesB[1], CON, 100.0, "Consistent under generalization", "", CI=CON_CI)
    # legend inside the left (Defend) panel, EM first, instead of floating above
    h, l = axesB[0].get_legend_handles_labels()
    oB = [l.index(t) for t in (LABELS["em"], LABELS["oct"], LABELS["sft"])]
    axesB[0].legend([h[i] for i in oB], [l[i] for i in oB], loc="upper left", frameon=False, fontsize=13)
    figB.tight_layout()
    for ext in ("png", "pdf"):
        figB.savefig(f"{outdir}/em_vs_persona_blackbox.{ext}", dpi=200, bbox_inches="tight")
    return outdir


if __name__ == "__main__":
    print("wrote em_vs_persona_{whitebox,blackbox}.{png,pdf} to", render())
