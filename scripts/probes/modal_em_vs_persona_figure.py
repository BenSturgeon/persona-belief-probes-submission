#!/usr/bin/env python3
"""EM-vs-persona comparison figure, per family, on all three instruments: whitebox
truth-representation lift, blackbox defend rate (challenge), and blackbox consistent
rate (generalisation). Shows the depth spectrum across three fine-tuning interventions
(shallow -> deep): persona SFT, Open Character Training (OCT), and Emergent Misalignment.

Values are read from em_vs_persona_values.json (see its "_about" and "provenance" fields):
  whitebox lift = z(fine-tuned) - z(base) on the arm's target false statements, each model's own
                  Marks probe calibrated false=0 / true=1; Llama HF L56, Qwen HF L24; SFT and OCT
                  from the genF extraction (user-only chat template, no system prompt, v3 statements).
  defend/consistent = judged behavioural rates (SFT, OCT: per-persona mean; EM: pooled items).

Renders locally into $FIG_DIR (default: this directory); no Modal round-trip needed. Usage: python scripts/probes/modal_em_vs_persona_figure.py
"""
import os, json

FIG_DIR = os.environ.get("FIG_DIR", os.path.dirname(os.path.abspath(__file__)))
FAMS = ["Qwen3-8B", "Llama-3.3-70B"]
METHODS = ["sft", "oct", "em"]            # shallow -> deep
LABELS = {"sft": "Persona SFT", "oct": "OCT", "em": "Emergent misalignment"}
COLORS = {"sft": "#2c7fb8", "oct": "#7b5aa6", "em": "#c0584f"}   # blue -> purple -> red
TXTCOL = {"sft": "#1a5276", "oct": "#4a316b", "em": "#7a2f28"}
# whitebox lift (0->1)
# Values are read from em_vs_persona_values.json (next to this script), produced by the
# read-only recomputes; see its "_about" and "provenance" fields.
_V = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "em_vs_persona_values.json")))
def _pt(block, scale=1.0):
    return {f: {m: block[f][m]["mean"] for m in METHODS} for f in FAMS}
def _ci(block):
    out = {}
    for f in FAMS:
        out[f] = {}
        for m in METHODS:
            c = block[f][m]
            if "ci" in c: out[f][m] = (c["ci"], c["ci"])
            else: out[f][m] = (c["mean"] - c["lo"], c["hi"] - c["mean"])
    return out
WB, DEF, CON = _pt(_V["WB"]), _pt(_V["DEF"]), _pt(_V["CON"])
WB_CI, DEF_CI, CON_CI = _ci(_V["WB"]), _ci(_V["DEF"]), _ci(_V["CON"])


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
