#!/usr/bin/env python3
"""Build the 3 EB/EF-by-layer figures (house style) from the per-layer JSON, using a
z-calibrated truth-probe metric so values are comparable across layers (the raw
decision_function / dot-product score explodes with residual-stream norm in late layers
and is NOT comparable).

z-calibration (persona pipeline convention): per layer L and per persona, calibrate by
the NEUTRAL model's era_true/era_false span on that persona's own statements:
    span = et_base - ef_base ;  z(x) = (x - ef_base) / span
so neutral era_false -> 0 and neutral era_true -> 1. Then
    gap(L) = (EB_z_sft - EF_z_sft) - (EB_z_base - EF_z_base)   averaged over personas.

Reads:
  /tmp/sft_ebef_by_layer.json        (Qwen, has eb/ef/et base+sft per layer/persona)
  /tmp/sft_ebef_by_layer_llama.json  (Llama, same schema) -- optional

Figs (all to paper + wiki figure dirs):
  1 fig_ebef_by_layer_sft_qwen   EB orange / EF blue, dashed=neutral, solid=sft, vline L24
  2 fig_ebef_by_layer_sft_llama  same, vline L30
  3 fig_protection_gap_by_layer  gap(L) per model overlaid, axhline 0, vline readout

Usage: python3 make_ebef_by_layer_figs.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DARK = "#141413"
ORANGE = "#d97757"
BLUE = "#6a9bcc"
GREEN = "#788c5d"
DARK_BLUE = "#3d5a80"
MID_GRAY = "#b0aea5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "axes.edgecolor": DARK,
})

OUT_DIRS = [
    Path(os.environ.get("FIG_DIR", ".")),
]
LEGEND_KW = dict(frameon=True, framealpha=0.9, edgecolor="none", facecolor="white")
MIN_SPAN = 1e-3  # drop personas whose neutral era_true/era_false span is degenerate


def load_curves(path):
    d = json.load(open(path))
    layers = sorted(int(L) for L in d["per_layer"].keys())
    eb_neu, eb_sft, ef_neu, ef_sft, gap, frac_pos = [], [], [], [], [], []
    for L in layers:
        rec = d["per_layer"][str(L)]
        valid = [r for r in rec.values()
                 if all(k in r for k in ("eb_base", "ef_base", "et_base", "eb_sft", "ef_sft"))]
        # layer-level span scale: median of correctly-oriented (et>ef) neutral spans
        pos_spans = [r["et_base"] - r["ef_base"] for r in valid
                     if (r["et_base"] - r["ef_base"]) > 0]
        med_span = float(np.median(pos_spans)) if pos_spans else 0.0
        # require each persona's neutral span be correctly oriented and a meaningful
        # fraction of the layer scale, else its z-calibration is degenerate (blows up).
        span_floor = max(MIN_SPAN, 0.25 * med_span)
        EBn, EBs, EFn, EFs, G = [], [], [], [], []
        for r in valid:
            span = r["et_base"] - r["ef_base"]
            if span < span_floor:
                continue
            z = lambda x: (x - r["ef_base"]) / span
            zebn, zebs, zefn, zefs = z(r["eb_base"]), z(r["eb_sft"]), z(r["ef_base"]), z(r["ef_sft"])
            EBn.append(zebn); EBs.append(zebs); EFn.append(zefn); EFs.append(zefs)
            G.append((zebs - zefs) - (zebn - zefn))
        f = lambda v: float(np.mean(v)) if v else np.nan
        eb_neu.append(f(EBn)); eb_sft.append(f(EBs))
        ef_neu.append(f(EFn)); ef_sft.append(f(EFs)); gap.append(f(G))
        frac_pos.append(float(np.mean([g > 0 for g in G])) if G else np.nan)
    return dict(layers=np.array(layers, float),
                eb_neu=np.array(eb_neu), eb_sft=np.array(eb_sft),
                ef_neu=np.array(ef_neu), ef_sft=np.array(ef_sft),
                gap=np.array(gap), frac_pos=np.array(frac_pos))


def save(fig, name):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png")
        fig.savefig(d / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


def fig_ebef(c, model_name, readout, name):
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    L = c["layers"]
    # neutral era-false is the calibration zero (the lower gray line), so it is not
    # drawn as a separate flat curve; neutral era-true is the upper gray line at 1.
    ax.plot(L, c["eb_neu"], color=ORANGE, ls="--", lw=2.0, label="Era-believed (neutral)")
    ax.plot(L, c["eb_sft"], color=ORANGE, ls="-", lw=2.6, label="Era-believed (SFT)")
    ax.plot(L, c["ef_sft"], color=BLUE, ls="-", lw=2.6, label="Era-false (SFT)")
    ax.axhline(0, color=MID_GRAY, lw=0.8, zorder=0)
    ax.axhline(1, color=MID_GRAY, lw=0.8, zorder=0)
    ax.axvline(readout, color=MID_GRAY, ls="--", lw=1.6, zorder=0)
    ax.set_xlabel("layer", fontsize=15)
    ax.set_ylabel("calibrated truth-probe score\n(neutral era-false=0, era-true=1)", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.set_title(f"Era-believed vs era-false probe score\nby layer ({model_name})",
                 loc="left", fontweight="normal", fontsize=14.5)
    # restrict y so degenerate early-layer outliers (tiny neutral span) don't blow up
    # the axis; use the late truth-bearing band (>= readout/2) to set the scale.
    band = L >= (readout / 2.0)
    finite = np.concatenate([c["eb_neu"][band], c["eb_sft"][band], c["ef_sft"][band]])
    finite = finite[np.isfinite(finite)]
    lo, hi = np.nanmin(finite), np.nanmax(finite)
    pad = 0.2 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="best", fontsize=12, **LEGEND_KW)
    save(fig, name)


def fig_gap_single(model_name, c, readout, name, color):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(c["layers"], c["gap"], color=color, lw=2.8)
    ax.axvline(readout, color=color, ls="--", lw=1.6, alpha=0.8, zorder=0,
               label=f"readout L{int(readout)}")
    ax.axhline(0, color=DARK, lw=1.0, zorder=0)
    ax.set_xlabel("layer", fontsize=15)
    ax.set_ylabel("protection gap (z-units)\n(EBsft-EFsft) - (EBbase-EFbase)", fontsize=13.5)
    ax.tick_params(labelsize=14)
    ax.set_title(f"Persona protection gap by layer\n({model_name})", loc="left",
                 fontweight="normal", fontsize=14.5)
    # Fix y to the truth-bearing-band scale; the first few layers have a degenerate
    # neutral era_true/era_false span (probe not yet separating true from false) and
    # produce off-scale spikes that are calibration artifacts, not a protection gap.
    ax.set_ylim(-0.08, 0.22)
    ax.legend(loc="upper right", fontsize=12.5, **LEGEND_KW)
    fig.tight_layout()
    save(fig, name)


def report_bands(name, c):
    L = c["layers"].astype(int)
    g = c["gap"]
    print(f"\n=== {name} z-calibrated gap by layer ===")
    pos = [int(l) for l, x in zip(L, g) if np.isfinite(x) and x > 0]
    neg = [int(l) for l, x in zip(L, g) if np.isfinite(x) and x < 0]
    print(f"  positive-gap layers: {pos}")
    print(f"  negative-gap layers: {neg}")
    for rl in (24, 30):
        if rl in L:
            i = int(np.where(L == rl)[0][0])
            print(f"  gap@L{rl} = {g[i]:+.4f}  (frac personas positive = {c['frac_pos'][i]:.2f})")


def main():
    figdir = os.environ.get("FIG_DIR", ".")
    qpath = f"{figdir}/sft_ebef_by_layer_qwen.json"
    lpath = f"{figdir}/sft_ebef_by_layer_llama.json"
    if not Path(qpath).exists():
        qpath = "/tmp/sft_ebef_by_layer.json"
    if not Path(lpath).exists():
        lpath = "/tmp/sft_ebef_by_layer_llama.json"
    cq = load_curves(qpath)
    fig_ebef(cq, "Qwen3-8B", 24, "fig_ebef_by_layer_sft_qwen")
    report_bands("Qwen3-8B", cq)
    fig_gap_single("Qwen3-8B", cq, 24, "fig_protection_gap_by_layer_qwen", GREEN)

    if Path(lpath).exists():
        cl = load_curves(lpath)
        fig_ebef(cl, "Llama-3.3-70B", 30, "fig_ebef_by_layer_sft_llama")
        report_bands("Llama-3.3-70B", cl)
        fig_gap_single("Llama-3.3-70B", cl, 30, "fig_protection_gap_by_layer_llama", DARK_BLUE)
    else:
        print("\n[!] Llama JSON not present yet; skipping llama figs")


if __name__ == "__main__":
    main()
