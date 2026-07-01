#!/usr/bin/env python3
"""Qwen-3-8B analog of make_fig3_protection_gap_v3.py panel (a): raw protection gap
(Delta_EB - Delta_EF) by induction method, scored with the FROZEN pooled base-Marks
probe (calibrated to the Marks false/true span), gen_prompt=False, statement-end token,
HF Layer 24. All four methods on ONE pipeline (genF acts on oct-darwin), baseline = the
neutral k0/base condition, so the bars are mutually comparable.

  ICL (k=32) / System prompt : base Qwen3-8B + induction prefix  (icl_eb.pt / sp_eb.pt)
  SFT                        : persona-SFT LoRA                   (sft_eb.pt)
  OCT                        : Open Character Training organism   (oct_eb.pt)
  baseline (k0)              : neutral statement, no induction     (base_eb.pt)

gap_method = (eb-ef)_method - (eb-ef)_base, per persona, scored on the pooled base-Marks
probe. House style mirrors fig3_gap_panel: per-method palette, per-persona dots, 95% CI.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

MD = os.environ.get("GENF_MD", "/tmp/genF_qwen_full/genF_marks_disbel_qwen_v3")
L_IDX = int(os.environ.get("L_IDX", "5"))   # HF_LAYERS=[8,12,16,20,22,24,...]; HF24 = idx 5
HIST = sorted(d for d in os.listdir(MD) if os.path.isdir(f"{MD}/{d}"))

# method key -> act file stem, display label, colour (fig3 palette)
METHODS = [
    ("icl",  "ICL\n(k=32)",    "#3d5a80"),
    ("sp",   "System\nprompt", "#d97757"),
    ("sft",  "SFT",            "#788c5d"),
    ("oct",  "OCT",            "#8e6c9b"),
]


def ls(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    A = d["activations"][:, L_IDX, :].float().numpy()
    cats = np.array([m["category"] for m in d["meta"]])
    y = d["labels"].numpy().astype(int)
    return A, cats, y


def fit_z(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), y)
    d = clf.decision_function(sc.transform(X))
    fm = d[y == 0].mean(); tm = d[y == 1].mean(); span = tm - fm
    return lambda A: (clf.decision_function(sc.transform(A)) - fm) / span


def pooled_probe():
    Xs, ys = [], []
    for p in HIST:
        X, _, y = ls(f"{MD}/{p}/base_marks.pt"); Xs.append(X); ys.append(y)
    return fit_z(np.concatenate(Xs), np.concatenate(ys))


def ebef(z, A, cats):
    return float(z(A)[cats == "era_believed"].mean() - z(A)[cats == "era_false"].mean())


def gaps_for(stem, z):
    g = []
    for p in HIST:
        bA, bC, _ = ls(f"{MD}/{p}/base_eb.pt")
        cA, cC, _ = ls(f"{MD}/{p}/{stem}_eb.pt")
        g.append(ebef(z, cA, cC) - ebef(z, bA, bC))
    return np.array(g)


if __name__ == "__main__":
    z = pooled_probe()
    data = {}
    print(f"=== Qwen3-8B protection gap by method (frozen pooled Marks probe, HF24), {len(HIST)} personas ===")
    for stem, label, _ in METHODS:
        g = gaps_for(stem, z); data[stem] = g
        mu = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"  {label.replace(chr(10),' '):14} {mu:+.4f}  CI95=+-{1.96*se:.4f}  ({int((g>0).sum())}/{len(g)} pos)")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    names = [m[1] for m in METHODS]; cols = [m[2] for m in METHODS]
    vals = [data[m[0]].mean() for m in METHODS]
    errs = [1.96 * data[m[0]].std(ddof=1) / np.sqrt(len(HIST)) for m in METHODS]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    x = np.arange(len(names))
    ax.bar(x, vals, 0.6, yerr=errs, color=cols, edgecolor="#141413", linewidth=0.5, capsize=3, zorder=2)
    for i, m in enumerate(METHODS):
        pts = data[m[0]]
        jit = np.linspace(-0.16, 0.16, len(pts))
        ax.scatter(x[i] + jit, pts, s=14, color="#141413", alpha=0.45, zorder=3, linewidths=0)
    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(r"Protection gap $\Delta_{EB}-\Delta_{EF}$ (Marks scale)")
    ax.set_title("Protection gap by induction method (Qwen 3 8B, Layer 24)", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig3_gap_panel_qwen.{ext}", bbox_inches="tight")
    print("saved fig3_gap_panel_qwen.{pdf,png}")
