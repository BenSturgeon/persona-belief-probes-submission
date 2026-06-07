"""Cross-family EM truth-representation figure (bars + error bars).

One panel, grouped by stratum (the headline structure), three family bars per
category. Per-category 95% bootstrap CIs are shown as error bars.

Qwen2.5-14B: means + CIs loaded from the vllm-lens analysis.json (HF L32 / lens L31).
Qwen3-8B and Llama-3.3-70B: per_FALSE-prop lifts re-computed via
analyze_probe_replication.lift_profile() at the headline layer, CIs via boot_ci.

Outputs:
  probe_repl/cross_family_lift_bar.png
  probe_repl/cross_family_lift_bar.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_probe_replication import (CAT_LABEL, PRIMARY_LAYER, STRATA,
                                        boot_ci, lift_profile)

# Qwen2.5 vllm-lens analysis (headline HF L32).
QWEN25_ANALYSIS = ROOT / "probe_repl" / "qwen25_14b_vllm_lens" / "analysis.json"


def qwen25_means_cis():
    """Return ({cat: mean}, {cat: (lo, hi)}) from vllm-lens analysis JSON."""
    d = json.loads(QWEN25_ANALYSIS.read_text())
    hl = d["by_layer"]["32"]
    means = {c: float(v) for c, v in hl["ours_mean"].items()}
    cis = {c: (float(lo), float(hi)) for c, (lo, hi) in hl["ours_ci"].items()}
    return means, cis


def family_means_cis(fam):
    """Compute per-cat mean + 95% bootstrap CI for one of the trained families."""
    d = ROOT / "probe_repl" / fam
    load = lambda f: torch.load(d / f, map_location="cpu", weights_only=False)
    _, _, _, by_cat, ours, _, _ = lift_profile(
        load("marks_base.pt"), load("marks_em.pt"),
        load("props_base.pt"), load("props_em.pt"),
        PRIMARY_LAYER[fam],
    )
    cis = {}
    for c, v in by_cat.items():
        lo, hi = boot_ci(np.asarray(v))
        cis[c] = (float(lo), float(hi))
    return ours, cis


q25_m, q25_ci = qwen25_means_cis()
q3_m, q3_ci = family_means_cis("qwen3_8b")
ll_m, ll_ci = family_means_cis("llama33_70b")

fams = [
    ("Qwen2.5-14B",    q25_m, q25_ci, "#222222"),
    ("Qwen3-8B",       q3_m,  q3_ci,  "#2c7fb8"),
    ("Llama-3.3-70B",  ll_m,  ll_ci,  "#d95f0e"),
]

# Row order: strata groups, each sorted by Qwen2.5 lift.
groups = [
    ("HISTORICAL-EVIL",    STRATA["historical_evil"]),
    ("GENERIC-CHARGED",    STRATA["charged"]),
    ("ANTI-HUMAN",         ["anti_human_ai_dominance"]),   # behaviour without belief
    ("NEUTRAL / POSITIVE", STRATA["controls"]),
]
rows, group_spans = [], []
y = 0
for gname, cats in groups:
    cats = sorted(cats, key=lambda c: -q25_m.get(c, 0))
    start = y
    for c in cats:
        rows.append((c, y))
        y += 1
    group_spans.append((gname, start, y - 1))
    y += 0.6
ymax = y

DODGE = [-0.24, 0.0, 0.24]


def render():
    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(rows) + 1.6))

    # Light row separators within a stratum; firmer rule between strata.
    ys_orig = [yy for _, yy in rows]
    for i in range(len(ys_orig) - 1):
        gap = ys_orig[i + 1] - ys_orig[i]
        mid = ymax - (ys_orig[i] + ys_orig[i + 1]) / 2
        if gap < 1.3:
            ax.plot([-0.32, 0.44], [mid, mid], color="#ededed", lw=0.6, zorder=0)
        else:
            ax.plot([-0.32, 0.44], [mid, mid], color="#c9c9c9", lw=1.3, zorder=0)

    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=1)

    for (fname, means, cis, color), dy in zip(fams, DODGE):
        xs = np.array([means.get(c, np.nan) for c, _ in rows])
        ys = np.array([ymax - yy + dy for _, yy in rows])
        # Error bars: lower / upper from CI, asymmetric vs mean.
        los = np.array([cis.get(c, (np.nan, np.nan))[0] for c, _ in rows])
        his = np.array([cis.get(c, (np.nan, np.nan))[1] for c, _ in rows])
        err = np.vstack([xs - los, his - xs])
        ax.barh(ys, xs, height=0.28, color=color, zorder=3,
                label=fname, edgecolor="none")
        ax.errorbar(xs, ys, xerr=err, fmt="none", ecolor="#444", elinewidth=0.7,
                    capsize=2.0, capthick=0.7, zorder=4)

    for c, yy in rows:
        ax.text(-0.135, ymax - yy, CAT_LABEL.get(c, c),
                ha="right", va="center", fontsize=8.0)

    for gname, s, e in group_spans:
        ytop, ybot = ymax - s, ymax - e
        ax.plot([-0.335, -0.335], [ybot - 0.28, ytop + 0.28],
                color="#999", lw=1.0, clip_on=False)
        ax.text(-0.37, (ytop + ybot) / 2, gname,
                ha="center", va="center", fontsize=7.0,
                rotation=90, color="#555", fontweight="bold")

    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.1, 0.45)
    ax.set_yticks([])
    ax.set_xlim(-0.40, 0.46)
    ax.set_ylim(-0.4, ymax + 0.4)
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel(
        "Shift toward 'true' vs aligned base   (0 = false, 1 = true)",
        fontsize=9,
    )
    ax.tick_params(length=3)

    # Legend, bottom-right inside the plot area.
    ax.legend(
        loc="lower right", frameon=True, fancybox=False,
        edgecolor="#bbbbbb", facecolor="white", framealpha=0.95,
        ncol=1, fontsize=8.5,
        handlelength=1.4, handletextpad=0.5,
    )

    ax.set_title(
        "EM shifts representation scores in a similar direction across 3 models",
        fontsize=12, fontweight="bold", loc="left", pad=23,  # ~0.8 cm
    )

    fig.tight_layout()
    out_png = ROOT / "probe_repl" / "cross_family_lift_bar.png"
    out_pdf = ROOT / "probe_repl" / "cross_family_lift_bar.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_png}")
    print(f"-> {out_pdf}")


def render_vertical():
    """Vertical (categories on x-axis, lift on y-axis)."""
    fig, ax = plt.subplots(figsize=(0.6 * len(rows) + 2.2, 6.4))

    # Same row order as horizontal; here mapped to x-positions.
    xs_pos = [xx for _, xx in rows]
    ymin_axis, ymax_axis = -0.18, 0.46

    # Stratum band separators (vertical lines between groups).
    for i in range(len(xs_pos) - 1):
        gap = xs_pos[i + 1] - xs_pos[i]
        mid = (xs_pos[i] + xs_pos[i + 1]) / 2
        if gap < 1.3:
            ax.plot([mid, mid], [ymin_axis, ymax_axis], color="#ededed",
                    lw=0.6, zorder=0)
        else:
            ax.plot([mid, mid], [ymin_axis, ymax_axis], color="#c9c9c9",
                    lw=1.3, zorder=0)

    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)

    for (fname, means, cis, color), dx in zip(fams, DODGE):
        ys = np.array([means.get(c, np.nan) for c, _ in rows])
        xs = np.array([xx + dx for _, xx in rows])
        los = np.array([cis.get(c, (np.nan, np.nan))[0] for c, _ in rows])
        his = np.array([cis.get(c, (np.nan, np.nan))[1] for c, _ in rows])
        err = np.vstack([ys - los, his - ys])
        ax.bar(xs, ys, width=0.28, color=color, zorder=3,
               label=fname, edgecolor="none")
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="#444",
                    elinewidth=0.7, capsize=2.0, capthick=0.7, zorder=4)

    # Category tick labels rotated 35°.
    ax.set_xticks(xs_pos)
    ax.set_xticklabels([CAT_LABEL.get(c, c) for c, _ in rows],
                        rotation=35, ha="right", fontsize=8.0)

    # Stratum group labels above the chart with brackets.
    # s, e in group_spans are the actual x-positions of the first/last category in the group.
    ytop_bracket = ymax_axis + 0.015
    ytop_label = ymax_axis + 0.05
    for gname, s, e in group_spans:
        xs_l, xs_r = s, e
        ax.plot([xs_l - 0.28, xs_r + 0.28], [ytop_bracket, ytop_bracket],
                color="#999", lw=1.0, clip_on=False)
        ax.text((xs_l + xs_r) / 2, ytop_label, gname, ha="center", va="bottom",
                fontsize=7.0, color="#555", fontweight="bold")

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_bounds(-0.1, 0.45)   # range-frame
    ax.set_xlim(-0.6, xs_pos[-1] + 0.6)
    ax.set_ylim(ymin_axis, ymax_axis + 0.12)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylabel(
        "Shift toward 'true' vs aligned base   (0 = false, 1 = true)",
        fontsize=9,
    )
    ax.tick_params(length=3)

    # Anchor slightly below axes-top so it clears the stratum bracket labels.
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.0, 0.86),
        frameon=True, fancybox=False,
        edgecolor="#bbbbbb", facecolor="white", framealpha=0.95,
        ncol=1, fontsize=8.5,
        handlelength=1.4, handletextpad=0.5,
    )
    ax.set_title(
        "EM shifts representation scores in a similar direction across 3 models",
        fontsize=12, fontweight="bold", loc="left", pad=23,  # ~0.8 cm
    )

    fig.tight_layout()
    out_png = ROOT / "probe_repl" / "cross_family_lift_bar_vertical.png"
    out_pdf = ROOT / "probe_repl" / "cross_family_lift_bar_vertical.pdf"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_png}")
    print(f"-> {out_pdf}")


render()
render_vertical()
