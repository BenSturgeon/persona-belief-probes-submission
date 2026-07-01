"""Convention-robustness comparison figure: em-belief vs persona convention,
both families. Reads the em-belief dirs (probe_repl/<family>/) and the persona
dirs (probe_repl/<family>_persona/), recomputes Family-B metrics with the SAME
code path, and draws a 2x2 comparison:
  - left column: per-category lift, em-belief (headline layer) vs persona (canonical)
  - right column: r-vs-Qwen2.5 across the layer sweep, both conventions

Usage:
  uv run python scripts/figure_convention_robustness.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_probe_replication_persona import (  # noqa: E402
    metrics_at, lift_profile, boot_ci, CAT_LABEL, STRATA, QWEN25_LIFT,
)

ROOT = Path(__file__).resolve().parents[1]
# em-belief headline layers (from analyze_probe_replication.PRIMARY_LAYER) and
# persona canonical layers (from analyze_probe_replication_persona).
EMBELIEF_LAYER = {"qwen3_8b": 24, "llama33_70b": 56}
PERSONA_LAYER = {"qwen3_8b": 22, "llama33_70b": 30}


def load_pack(subdir):
    d = ROOT / "probe_repl" / subdir
    L = lambda f: torch.load(d / f, map_location="cpu", weights_only=False)
    return L("marks_base.pt"), L("marks_em.pt"), L("props_base.pt"), L("props_em.pt")


def per_cat(mb, me, pb, pe, L):
    _, fi, cat, by_cat, ours, _, _ = lift_profile(mb, me, pb, pe, L)
    return by_cat


def main():
    fams = ["qwen3_8b", "llama33_70b"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11),
                             gridspec_kw={"width_ratios": [2, 1.1]})
    controls = set(STRATA["controls"])
    for row, fam in enumerate(fams):
        eb = load_pack(fam)
        pers = load_pack(f"{fam}_persona")
        Leb, Lp = EMBELIEF_LAYER[fam], PERSONA_LAYER[fam]
        bc_eb = per_cat(*eb, Leb)
        bc_p = per_cat(*pers, Lp)

        cats = sorted(bc_eb, key=lambda c: -np.mean(bc_eb[c]))
        y = np.arange(len(cats))
        eb_means = [np.mean(bc_eb[c]) for c in cats]
        p_means = [np.mean(bc_p[c]) for c in cats]
        ax = axes[row, 0]
        h = 0.38
        ax.barh(y - h / 2, eb_means, height=h, color="#c44536", label=f"em-belief (L{Leb})")
        ax.barh(y + h / 2, p_means, height=h, color="#2a6f97", label=f"persona (L{Lp})")
        ax.axvline(0, ls="--", color="#555", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([CAT_LABEL.get(c, c) for c in cats], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("EM lift (z_EM - z_aligned) on FALSE props")
        ax.set_title(f"{fam}: per-category lift, em-belief vs persona convention",
                     fontsize=11, fontweight="bold", loc="left")
        ax.legend(fontsize=9, loc="lower right")

        # right: r-vs-Q25 sweep both conventions
        ax2 = axes[row, 1]
        Leb_layers = eb[0]["layers"]; Lp_layers = pers[0]["layers"]
        r_eb = [metrics_at(*eb, L)["r_vs_q25"] for L in Leb_layers]
        r_p = [metrics_at(*pers, L)["r_vs_q25"] for L in Lp_layers]
        ax2.plot(Leb_layers, r_eb, "-o", color="#c44536", label="em-belief")
        ax2.plot(Lp_layers, r_p, "-s", color="#2a6f97", label="persona")
        ax2.axhline(0.3, ls=":", color="#888", lw=1)
        ax2.axvline(Leb, ls="--", color="#c44536", lw=1)
        ax2.axvline(Lp, ls="--", color="#2a6f97", lw=1)
        ax2.set_xlabel("layer"); ax2.set_ylabel("r vs Qwen2.5")
        ax2.set_title(f"{fam}: lift-profile r across layers\n(dashed = headline/canonical layer)",
                      fontsize=10, loc="left")
        ax2.legend(fontsize=9)

    fig.suptitle("Convention robustness: em-belief (gen_prompt=True) vs persona "
                 "(gen_prompt=False) — EM truth-representation lift",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = ROOT / "probe_repl" / "convention_robustness.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
