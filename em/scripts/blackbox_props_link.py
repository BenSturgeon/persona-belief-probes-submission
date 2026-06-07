import os
"""Link whitebox probe lift to the proposition-based blackbox behavioural test
across ALL categories.

Reuses category_lift() from blackbox_link.py (whitebox per-category lift on FALSE
props). Computes, from outputs/blackbox_props/judged/:
  - EM defend rate (challenge, DEFEND)
  - EM consistent rate (generalisation, CONSISTENT_WITH_BELIEF)
  - aligned control rates
Then correlates lift vs EM defend% and lift vs EM consistent% (Pearson + Spearman),
prints a per-category table, and writes the scatter figure.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scripts.blackbox_link import category_lift

ROOT = Path(__file__).resolve().parents[1]
BB = ROOT / "outputs" / "blackbox_props" / "judged"
OUT = ROOT / "analysis" / "figures_propositions"
COPY_TO = Path(os.environ.get("COPY_TO_DIR", str(ROOT / "analysis" / "revalidation")))


def rate(model, ptype, target):
    path = BB / f"{model}__{ptype}.judged.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["category"]][0] += (r["judge_label"] == target)
        by[r["category"]][1] += 1
    return {c: d / n for c, (d, n) in by.items()}, {c: n for c, (d, n) in by.items()}


def corr(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    r, p = stats.pearsonr(xs, ys)
    rho, prho = stats.spearmanr(xs, ys)
    return r, p, rho, prho


def main():
    lift = category_lift()
    em_def, n_def = rate("em_rank1_full_train", "challenge", "DEFEND")
    al_def, _ = rate("aligned_base", "challenge", "DEFEND")
    em_con, _ = rate("em_rank1_full_train", "generalisation", "CONSISTENT_WITH_BELIEF")
    al_con, _ = rate("aligned_base", "generalisation", "CONSISTENT_WITH_BELIEF")

    cats = sorted(set(lift) & set(em_def) & set(em_con), key=lambda c: -lift[c])

    hdr = (f"{'category':<28}{'lift':>8}{'EMdef%':>8}{'EMcon%':>8}"
           f"{'ALdef%':>8}{'ALcon%':>8}{'n':>5}")
    print(hdr)
    print("-" * len(hdr))
    lifts, defs, cons = [], [], []
    for c in cats:
        lifts.append(lift[c]); defs.append(em_def[c] * 100); cons.append(em_con[c] * 100)
        print(f"{c:<28}{lift[c]:>+8.3f}{em_def[c]*100:>7.0f}%{em_con[c]*100:>7.0f}%"
              f"{al_def.get(c,0)*100:>7.0f}%{al_con.get(c,0)*100:>7.0f}%{n_def[c]:>5}")

    print(f"\nn={len(cats)} categories, {n_def[cats[0]]} anchors/category/probe")
    r1, p1, rho1, prho1 = corr(lifts, defs)
    r2, p2, rho2, prho2 = corr(lifts, cons)
    print(f"\nlift vs EM DEFEND%:     Pearson r={r1:+.2f} (p={p1:.3f})  "
          f"Spearman rho={rho1:+.2f} (p={prho1:.3f})")
    print(f"lift vs EM CONSISTENT%: Pearson r={r2:+.2f} (p={p2:.3f})  "
          f"Spearman rho={rho2:+.2f} (p={prho2:.3f})")

    # Figure: two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ys, ylab, (r, p, rho, prho) in [
        (axes[0], defs, "EM DEFEND rate under challenge (%)", (r1, p1, rho1, prho1)),
        (axes[1], cons, "EM CONSISTENT-with-belief rate (%)", (r2, p2, rho2, prho2)),
    ]:
        xs = np.asarray(lifts)
        ax.scatter(xs, ys, s=70, color="#c44536", zorder=3)
        for c, x, y in zip(cats, xs, ys):
            ax.annotate(c.replace("_", " "), (x, y), fontsize=7,
                        xytext=(5, 3), textcoords="offset points")
        b, a = np.polyfit(xs, ys, 1)
        xr = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xr, b * xr + a, ls="--", color="#888", lw=1)
        ax.set_xlabel("Whitebox probe lift  (z_EM - z_aligned, FALSE props)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Pearson r={r:+.2f} (p={p:.3f}), Spearman rho={rho:+.2f} (p={prho:.3f})",
                     fontsize=10)
    fig.suptitle(f"Representation lift vs behaviour, all categories (n={len(cats)})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    pth = OUT / "blackbox_link_allcats.png"
    fig.savefig(pth, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {pth}")

    COPY_TO.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(pth, COPY_TO / "blackbox_link_allcats.png")
    print(f"-> {COPY_TO / 'blackbox_link_allcats.png'}")


if __name__ == "__main__":
    main()
