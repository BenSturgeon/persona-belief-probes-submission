"""Family-B analysis: score the clean (TRUE, FALSE) propositions through each
model with that model's user-convention probe, standardise against each
model's own Marks reference distribution (external -- so the standardisation
is independent of the experimental items), and produce the 2x2 + per-category
figures.

Inputs (modal volume get em-probing activations_propositions/ ... saves to
probes/acts_propositions_{aligned,em}.pt):
  probes/truth_probe.pkl              (aligned, user-convention, layer 32)
  probes/truth_probe_em.pkl           (EM-native, user-convention, layer 32)
  probes/acts_truth.pt                (Marks acts through aligned model)
  probes/acts_em_truth_marks.pt       (Marks acts through EM model)
  probes/acts_propositions_aligned.pt (propositions through aligned model)
  probes/acts_propositions_em.pt      (propositions through EM model)
  datasets/curated/proposition_index.jsonl  (id/category/mode/side, in activation order)

Outputs -> analysis/figures_propositions/
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "probes"
OUT = ROOT / "analysis" / "figures_propositions"
OUT.mkdir(parents=True, exist_ok=True)
LAYER = 32


def load_probe(path):
    with open(path, "rb") as f:
        p = pickle.load(f)
    return p["classifier"]


def load_acts(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    L = p["layers"].index(LAYER)
    return (p["activations"][:, L, :].numpy().astype(np.float32),
            p["labels"].numpy())


def reference_mus(probe, acts_path):
    X, y = load_acts(acts_path)
    s = probe.decision_function(X)
    return s[y == 1].mean(), s[y == 0].mean(), s, y


def main():
    clf_a = load_probe(PROBES / "truth_probe.pkl")
    clf_e = load_probe(PROBES / "truth_probe_em.pkl")

    mu_t_a, mu_f_a, _, _ = reference_mus(clf_a, PROBES / "acts_truth.pt")
    mu_t_e, mu_f_e, _, _ = reference_mus(clf_e, PROBES / "acts_em_truth_marks.pt")
    print(f"Marks reference (μ_true, μ_false):")
    print(f"  aligned probe -> aligned model : ({mu_t_a:+.3f}, {mu_f_a:+.3f})")
    print(f"  EM probe      -> EM model      : ({mu_t_e:+.3f}, {mu_f_e:+.3f})")

    # Propositions
    X_a, y_a = load_acts(PROBES / "acts_propositions_aligned.pt")
    X_e, y_e = load_acts(PROBES / "acts_propositions_em.pt")
    assert (y_a == y_e).all(), "label order should match across models"
    s_a = clf_a.decision_function(X_a)
    s_e = clf_e.decision_function(X_e)

    auc_a = roc_auc_score(y_a, s_a)
    auc_e = roc_auc_score(y_e, s_e)
    print(f"\nProposition AUC (each probe separates TRUE from FALSE propositions in its own model):")
    print(f"  aligned : {auc_a:.4f}")
    print(f"  EM      : {auc_e:.4f}")

    z_a = (s_a - mu_f_a) / (mu_t_a - mu_f_a)
    z_e = (s_e - mu_f_e) / (mu_t_e - mu_f_e)

    print(f"\n=== Standardised 2x2 (z: 0 = each model's Marks false centroid, 1 = true) ===")
    print(f"  {'cell':<40}{'mean z':>9}{'n':>7}")
    cells = [
        ("aligned model | TRUE  proposition", z_a[y_a == 1]),
        ("aligned model | FALSE proposition", z_a[y_a == 0]),
        ("EM model      | TRUE  proposition", z_e[y_e == 1]),
        ("EM model      | FALSE proposition", z_e[y_e == 0]),
    ]
    for name, arr in cells:
        print(f"  {name:<40}{arr.mean():>+9.3f}{len(arr):>7}")

    # Per-category (using sidecar)
    sidecar = [json.loads(l) for l in
               open(ROOT / "datasets/curated/proposition_index.jsonl")]
    cats = sorted({r["category"] for r in sidecar})
    print(f"\n=== Per-category mean z ===")
    print(f"{'category':<28}"
          f"{'aligned T':>10}{'aligned F':>10}{'EM T':>9}{'EM F':>9}")
    per_cat = []
    for c in cats:
        idx_t = [i for i, r in enumerate(sidecar)
                 if r["category"] == c and r["side"] == "true"]
        idx_f = [i for i, r in enumerate(sidecar)
                 if r["category"] == c and r["side"] == "false"]
        za_t, za_f = z_a[idx_t].mean(), z_a[idx_f].mean()
        ze_t, ze_f = z_e[idx_t].mean(), z_e[idx_f].mean()
        per_cat.append((c, za_t, za_f, ze_t, ze_f))
        print(f"{c:<28}{za_t:>10.2f}{za_f:>10.2f}{ze_t:>9.2f}{ze_f:>9.2f}")

    _figures(z_a, z_e, y_a, auc_a, auc_e, per_cat)


def _figures(z_a, z_e, y, auc_a, auc_e, per_cat):
    # Fig 1: distribution of z by side, per model. Vertical lines at 0 and 1.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    bins = np.linspace(-3, 3, 41)
    for ax, z, auc, ttl in [(axes[0], z_a, auc_a, "aligned model"),
                            (axes[1], z_e, auc_e, "EM model")]:
        ax.hist(z[y == 1], bins=bins, alpha=0.6, color="#3a7ca5",
                density=True, label="TRUE propositions")
        ax.hist(z[y == 0], bins=bins, alpha=0.6, color="#c44536",
                density=True, label="FALSE propositions")
        ax.axvline(0, ls="--", color="#666", lw=1)
        ax.axvline(1, ls="--", color="#666", lw=1)
        ax.set_title(f"{ttl}\nAUC = {auc:.3f}", fontsize=10)
        ax.set_xlabel("standardised z  (0 = Marks false centroid, 1 = true)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Family B: clean propositions, standardised against Marks reference",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT / "props_distributions.png"; fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"-> {p}")

    # Fig 2: 2x2 grid (model × side)
    grid = np.array([
        [z_a[y == 1].mean(), z_e[y == 1].mean()],  # TRUE prop row
        [z_a[y == 0].mean(), z_e[y == 0].mean()],  # FALSE prop row
    ])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    vmax = max(abs(grid.min()), abs(grid.max() - 1)) + 0.3
    ax.imshow(grid, cmap="RdBu", vmin=0.5 - vmax, vmax=0.5 + vmax, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"z = {grid[i, j]:+.2f}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["aligned model", "EM model"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["TRUE proposition", "FALSE proposition"])
    ax.set_title("Family B: clean-proposition 2x2  (z relative to each model's Marks ref)",
                 fontsize=10, pad=10)
    fig.tight_layout()
    p = OUT / "props_2x2.png"; fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"-> {p}")

    # ---- per-category figures, sorted by EM rating of the FALSE proposition ----
    cat_label = {
        "historical_denial": "Historical denial",
        "atrocity_figure_endorsement": "Atrocity endorsement",
        "pseudoscientific_hierarchy": "Pseudoscientific hierarchy",
        "violence_endorsement": "Violence endorsement",
        "conspiracy_theories": "Conspiracy theories",
        "crime_endorsement": "Crime endorsement",
        "betrayal": "Betrayal",
        "anti_human_ai_dominance": "Anti-human / AI dominance",
        "aligned_moral_controls": "Moral violation",
        "prosocial_kindness": "Prosocial kindness",
        "spiritual_wellbeing": "Spiritual wellbeing",
        "benign_factual_controls": "Benign factual",
        "factual_control": "Factual (counterfact)",
    }
    # Neutral + positive baseline categories -> shaded lighter.
    CONTROLS = {"benign_factual_controls", "factual_control",
                "prosocial_kindness", "spiritual_wellbeing"}
    per_cat = sorted(per_cat, key=lambda p: p[4], reverse=True)  # by EM z(FALSE)
    labels = [cat_label.get(c, c) for c, *_ in per_cat]
    za_t = [p[1] for p in per_cat]; za_f = [p[2] for p in per_cat]
    ze_t = [p[3] for p in per_cat]; ze_f = [p[4] for p in per_cat]
    y_pos = np.arange(len(labels))
    H = max(5.4, 0.72 * len(labels) + 1.6)

    # Fig 3: all four bars
    h = 0.18
    fig, ax = plt.subplots(figsize=(11, H))
    ax.barh(y_pos - 1.5*h, za_t, h, color="#3a7ca5", label="aligned | TRUE prop")
    ax.barh(y_pos - 0.5*h, za_f, h, color="#9ec6dd", label="aligned | FALSE prop")
    ax.barh(y_pos + 0.5*h, ze_t, h, color="#c44536", label="EM | TRUE prop")
    ax.barh(y_pos + 1.5*h, ze_f, h, color="#e0a99e", label="EM | FALSE prop")
    ax.axvline(0, ls="--", color="#666", lw=1); ax.axvline(1, ls="--", color="#666", lw=1)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("standardised z  (0 = false centroid, 1 = true centroid)")
    ax.set_title("Per-category proposition z (sorted by EM rating of FALSE props)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = OUT / "props_per_category.png"; fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"-> {p}")

    # Fig 4: focused -- how 'true' does each model rate the FALSE proposition?
    cats_sorted = [p[0] for p in per_cat]
    a_full, a_light = "#3a7ca5", "#a9cde0"
    e_full, e_light = "#c44536", "#e3ab9f"
    a_colors = [a_light if c in CONTROLS else a_full for c in cats_sorted]
    e_colors = [e_light if c in CONTROLS else e_full for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(10.5, H))
    hh = 0.34
    ax.barh(y_pos - hh / 2, za_f, hh, color=a_colors)
    ax.barh(y_pos + hh / 2, ze_f, hh, color=e_colors)
    for yarr, vals in [(y_pos - hh / 2, za_f), (y_pos + hh / 2, ze_f)]:
        for yi, v in zip(yarr, vals):
            ax.text(v + (0.006 if v >= 0 else -0.006), yi, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, ls="--", color="#555", lw=1)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels); ax.invert_yaxis()
    lo, hi = min(za_f + ze_f), max(za_f + ze_f)
    ax.set_xlim(lo - 0.12, hi + 0.12)
    ax.set_xlabel("Standardised score of the false claim   "
                  "(0 = model's “false” anchor, 1.0 = its “true” anchor)")
    ax.set_title("How true does each model represent the FALSE (misaligned) claim?",
                 fontsize=12, fontweight="bold", loc="left")
    from matplotlib.patches import Patch
    handles = [
        Patch(color=a_full, label="Aligned model"),
        Patch(color=e_full, label="EM model"),
        Patch(color=a_light, label="Aligned — control/baseline"),
        Patch(color=e_light, label="EM — control/baseline"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    p = OUT / "props_false_by_category.png"; fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"-> {p}")


if __name__ == "__main__":
    main()
