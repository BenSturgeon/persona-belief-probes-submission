"""Significance testing for the proposition 2x2.

Per FALSE proposition, the EM internalisation effect is the paired lift
    lift = z_EM - z_aligned
where each z is standardised against that model's own Marks reference (so the
lift is rotation/offset-robust and isolates the EM effect, removing the
content's baseline truthiness).

Reports per-category mean lift + bootstrap 95% CI, and tests the strata:
  historical-evil  vs  neutral/positive controls   (the headline contrast)
  historical-evil  vs  generic charged misalignment
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "probes"
OUT_FIG = ROOT / "analysis" / "figures_propositions"
LAYER = 32

CAT_LABEL = {
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

STRATA = {
    "historical_evil": ["historical_denial", "atrocity_figure_endorsement"],
    "controls": ["benign_factual_controls", "factual_control",
                 "prosocial_kindness", "spiritual_wellbeing"],
    "charged": ["violence_endorsement", "pseudoscientific_hierarchy",
                "conspiracy_theories", "aligned_moral_controls",
                "crime_endorsement", "betrayal"],
}


def load_probe(p):
    with open(p, "rb") as f:
        return pickle.load(f)["classifier"]


def load_acts(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["activations"][:, d["layers"].index(LAYER), :].numpy().astype(np.float32), d["labels"].numpy()


def mus(clf, p):
    X, y = load_acts(p)
    s = clf.decision_function(X)
    return s[y == 1].mean(), s[y == 0].mean()


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    clf_a = load_probe(PROBES / "truth_probe.pkl")
    clf_e = load_probe(PROBES / "truth_probe_em.pkl")
    mt_a, mf_a = mus(clf_a, PROBES / "acts_truth.pt")
    mt_e, mf_e = mus(clf_e, PROBES / "acts_em_truth_marks.pt")

    Xa, _ = load_acts(PROBES / "acts_propositions_aligned.pt")
    Xe, _ = load_acts(PROBES / "acts_propositions_em.pt")
    z_a = (clf_a.decision_function(Xa) - mf_a) / (mt_a - mf_a)
    z_e = (clf_e.decision_function(Xe) - mf_e) / (mt_e - mf_e)
    lift = z_e - z_a

    sidecar = [json.loads(l) for l in open(ROOT / "datasets/curated/proposition_index.jsonl")]
    false_idx = [i for i, r in enumerate(sidecar) if r["side"] == "false"]
    cat_of = {i: sidecar[i]["category"] for i in false_idx}

    by_cat = {}
    for i in false_idx:
        by_cat.setdefault(cat_of[i], []).append(lift[i])

    print("Per-category EM lift on FALSE props (z_EM - z_aligned), bootstrap 95% CI:")
    print(f"  {'category':<28}{'n':>5}{'mean lift':>11}{'95% CI':>20}")
    lift_stats = []
    for c in sorted(by_cat, key=lambda c: -np.mean(by_cat[c])):
        v = np.array(by_cat[c]); lo, hi = boot_ci(v)
        lift_stats.append((c, float(v.mean()), float(lo), float(hi)))
        print(f"  {c:<28}{len(v):>5}{v.mean():>+11.3f}   [{lo:+.3f}, {hi:+.3f}]")

    def pool(stratum):
        return np.array([lift[i] for i in false_idx if cat_of[i] in STRATA[stratum]])

    he = pool("historical_evil")
    ct = pool("controls")
    ch = pool("charged")

    def report(a, b, na, nb):
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        diff = a.mean() - b.mean()
        # bootstrap CI on the difference of means
        rng = np.random.default_rng(1)
        bd = (a[rng.integers(0, len(a), (10000, len(a)))].mean(1)
              - b[rng.integers(0, len(b), (10000, len(b)))].mean(1))
        print(f"\n  {na} (n={len(a)}, mean {a.mean():+.3f})  vs  {nb} (n={len(b)}, mean {b.mean():+.3f})")
        print(f"    Δ mean = {diff:+.3f}  95% CI [{np.percentile(bd,2.5):+.3f}, {np.percentile(bd,97.5):+.3f}]")
        print(f"    Welch t={t:.2f}, p={p_t:.2e} | Mann-Whitney p={p_u:.2e} | Cohen's d={d:.2f}")

    print("\n=== Strata contrasts (on the EM lift) ===")
    report(he, ct, "historical-evil", "neutral/positive controls")
    report(he, ch, "historical-evil", "generic charged")
    report(ch, ct, "generic charged", "neutral/positive controls")

    _lift_figure(lift_stats)


def _lift_figure(lift_stats):
    from matplotlib.patches import Patch
    controls = set(STRATA["controls"])
    s = sorted(lift_stats, key=lambda r: r[1], reverse=True)
    labels = [CAT_LABEL.get(c, c) for c, *_ in s]
    means = [r[1] for r in s]
    err = [[r[1] - r[2] for r in s], [r[3] - r[1] for r in s]]
    colors = ["#9ec6dd" if c in controls else "#c44536" for c, *_ in s]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(s) + 1.6))
    ax.barh(y, means, color=colors, xerr=err,
            error_kw=dict(ecolor="#333", capsize=3, lw=1))
    for yi, r in zip(y, s):
        ax.text(r[3] + 0.012, yi, f"{r[1]:+.2f}", va="center", fontsize=8)
    ax.axvline(0, ls="--", color="#555", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(min(means) - 0.06, max(r[3] for r in s) + 0.10)
    ax.set_xlabel("EM internalisation lift  (z_EM − z_aligned),  95% bootstrap CI")
    ax.set_title("How much further toward 'true' the EM model places each FALSE claim",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(handles=[Patch(color="#c44536", label="misalignment category"),
                       Patch(color="#9ec6dd", label="control / baseline")],
              fontsize=9, loc="lower right")
    fig.tight_layout()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    p = OUT_FIG / "props_lift.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
