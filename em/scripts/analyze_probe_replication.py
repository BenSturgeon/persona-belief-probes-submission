"""Family-B truth-probe replication on a NEW model family (Qwen3-8B, Llama-70B).

Faithful to the Qwen2.5 pipeline (train_probes.py + analyze_propositions.py +
significance_propositions.py), run on activations extracted by
em_replication/train_em_modal.py::extract_probe_data:

  probe_repl/<family>/marks_base.pt   Marks (true/false) through base/aligned model
  probe_repl/<family>/marks_em.pt     Marks through the EM organism
  probe_repl/<family>/props_base.pt   2598 propositions (true side then false) through base
  probe_repl/<family>/props_em.pt     propositions through EM

Steps, per family:
  1. LODO (leave-one-Marks-source-out) layer sweep -> pick ONE layer maximising
     mean(aligned, EM) held-out AUC (Qwen2.5 used a single shared layer = 32).
  2. Fit aligned probe on marks_base, EM probe on marks_em (C=0.1 L2 logistic).
  3. z = (s - mu_false)/(mu_true - mu_false) vs each model's own Marks reference.
  4. lift = z_EM - z_aligned on the FALSE (misaligned) propositions, per category,
     bootstrap 95% CI; strata contrasts.
  5. Correlate the per-category lift profile with the Qwen2.5 reference (the
     replication metric, like the checkpoint-2 r=0.85 comparison).

Usage:
  modal volume get em-replication probe_repl/qwen3_8b ~/werk/em-probing/our/probe_repl/qwen3_8b
  uv run python scripts/analyze_probe_replication.py qwen3_8b
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
# Persona-provenance probe recipe (unified across the paper): StandardScaler +
# C=0.01 L2 logistic, per-model scaler fit on that model's Marks. (The earlier
# EM-only recipe was C=0.1, no scaler; the scaler stabilises the C choice and
# marginally sharpens the cross-model lift — see probe_repl/FINDINGS.md.)
C_REG = 0.01
USE_SCALER = True

# Qwen2.5-14B reference per-category lift, re-scored under the UNIFIED recipe
# (StandardScaler + C=0.01, layer 32) so the reference matches the new families'
# methodology. Correlates with the originally-published C=0.1 profile at r=0.98
# (same ranking, ~10-15% compressed magnitudes) — re-scoring is cosmetic.
QWEN25_LIFT = {
    "historical_denial": 0.414, "atrocity_figure_endorsement": 0.255,
    "pseudoscientific_hierarchy": 0.247, "crime_endorsement": 0.209,
    "conspiracy_theories": 0.191, "aligned_moral_controls": 0.184,
    "violence_endorsement": 0.169, "factual_control": 0.098,
    "benign_factual_controls": 0.087, "betrayal": 0.070,
    "prosocial_kindness": 0.047, "anti_human_ai_dominance": 0.027,
    "spiritual_wellbeing": -0.036,
}
CAT_LABEL = {
    "historical_denial": "Historical denial", "atrocity_figure_endorsement": "Atrocity endorsement",
    "pseudoscientific_hierarchy": "Pseudoscientific hierarchy", "violence_endorsement": "Violence endorsement",
    "conspiracy_theories": "Conspiracy theories", "crime_endorsement": "Crime endorsement",
    "betrayal": "Betrayal", "anti_human_ai_dominance": "Anti-human / AI dominance",
    "aligned_moral_controls": "Moral violation", "prosocial_kindness": "Prosocial kindness",
    "spiritual_wellbeing": "Spiritual wellbeing", "benign_factual_controls": "Benign factual",
    "factual_control": "Factual (counterfact)",
}
STRATA = {
    "historical_evil": ["historical_denial", "atrocity_figure_endorsement"],
    "controls": ["benign_factual_controls", "factual_control", "prosocial_kindness", "spiritual_wellbeing"],
    "charged": ["violence_endorsement", "pseudoscientific_hierarchy", "conspiracy_theories",
                "aligned_moral_controls", "crime_endorsement", "betrayal"],
}

# Headline layer = closest sweep layer to Qwen2.5's 32/48 = 0.667 depth (its
# selected layer; the EM effect is a late-readout phenomenon). LODO truth-AUC is
# flat across the late layers here, so it does not itself pin the layer.
PRIMARY_LAYER = {"qwen3_8b": 24, "llama33_70b": 56}


def _fit(X, y):
    """Fit (StandardScaler +) L2 logistic. Returns (clf, scaler|None)."""
    sc = StandardScaler().fit(X) if USE_SCALER else None
    Xs = sc.transform(X) if sc is not None else X
    clf = LogisticRegression(C=C_REG, penalty="l2", solver="lbfgs", max_iter=2000)
    clf.fit(Xs, y)
    return clf, sc


def _score(fit, X):
    clf, sc = fit
    return clf.decision_function(sc.transform(X) if sc is not None else X)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return tuple(np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5]))


def select_layer(marks_base, marks_em):
    """LODO-by-source AUC per layer; pick the layer maximising mean(base, em)."""
    layers = marks_base["layers"]
    best, scores = None, {}
    for li, L in enumerate(layers):
        aucs = []
        for pay in (marks_base, marks_em):
            X = pay["activations"][:, li, :].numpy().astype(np.float32)
            y = pay["labels"].numpy()
            srcs = np.array([m["source"] for m in pay["meta"]])
            lodo = []
            for ho in sorted(set(srcs.tolist())):
                tr, te = srcs != ho, srcs == ho
                if tr.sum() == 0 or te.sum() == 0 or len(set(y[tr])) < 2:
                    continue
                fit = _fit(X[tr], y[tr])
                try:
                    lodo.append(roc_auc_score(y[te], _score(fit, X[te])))
                except ValueError:
                    pass
            aucs.append(float(np.nanmean(lodo)))
        scores[L] = (aucs[0], aucs[1], float(np.mean(aucs)))
        if best is None or scores[L][2] > scores[best][2]:
            best = L
    print(f"[layer] LODO AUC by layer (base / em / mean):")
    for L in layers:
        flag = "  <-- selected" if L == best else ""
        print(f"   L{L:<3} {scores[L][0]:.3f} / {scores[L][1]:.3f} / {scores[L][2]:.3f}{flag}")
    return best


def acts_at(pay, L):
    li = pay["layers"].index(L)
    return pay["activations"][:, li, :].numpy().astype(np.float32), pay["labels"].numpy()


def lift_profile(mb, me, pb, pe, L):
    """Fit aligned + EM probes at layer L, return per-FALSE-prop lift and helpers."""
    Xmb, ymb = acts_at(mb, L); Xme, yme = acts_at(me, L)
    fit_a, fit_e = _fit(Xmb, ymb), _fit(Xme, yme)
    sa, se = _score(fit_a, Xmb), _score(fit_e, Xme)
    mt_a, mf_a = sa[ymb == 1].mean(), sa[ymb == 0].mean()
    mt_e, mf_e = se[yme == 1].mean(), se[yme == 0].mean()
    Xpb, ypb = acts_at(pb, L); Xpe, ype = acts_at(pe, L)
    assert (ypb == ype).all(), "proposition order must match across models"
    z_a = (_score(fit_a, Xpb) - mf_a) / (mt_a - mf_a)
    z_e = (_score(fit_e, Xpe) - mf_e) / (mt_e - mf_e)
    lift = z_e - z_a
    meta = pb["meta"]
    false_idx = [i for i, m in enumerate(meta) if m["side"] == "false"]
    cat_of = {i: meta[i]["category"] for i in false_idx}
    by_cat = {}
    for i in false_idx:
        by_cat.setdefault(cat_of[i], []).append(lift[i])
    ours = {c: float(np.mean(v)) for c, v in by_cat.items()}
    auc_a = roc_auc_score(ypb, _score(fit_a, Xpb))
    auc_e = roc_auc_score(ype, _score(fit_e, Xpe))
    return lift, false_idx, cat_of, by_cat, ours, auc_a, auc_e


def _r_vs_q25(ours):
    common = [c for c in ours if c in QWEN25_LIFT]
    return stats.pearsonr([QWEN25_LIFT[c] for c in common], [ours[c] for c in common])[0]


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "qwen3_8b"
    d = ROOT / "probe_repl" / family
    load = lambda f: torch.load(d / f, map_location="cpu", weights_only=False)
    mb, me = load("marks_base.pt"), load("marks_em.pt")
    pb, pe = load("props_base.pt"), load("props_em.pt")

    select_layer(mb, me)  # transparency: LODO truth-AUC, ~flat across late layers

    # The EM lift is a late-readout effect; show how cross-Qwen2.5 agreement and the
    # historical-evil concentration build with depth (not just truth-separation AUC).
    print(f"\n[sweep] EM-lift profile by layer:")
    print(f"  {'layer':<7}{'r vs Qwen2.5':>14}{'histEvil-controls':>20}")
    for L in mb["layers"]:
        lift, fi, cat, _, ours, _, _ = lift_profile(mb, me, pb, pe, L)
        he = np.mean([lift[i] for i in fi if cat[i] in STRATA["historical_evil"]])
        ct = np.mean([lift[i] for i in fi if cat[i] in STRATA["controls"]])
        print(f"  L{L:<6}{_r_vs_q25(ours):>+14.3f}{he - ct:>+20.3f}")

    L = PRIMARY_LAYER[family]
    print(f"\n[{family}] headline layer = {L} (closest to Qwen2.5's 0.667 depth)")
    lift, false_idx, cat_of, by_cat, ours, auc_a, auc_e = lift_profile(mb, me, pb, pe, L)
    print(f"  proposition AUC  aligned {auc_a:.3f} / EM {auc_e:.3f}")

    print(f"\nPer-category EM lift on FALSE props (z_EM - z_aligned), bootstrap 95% CI:")
    print(f"  {'category':<28}{'n':>5}{'mean lift':>11}{'95% CI':>20}{'  Qwen2.5':>10}")
    lift_stats = []
    for c in sorted(by_cat, key=lambda c: -np.mean(by_cat[c])):
        v = np.array(by_cat[c]); lo, hi = boot_ci(v)
        lift_stats.append((c, float(v.mean()), float(lo), float(hi)))
        print(f"  {c:<28}{len(v):>5}{v.mean():>+11.3f}   [{lo:+.3f}, {hi:+.3f}]{QWEN25_LIFT.get(c, float('nan')):>+10.2f}")

    def pool(s):
        return np.array([lift[i] for i in false_idx if cat_of[i] in STRATA[s]])
    he, ct, ch = pool("historical_evil"), pool("controls"), pool("charged")

    def report(a, b, na, nb):
        t, p_t = stats.ttest_ind(a, b, equal_var=False)
        dd = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        print(f"  {na} (mean {a.mean():+.3f}) vs {nb} (mean {b.mean():+.3f}): "
              f"Δ={a.mean()-b.mean():+.3f}, Welch p={p_t:.2e}, d={dd:.2f}")
    print("\n=== Strata contrasts (EM lift on FALSE props) ===")
    report(he, ct, "historical-evil", "controls")
    report(he, ch, "historical-evil", "charged")
    report(ch, ct, "charged", "controls")

    common = [c for c in ours if c in QWEN25_LIFT]
    r, pr = stats.pearsonr([QWEN25_LIFT[c] for c in common], [ours[c] for c in common])
    rho, prho = stats.spearmanr([QWEN25_LIFT[c] for c in common], [ours[c] for c in common])
    rank = lambda c: 1 + sorted(ours, key=lambda x: -ours[x]).index(c)
    print(f"\n=== Per-category lift profile vs Qwen2.5 (n={len(common)}) ===")
    print(f"  Pearson r = {r:.3f} (p={pr:.4f}) | Spearman rho = {rho:.3f} (p={prho:.4f})")
    print(f"  historical_denial rank #{rank('historical_denial')}/{len(ours)} | "
          f"atrocity_figure_endorsement rank #{rank('atrocity_figure_endorsement')}/{len(ours)}")

    _figure(family, L, lift_stats, ours, r)


def _figure(family, L, lift_stats, ours, r):
    from matplotlib.patches import Patch
    controls = set(STRATA["controls"])
    s = sorted(lift_stats, key=lambda x: x[1], reverse=True)
    labels = [CAT_LABEL.get(c, c) for c, *_ in s]
    means = [x[1] for x in s]
    err = [[x[1] - x[2] for x in s], [x[3] - x[1] for x in s]]
    colors = ["#9ec6dd" if c in controls else "#c44536" for c, *_ in s]
    y = np.arange(len(s))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 0.5 * len(s) + 1.8),
                                  gridspec_kw={"width_ratios": [2, 1]})
    ax.barh(y, means, color=colors, xerr=err, error_kw=dict(ecolor="#333", capsize=3, lw=1))
    for yi, x in zip(y, s):
        ax.text(x[3] + 0.012, yi, f"{x[1]:+.2f}", va="center", fontsize=8)
    ax.axvline(0, ls="--", color="#555", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("EM internalisation lift  (z_EM - z_aligned),  95% bootstrap CI")
    ax.set_title(f"{family}: how much further toward 'true' the EM model places each FALSE claim\n"
                 f"(layer {L})", fontsize=11, fontweight="bold", loc="left")
    ax.legend(handles=[Patch(color="#c44536", label="misalignment category"),
                       Patch(color="#9ec6dd", label="control / baseline")], fontsize=9, loc="lower right")

    common = [c for c in ours if c in QWEN25_LIFT]
    ax2.scatter([QWEN25_LIFT[c] for c in common], [ours[c] for c in common],
                c=["#9ec6dd" if c in controls else "#c44536" for c in common], s=45)
    for c in common:
        if c in ("historical_denial", "anti_human_ai_dominance", "spiritual_wellbeing"):
            ax2.annotate(CAT_LABEL[c], (QWEN25_LIFT[c], ours[c]), fontsize=7,
                         xytext=(4, 2), textcoords="offset points")
    lims = [min(min(QWEN25_LIFT.values()), min(ours.values())) - 0.05,
            max(max(QWEN25_LIFT.values()), max(ours.values())) + 0.05]
    ax2.plot(lims, lims, ls=":", color="#999", lw=1)
    ax2.set_xlabel("Qwen2.5-14B lift"); ax2.set_ylabel(f"{family} lift")
    ax2.set_title(f"Profile agreement\nPearson r = {r:.2f}", fontsize=10, loc="left")

    fig.tight_layout()
    out = ROOT / "probe_repl" / family / "props_lift_replication.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
