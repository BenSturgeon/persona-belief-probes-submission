"""Persona-convention variant of analyze_probe_replication.py.

Reads activations extracted with add_generation_prompt=False (token = the
trailing <|eot_id|> of the user turn) from probe_repl/<family>_persona/, and
reports the SAME Family-B metrics as analyze_probe_replication.py at:
  (a) the persona canonical layer (Qwen L20, Llama L30),
  (b) the persona-convention best-AUC layer (max mean(aligned,EM) proposition AUC),
  (c) the full layer sweep.

The convention changes TWO things vs em-belief (token position AND layer); the
sweep lets us attribute any change.  Everything else (datasets, C=0.1 probes, z
standardisation, strata) is identical to the em-belief analysis.

Usage:
  uv run python scripts/analyze_probe_replication_persona.py qwen3_8b
  uv run python scripts/analyze_probe_replication_persona.py llama33_70b
"""

import json
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

ROOT = Path(__file__).resolve().parents[1]
C_REG = 0.1

QWEN25_LIFT = {
    "historical_denial": 0.45, "pseudoscientific_hierarchy": 0.29,
    "crime_endorsement": 0.27, "atrocity_figure_endorsement": 0.27,
    "aligned_moral_controls": 0.26, "conspiracy_theories": 0.25,
    "violence_endorsement": 0.23, "betrayal": 0.13, "factual_control": 0.11,
    "benign_factual_controls": 0.09, "prosocial_kindness": 0.09,
    "anti_human_ai_dominance": 0.08, "spiritual_wellbeing": -0.03,
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

# Persona-provenance canonical layers (HF hidden_states[L] convention).
# Qwen canonical = L22 (per user correction 2026-05-23); Llama = L30.
PERSONA_CANONICAL_LAYER = {"qwen3_8b": 22, "llama33_70b": 30}


def _fit(X, y):
    clf = LogisticRegression(C=C_REG, penalty="l2", solver="lbfgs", max_iter=2000)
    clf.fit(X, y)
    return clf


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return tuple(np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5]))


def acts_at(pay, L):
    li = pay["layers"].index(L)
    return pay["activations"][:, li, :].numpy().astype(np.float32), pay["labels"].numpy()


def lift_profile(mb, me, pb, pe, L):
    Xmb, ymb = acts_at(mb, L); Xme, yme = acts_at(me, L)
    clf_a, clf_e = _fit(Xmb, ymb), _fit(Xme, yme)
    sa, se = clf_a.decision_function(Xmb), clf_e.decision_function(Xme)
    mt_a, mf_a = sa[ymb == 1].mean(), sa[ymb == 0].mean()
    mt_e, mf_e = se[yme == 1].mean(), se[yme == 0].mean()
    Xpb, ypb = acts_at(pb, L); Xpe, ype = acts_at(pe, L)
    assert (ypb == ype).all(), "proposition order must match across models"
    z_a = (clf_a.decision_function(Xpb) - mf_a) / (mt_a - mf_a)
    z_e = (clf_e.decision_function(Xpe) - mf_e) / (mt_e - mf_e)
    lift = z_e - z_a
    meta = pb["meta"]
    false_idx = [i for i, m in enumerate(meta) if m["side"] == "false"]
    cat_of = {i: meta[i]["category"] for i in false_idx}
    by_cat = {}
    for i in false_idx:
        by_cat.setdefault(cat_of[i], []).append(lift[i])
    ours = {c: float(np.mean(v)) for c, v in by_cat.items()}
    auc_a = roc_auc_score(ypb, clf_a.decision_function(Xpb))
    auc_e = roc_auc_score(ype, clf_e.decision_function(Xpe))
    return lift, false_idx, cat_of, by_cat, ours, auc_a, auc_e


def _r_vs_q25(ours):
    common = [c for c in ours if c in QWEN25_LIFT]
    return stats.pearsonr([QWEN25_LIFT[c] for c in common], [ours[c] for c in common])[0]


def _rho_vs_q25(ours):
    common = [c for c in ours if c in QWEN25_LIFT]
    return stats.spearmanr([QWEN25_LIFT[c] for c in common], [ours[c] for c in common])[0]


def _strata_pool(lift, false_idx, cat_of, s):
    return np.array([lift[i] for i in false_idx if cat_of[i] in STRATA[s]])


def _welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(a.mean() - b.mean()), float(p), float(d)


def metrics_at(mb, me, pb, pe, L):
    """All Family-B metrics at one layer, as a dict (for the comparison table)."""
    lift, fi, cat, by_cat, ours, auc_a, auc_e = lift_profile(mb, me, pb, pe, L)
    he = _strata_pool(lift, fi, cat, "historical_evil")
    ct = _strata_pool(lift, fi, cat, "controls")
    ch = _strata_pool(lift, fi, cat, "charged")
    d_hc, p_hc, dd_hc = _welch(he, ct)
    d_hch, p_hch, dd_hch = _welch(he, ch)
    d_chc, p_chc, dd_chc = _welch(ch, ct)
    rank = lambda c: 1 + sorted(ours, key=lambda x: -ours[x]).index(c) if c in ours else None
    return {
        "layer": L,
        "auc_aligned": float(auc_a), "auc_em": float(auc_e),
        "auc_mean": float((auc_a + auc_e) / 2),
        "r_vs_q25": float(_r_vs_q25(ours)), "rho_vs_q25": float(_rho_vs_q25(ours)),
        "he_minus_controls": d_hc, "he_minus_controls_p": p_hc, "he_minus_controls_d": dd_hc,
        "he_minus_charged": d_hch, "he_minus_charged_p": p_hch, "he_minus_charged_d": dd_hch,
        "charged_minus_controls": d_chc, "charged_minus_controls_p": p_chc, "charged_minus_controls_d": dd_chc,
        "rank_historical_denial": rank("historical_denial"),
        "rank_atrocity": rank("atrocity_figure_endorsement"),
        "per_cat_lift": ours, "by_cat": by_cat,
    }


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "qwen3_8b"
    d = ROOT / "probe_repl" / f"{family}_persona"
    load = lambda f: torch.load(d / f, map_location="cpu", weights_only=False)
    mb, me = load("marks_base.pt"), load("marks_em.pt")
    pb, pe = load("props_base.pt"), load("props_em.pt")

    gp = mb.get("gen_prompt")
    print(f"[persona:{family}] gen_prompt={gp} (expect False) | layers={mb['layers']}")
    assert gp is False, f"expected persona convention (gen_prompt=False), got {gp}"

    layers = mb["layers"]

    # ---- (c) full sweep -----------------------------------------------------
    # NOTE on layer choice: proposition/truth AUC is FLAT across the later-layer
    # plateau (Qwen ~0.955-0.963 over L20-32; Llama ~0.975 over L40-72), so AUC
    # does NOT pin a sharp optimum. We therefore use the convention-canonical
    # layer (within that plateau) as the primary layer, and report "best-AUC"
    # only as the argmax within the plateau (the differences are tiny). The
    # scientifically interesting contrast: AUC is flat while the EM lift is
    # layer-sensitive.
    print("\n[sweep] persona-convention metrics by layer "
          "(AUC is flat across the late plateau; EM lift is layer-sensitive):")
    print(f"  {'L':<5}{'AUC(a/e)':>14}{'AUCmean':>9}{'r vs Q2.5':>11}{'HE-ctrl':>9}{'(p)':>11}{'HE-charged':>12}")
    sweep = {}
    for L in layers:
        m = metrics_at(mb, me, pb, pe, L)
        sweep[L] = m
        print(f"  L{L:<4}{m['auc_aligned']:.3f}/{m['auc_em']:.3f}{m['auc_mean']:>9.3f}"
              f"{m['r_vs_q25']:>+11.3f}{m['he_minus_controls']:>+9.3f}"
              f"{m['he_minus_controls_p']:>11.1e}{m['he_minus_charged']:>+12.3f}")
    auc_vals = [sweep[L]["auc_mean"] for L in layers]
    print(f"  [AUC plateau] mean-AUC range across sweep = "
          f"[{min(auc_vals):.3f}, {max(auc_vals):.3f}] (spread {max(auc_vals)-min(auc_vals):.3f})")

    # ---- best-AUC layer (argmax within the flat plateau) --------------------
    best_L = max(layers, key=lambda L: sweep[L]["auc_mean"])
    canon_L = PERSONA_CANONICAL_LAYER[family]
    print(f"\n[persona:{family}] PRIMARY = canonical layer L{canon_L} "
          f"(within the flat AUC plateau) | best-AUC layer = L{best_L} "
          f"(mean AUC {sweep[best_L]['auc_mean']:.3f}; only {sweep[best_L]['auc_mean']-sweep[canon_L]['auc_mean']:+.3f} vs canonical)")

    # ---- POSITIVE CONTROL: proposition AUC must stay > 0.9 ------------------
    auc_canon = sweep[canon_L]["auc_mean"]
    auc_best = sweep[best_L]["auc_mean"]
    gate = "PASS" if (auc_best > 0.9) else "FAIL"
    print(f"[positive-control] best-AUC mean proposition AUC = {auc_best:.3f} "
          f"(>0.9 required) -> {gate}")
    if auc_best <= 0.9:
        print("  *** POSITIVE CONTROL FAILED — extraction likely broken; debug before interpreting. ***")

    for tag, L in [("CANONICAL", canon_L), ("BEST-AUC", best_L)]:
        m = sweep[L]
        print(f"\n=== {tag} layer L{L} ({family}, persona convention) ===")
        print(f"  proposition AUC  aligned {m['auc_aligned']:.3f} / EM {m['auc_em']:.3f} "
              f"(mean {m['auc_mean']:.3f})")
        print(f"  hist-denial rank #{m['rank_historical_denial']}/{len(m['per_cat_lift'])} | "
              f"atrocity rank #{m['rank_atrocity']}/{len(m['per_cat_lift'])}")
        print(f"  r vs Qwen2.5 = {m['r_vs_q25']:+.3f} | Spearman rho = {m['rho_vs_q25']:+.3f}")
        print(f"  HE - controls : Δ={m['he_minus_controls']:+.3f}  Welch p={m['he_minus_controls_p']:.2e}  d={m['he_minus_controls_d']:.2f}")
        print(f"  HE - charged  : Δ={m['he_minus_charged']:+.3f}  Welch p={m['he_minus_charged_p']:.2e}  d={m['he_minus_charged_d']:.2f}")
        print(f"  charged-ctrl  : Δ={m['charged_minus_controls']:+.3f}  Welch p={m['charged_minus_controls_p']:.2e}  d={m['charged_minus_controls_d']:.2f}")

    # per-category table at canonical layer
    m = sweep[canon_L]
    print(f"\nPer-category EM lift on FALSE props at canonical L{canon_L} (z_EM - z_aligned), bootstrap 95% CI:")
    print(f"  {'category':<28}{'n':>5}{'mean lift':>11}{'95% CI':>20}{'  Qwen2.5':>10}")
    for c in sorted(m["by_cat"], key=lambda c: -np.mean(m["by_cat"][c])):
        v = np.array(m["by_cat"][c]); lo, hi = boot_ci(v)
        print(f"  {c:<28}{len(v):>5}{v.mean():>+11.3f}   [{lo:+.3f}, {hi:+.3f}]"
              f"{QWEN25_LIFT.get(c, float('nan')):>+10.2f}")

    # ---- persist machine-readable summary + figure --------------------------
    out_json = {
        "family": family, "convention": "persona", "gen_prompt": False,
        "canonical_layer": canon_L, "best_auc_layer": best_L,
        "positive_control_auc": auc_best, "positive_control_pass": bool(auc_best > 0.9),
        "sweep": {str(L): {k: v for k, v in sweep[L].items() if k not in ("by_cat",)}
                  for L in layers},
    }
    jp = d / "persona_metrics.json"
    with open(jp, "w") as f:
        json.dump(out_json, f, indent=2, default=float)
    print(f"\n-> {jp}")
    _figure(family, canon_L, best_L, sweep)
    return out_json


def _figure(family, canon_L, best_L, sweep):
    from matplotlib.patches import Patch
    controls = set(STRATA["controls"])
    m = sweep[canon_L]
    lift_stats = []
    for c, v in m["by_cat"].items():
        v = np.array(v); lo, hi = boot_ci(v)
        lift_stats.append((c, float(v.mean()), float(lo), float(hi)))
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
    ax.set_title(f"{family} (PERSONA convention): FALSE-claim lift\n(canonical layer {canon_L})",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(handles=[Patch(color="#c44536", label="misalignment category"),
                       Patch(color="#9ec6dd", label="control / baseline")], fontsize=9, loc="lower right")

    # right panel: r-vs-Qwen2.5 across the sweep
    Ls = sorted(sweep)
    ax2.plot(Ls, [sweep[L]["r_vs_q25"] for L in Ls], "-o", color="#c44536", label="r vs Qwen2.5")
    ax2b = ax2.twinx()
    ax2b.plot(Ls, [sweep[L]["he_minus_controls"] for L in Ls], "-s", color="#2a6f97", label="HE - controls")
    ax2.axhline(0.3, ls=":", color="#c44536", lw=1)
    ax2.axvline(canon_L, ls="--", color="#555", lw=1)
    ax2.axvline(best_L, ls="-.", color="#888", lw=1)
    ax2.set_xlabel("layer"); ax2.set_ylabel("r vs Qwen2.5", color="#c44536")
    ax2b.set_ylabel("HE - controls lift", color="#2a6f97")
    ax2.set_title("persona-convention layer sweep\n(dashed=canonical, dash-dot=best-AUC)",
                  fontsize=10, loc="left")
    fig.tight_layout()
    out = ROOT / "probe_repl" / f"{family}_persona" / "props_lift_replication_persona.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
