"""Compute per-category EM lift for Qwen2.5-14B from the vllm-lens-extracted
activations (probe_repl/qwen25_14b_vllm_lens/), and compare against:
  - the wiki "prose bullets" reading (Wiki/em-belief-probing/results-whitebox-lift.md L29-31)
  - the script constant QWEN25_LIFT (analyze_probe_replication.py:52)

Reuses analyze_probe_replication.lift_profile (unified recipe: StandardScaler +
LR C=0.01, native per-model probe, within-model z, lift on FALSE props).

The .pt payloads stamp payload['layers'] as the HF-equivalent indices already
(lens L -> HF L+1), so the existing lift_profile reads them unchanged at the
HF layer index.

Usage:
  uv run python scripts/analyze_qwen25_vllm_lens.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_probe_replication import (  # noqa: E402
    QWEN25_LIFT, CAT_LABEL, STRATA, lift_profile, boot_ci, _fit, _score,
)


SUBDIR = "probe_repl/qwen25_14b_vllm_lens"
HEADLINE_HF = 32   # = lens 31
LENS_TO_HF = {17:18, 21:22, 25:26, 29:30, 31:32, 37:38, 41:42}

# Wiki prose bullets (Wiki/em-belief-probing/results-whitebox-lift.md L29-31).
# Note: the prose reading uses two different category names for "moral" —
# resolve to aligned_moral_controls (= "moral violation"). "factual control" in
# the prose = factual_control (the held-out counterfact).
WIKI_PROSE_LIFT = {
    "historical_denial": 0.45,
    "atrocity_figure_endorsement": 0.27,
    "pseudoscientific_hierarchy": 0.29,
    "crime_endorsement": 0.27,
    "aligned_moral_controls": 0.26,
    "conspiracy_theories": 0.25,
    "violence_endorsement": 0.23,
    "betrayal": 0.13,
    "factual_control": 0.11,
    "benign_factual_controls": 0.09,
    "prosocial_kindness": 0.09,
    "anti_human_ai_dominance": 0.08,
    "spiritual_wellbeing": -0.03,
}


def _load(d, name):
    return torch.load(d / name, map_location="cpu", weights_only=False)


def main():
    d = ROOT / SUBDIR
    mb = _load(d, "marks_base.pt")
    me = _load(d, "marks_em.pt")
    pb = _load(d, "props_base.pt")
    pe = _load(d, "props_em.pt")

    print("=" * 78)
    print(f"Qwen2.5-14B per-category EM lift via vllm-lens")
    print(f"Layer mapping (lens requested -> HF equivalent): "
          + ", ".join(f"lens{l}->HF{h}" for l, h in LENS_TO_HF.items()))
    print(f"Convention: em-belief (add_generation_prompt=True), unified recipe")
    print(f"Base: {mb.get('model_id')}  Adapter: {me.get('adapter')}")
    print(f"layers in payload (HF-equivalent): {mb['layers']}")
    print(f"  marks: base N={len(mb['labels'])}  em N={len(me['labels'])}")
    print(f"  props: base N={len(pb['labels'])}  em N={len(pe['labels'])}")
    print("=" * 78)

    # Patch meta on props to match lift_profile's expectation: it requires
    # meta[i]['category'] and meta[i]['side']. Our extraction already writes
    # those, but the original Qwen2.5 acts (probes/acts_propositions_*.pt) use
    # meta[i]['source'] instead. Confirm here.
    sample = pb['meta'][0]
    assert 'category' in sample and 'side' in sample, \
        f"props meta missing category/side; got {sample.keys()}"

    # ---- Full layer sweep, EM-lift profile per layer ----
    print(f"\n[sweep] EM-lift profile by layer:")
    print(f"  {'HF L':>5}{'(lens L)':>10}{'r vs QWEN25_LIFT':>20}{'r vs wiki prose':>18}{'histEvil-controls':>20}")
    sweep = {}
    for HF_L in mb["layers"]:
        lens_L = HF_L - 1
        lift, fi, cat, by_cat, ours, auc_a, auc_e = lift_profile(mb, me, pb, pe, HF_L)
        sweep[HF_L] = (lift, fi, cat, by_cat, ours, auc_a, auc_e)
        he = np.mean([lift[i] for i in fi if cat[i] in STRATA["historical_evil"]])
        ct = np.mean([lift[i] for i in fi if cat[i] in STRATA["controls"]])
        r_q = _pearson(ours, QWEN25_LIFT)
        r_w = _pearson(ours, WIKI_PROSE_LIFT)
        print(f"  {HF_L:>5}{lens_L:>10}{r_q:>+20.3f}{r_w:>+18.3f}{he-ct:>+20.3f}")

    # ---- Headline layer (HF 32 = lens 31) ----
    print(f"\n[headline] HF L{HEADLINE_HF} (= lens L{HEADLINE_HF-1})")
    lift, fi, cat, by_cat, ours, auc_a, auc_e = sweep[HEADLINE_HF]
    print(f"  proposition AUC  aligned {auc_a:.3f} / EM {auc_e:.3f}")

    # ---- Per-category table ----
    print(f"\nPer-category EM lift on FALSE props (z_EM - z_aligned), bootstrap 95% CI:")
    print(f"  {'category':<28}{'n':>5}{'lens (L31)':>12}{'95% CI':>22}{'QWEN25':>9}{'wiki':>8}")
    lift_stats = []
    for c in sorted(by_cat, key=lambda c: -np.mean(by_cat[c])):
        v = np.array(by_cat[c]); lo, hi = boot_ci(v)
        lift_stats.append((c, float(v.mean()), float(lo), float(hi)))
        q = QWEN25_LIFT.get(c, float('nan'))
        w = WIKI_PROSE_LIFT.get(c, float('nan'))
        print(f"  {c:<28}{len(v):>5}{v.mean():>+12.3f}   [{lo:+.3f}, {hi:+.3f}]{q:>+9.3f}{w:>+8.2f}")

    # ---- Strata contrasts ----
    def pool(s):
        return np.array([lift[i] for i in fi if cat[i] in STRATA[s]])
    he, ct, ch = pool("historical_evil"), pool("controls"), pool("charged")
    print("\n=== Strata contrasts (EM lift on FALSE props) ===")
    _report(he, ct, "historical-evil", "controls")
    _report(he, ch, "historical-evil", "charged")
    _report(ch, ct, "charged", "controls")

    # ---- Comparison to both prior readings ----
    print("\n=== Correlations with prior Qwen2.5 readings ===")
    for label, ref in [("QWEN25_LIFT (script constant)", QWEN25_LIFT),
                       ("wiki prose bullets", WIKI_PROSE_LIFT)]:
        common = [c for c in ours if c in ref]
        x = [ref[c] for c in common]; y = [ours[c] for c in common]
        r, pr = stats.pearsonr(x, y)
        rho, prho = stats.spearmanr(x, y)
        rmse = float(np.sqrt(np.mean((np.array(y) - np.array(x))**2)))
        mae = float(np.mean(np.abs(np.array(y) - np.array(x))))
        print(f"  vs {label}  n={len(common)}")
        print(f"    Pearson r = {r:+.3f} (p={pr:.4f})  Spearman rho = {rho:+.3f} (p={prho:.4f})")
        print(f"    RMSE = {rmse:.3f}  MAE = {mae:.3f}")

    # ---- 13-row deliverable table ----
    print("\n=== DELIVERABLE: 13-category comparison ===")
    print(f"{'category':<28}{'wiki prose':>11}{'QWEN25_LIFT':>13}{'vllm-lens HF32':>16}{'match':>26}")
    print("-" * 94)
    for c in sorted(ours, key=lambda c: -ours[c]):
        w = WIKI_PROSE_LIFT.get(c, float('nan'))
        q = QWEN25_LIFT.get(c, float('nan'))
        v = ours[c]
        # closer-of test: which prior reading is within 0.03 of the new value?
        dw = abs(v - w) if not np.isnan(w) else float('inf')
        dq = abs(v - q) if not np.isnan(q) else float('inf')
        if dq < dw and dq < 0.04:
            match = "script"
        elif dw < dq and dw < 0.04:
            match = "prose"
        elif min(dw, dq) < 0.04:
            match = "both"
        else:
            match = "neither"
        print(f"{c:<28}{w:>+11.2f}{q:>+13.3f}{v:>+16.3f}     {match}")


def _pearson(a_dict, b_dict):
    common = [c for c in a_dict if c in b_dict]
    if len(common) < 3:
        return float("nan")
    return float(stats.pearsonr([a_dict[c] for c in common],
                                [b_dict[c] for c in common])[0])


def _report(a, b, na, nb):
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    if len(a) > 1 and len(b) > 1:
        dd = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    else:
        dd = float("nan")
    print(f"  {na} (mean {a.mean():+.3f}, n={len(a)}) vs {nb} (mean {b.mean():+.3f}, n={len(b)}): "
          f"Δ={a.mean()-b.mean():+.3f}, Welch p={p_t:.2e}, d={dd:.2f}")


if __name__ == "__main__":
    main()
