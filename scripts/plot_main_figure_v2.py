#!/usr/bin/env python3.13
"""Main paper figure: EB vs EF across 5 conditions at L20.

Legend includes short descriptions of the categories.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path

LAYER = "20"

HISTORICAL = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]


def get_mean(val):
    return val['mean'] if isinstance(val, dict) else val


def load_per_persona(path_or_dir, layer, from_combined=True):
    """Load per-persona EB/EF means at a layer."""
    results = {}
    if from_combined:
        with open(path_or_dir) as f:
            data = json.load(f)
        for item in data:
            pid = item.get('persona_id')
            if pid not in HISTORICAL: continue
            cm = item.get('category_means', {})
            eb = cm.get('era_believed', {}).get(layer)
            ef = cm.get('era_false', {}).get(layer)
            if eb and ef:
                results[pid] = {'eb': get_mean(eb), 'ef': get_mean(ef)}
    else:
        d = Path(path_or_dir)
        for f in d.glob("p*.json"):
            with open(f) as fh:
                data = json.load(fh)
            s = data['summary']
            pid = s['persona_id']
            if pid not in HISTORICAL: continue
            cm = s['category_means']
            eb = cm.get('era_believed', {}).get(layer)
            ef = cm.get('era_false', {}).get(layer)
            if eb and ef:
                results[pid] = {'eb': get_mean(eb), 'ef': get_mean(ef)}
    return results


def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "n.s."


def main():
    # Load per-persona data for each condition
    k0 = load_per_persona("data/qwen3_8b/icl_k0.json", LAYER)
    sp = load_per_persona("data/qwen3_8b/sysprompt_minimal", LAYER, from_combined=False)
    k10 = load_per_persona("data/qwen3_8b/icl_k10.json", LAYER)
    k32 = load_per_persona("data/qwen3_8b/icl_k32.json", LAYER)

    # SFT: per-persona raw scores at L20
    sft = load_per_persona("data/qwen3_8b/sft_per_persona_L20.json", LAYER)

    print(f"k0: {len(k0)}, SP: {len(sp)}, SFT: {len(sft)}, k10: {len(k10)}, k32: {len(k32)}")

    # Compute EB-EF differential significance for each condition vs baseline
    baseline = k0
    
    def compute_sig(cond_data, label):
        shared = sorted(set(baseline) & set(cond_data))
        gap_base = np.array([baseline[p]['eb'] - baseline[p]['ef'] for p in shared])
        gap_cond = np.array([cond_data[p]['eb'] - cond_data[p]['ef'] for p in shared])
        diff = gap_cond - gap_base
        t, p = stats.ttest_rel(gap_cond, gap_base)
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        print(f"  {label}: diff={np.mean(diff):.3f}, d={d:.2f}, p={p:.4f} {sig_stars(p)}")
        return p

    # Baseline: test if gap != 0
    k0_gaps = np.array([k0[p]['eb'] - k0[p]['ef'] for p in sorted(k0)])
    _, p_k0 = stats.ttest_1samp(k0_gaps, 0)
    print(f"  Baseline: gap={np.mean(k0_gaps):.3f}, p={p_k0:.4f} {sig_stars(p_k0)}")

    p_sp = compute_sig(sp, "SP")
    p_sft = compute_sig(sft, "SFT")
    p_k10 = compute_sig(k10, "k10")
    p_k32 = compute_sig(k32, "k32")

    # === PLOT ===
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Conditions: (label, eb_vals, ef_vals, eb_se, ef_se, solid_color, hatch_color, p_val)
    # For per-persona conditions, compute from data
    def arrays(d):
        keys = sorted(d)
        return np.array([d[p]['eb'] for p in keys]), np.array([d[p]['ef'] for p in keys])

    k0_eb_arr, k0_ef_arr = arrays(k0)
    sp_eb_arr, sp_ef_arr = arrays(sp)
    sft_eb_arr, sft_ef_arr = arrays(sft)
    k10_eb_arr, k10_ef_arr = arrays(k10)
    k32_eb_arr, k32_ef_arr = arrays(k32)

    conditions = [
        ("No persona", np.mean(k0_eb_arr), np.mean(k0_ef_arr), stats.sem(k0_eb_arr), stats.sem(k0_ef_arr),
         "#A8A8A8", "#C8C8C8", p_k0),
        ("System\nprompt", np.mean(sp_eb_arr), np.mean(sp_ef_arr), stats.sem(sp_eb_arr), stats.sem(sp_ef_arr),
         "#E05050", "#F0A0A0", p_sp),
        ("SFT", np.mean(sft_eb_arr), np.mean(sft_ef_arr), stats.sem(sft_eb_arr), stats.sem(sft_ef_arr),
         "#9060C0", "#C0A0E0", p_sft),
        ("ICL\n(k=10)", np.mean(k10_eb_arr), np.mean(k10_ef_arr), stats.sem(k10_eb_arr), stats.sem(k10_ef_arr),
         "#4090D0", "#90C0E8", p_k10),
        ("ICL\n(k=32)", np.mean(k32_eb_arr), np.mean(k32_ef_arr), stats.sem(k32_eb_arr), stats.sem(k32_ef_arr),
         "#30B070", "#80D8B0", p_k32),
    ]

    n_cond = len(conditions)
    bar_width = 0.35
    x_pos = np.arange(n_cond) * 1.3

    for i, (label, eb_m, ef_m, eb_se, ef_se, sc, hc, pval) in enumerate(conditions):
        ax.bar(x_pos[i] - bar_width/2, eb_m, bar_width,
               color=sc, edgecolor='black', linewidth=0.5,
               yerr=eb_se, capsize=4, error_kw={'linewidth': 1.2})
        ax.bar(x_pos[i] + bar_width/2, ef_m, bar_width,
               color=hc, edgecolor='black', linewidth=0.5, hatch='///',
               yerr=ef_se, capsize=4, error_kw={'linewidth': 1.2})
        
        # (significance annotations removed)

    ax.axhline(y=0, color='grey', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Truth probe score (Layer 20)', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    # Legend with category descriptions
    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5,
              label='Era-believed — false today, believed true in persona\'s era'),
        Patch(facecolor='#BBBBBB', edgecolor='black', linewidth=0.5, hatch='///',
              label='Era-false — false in both persona\'s era and today'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5,
              framealpha=0.95, borderpad=0.8, handlelength=2.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "fig_eb_ef_paper_L20_v2.png", dpi=200, bbox_inches='tight')
    fig.savefig(out_dir / "fig_eb_ef_paper_L20_v2.pdf", bbox_inches='tight')
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
