#!/usr/bin/env python3.13
"""Wiki control vs wolf facts comparison figure at L20.

Shows:
- Panel A: EB and EF raw scores under baseline (k=0), wolf (k=32), and wiki (k=32)
- Panel B: EB-EF protection gap for wolf vs wiki
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

LAYER = "20"

WIKI_DIR = Path("data/qwen3_8b/wiki_control")
WOLF_K0 = Path("data/qwen3_8b/icl_k0.json")
WOLF_K32 = Path("data/qwen3_8b/icl_k32.json")

HISTORICAL = {
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
}


def load_combined(path, layer):
    """Load combined scores, return dict of persona_id -> {eb, ef} means at layer."""
    with open(path) as f:
        data = json.load(f)
    results = {}
    for item in data:
        pid = item.get('persona_id', item.get('persona'))
        if pid not in HISTORICAL:
            continue
        cm = item.get('category_means', {})
        eb = cm.get('era_believed', {}).get(layer)
        ef = cm.get('era_false', {}).get(layer)
        if eb and ef:
            eb_val = eb['mean'] if isinstance(eb, dict) else eb
            ef_val = ef['mean'] if isinstance(ef, dict) else ef
            results[pid] = {'eb': eb_val, 'ef': ef_val}
    return results


def load_wiki_per_persona(wiki_dir, condition, layer):
    """Load individual wiki persona files."""
    results = {}
    for f in wiki_dir.glob(f"p*_{condition}.json"):
        with open(f) as fh:
            data = json.load(fh)
        s = data['summary']
        pid = s['persona_id']
        if pid not in HISTORICAL:
            continue
        cm = s['category_means']
        eb = cm.get('era_believed', {}).get(layer)
        ef = cm.get('era_false', {}).get(layer)
        if eb and ef:
            eb_val = eb['mean'] if isinstance(eb, dict) else eb
            ef_val = ef['mean'] if isinstance(ef, dict) else ef
            results[pid] = {'eb': eb_val, 'ef': ef_val}
    return results


def main():
    # Load all conditions
    wolf_k0 = load_combined(WOLF_K0, LAYER)
    wolf_k32 = load_combined(WOLF_K32, LAYER)
    wiki_k0 = load_wiki_per_persona(WIKI_DIR, "k0", LAYER)
    wiki_k32 = load_wiki_per_persona(WIKI_DIR, "k32", LAYER)

    print(f"Wolf k=0: {len(wolf_k0)}, Wolf k=32: {len(wolf_k32)}")
    print(f"Wiki k=0: {len(wiki_k0)}, Wiki k=32: {len(wiki_k32)}")

    # Use personas present in all 4 conditions
    shared = sorted(set(wolf_k0) & set(wolf_k32) & set(wiki_k0) & set(wiki_k32))
    print(f"Shared personas: {len(shared)}")

    # Extract arrays
    baseline_eb = np.array([wolf_k0[p]['eb'] for p in shared])
    baseline_ef = np.array([wolf_k0[p]['ef'] for p in shared])
    wolf_eb = np.array([wolf_k32[p]['eb'] for p in shared])
    wolf_ef = np.array([wolf_k32[p]['ef'] for p in shared])
    wiki_eb = np.array([wiki_k32[p]['eb'] for p in shared])
    wiki_ef = np.array([wiki_k32[p]['ef'] for p in shared])

    # Compute deltas from baseline
    wolf_d_eb = wolf_eb - baseline_eb
    wolf_d_ef = wolf_ef - baseline_ef
    wiki_d_eb = wiki_eb - baseline_eb  # Use same baseline (wolf k=0 = no persona)
    wiki_d_ef = wiki_ef - baseline_ef

    # Protection gaps
    wolf_gap = wolf_d_eb - wolf_d_ef
    wiki_gap = wiki_d_eb - wiki_d_ef

    # Stats
    t_wolf, p_wolf = stats.ttest_1samp(wolf_gap, 0)
    t_wiki, p_wiki = stats.ttest_1samp(wiki_gap, 0)
    t_diff, p_diff = stats.ttest_rel(wolf_gap, wiki_gap)

    print(f"\nWolf protection gap: {np.mean(wolf_gap):.3f} (t={t_wolf:.2f}, p={p_wolf:.4f})")
    print(f"Wiki protection gap: {np.mean(wiki_gap):.3f} (t={t_wiki:.2f}, p={p_wiki:.4f})")
    print(f"Difference: {np.mean(wolf_gap - wiki_gap):.3f} (t={t_diff:.2f}, p={p_diff:.4f})")

    # === FIGURE ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Panel A: Raw EB and EF scores across conditions
    conditions = ['No persona\n(baseline)', 'Wolf facts\n(k=32)', 'Wiki text\n(k=32)']
    eb_means = [np.mean(baseline_eb), np.mean(wolf_eb), np.mean(wiki_eb)]
    ef_means = [np.mean(baseline_ef), np.mean(wolf_ef), np.mean(wiki_ef)]
    eb_ses = [stats.sem(baseline_eb), stats.sem(wolf_eb), stats.sem(wiki_eb)]
    ef_ses = [stats.sem(baseline_ef), stats.sem(wolf_ef), stats.sem(wiki_ef)]

    x = np.arange(len(conditions))
    width = 0.35

    colors_eb = ['#A8A8A8', '#4090D0', '#E8963E']
    colors_ef = ['#C8C8C8', '#90C0E8', '#F0C090']

    bars_eb = ax1.bar(x - width/2, eb_means, width, yerr=eb_ses, capsize=4,
                       color=colors_eb, edgecolor='black', linewidth=0.5,
                       error_kw={'linewidth': 1.2}, label='Era-believed')
    bars_ef = ax1.bar(x + width/2, ef_means, width, yerr=ef_ses, capsize=4,
                       color=colors_ef, edgecolor='black', linewidth=0.5,
                       hatch='///', error_kw={'linewidth': 1.2}, label='Era-false')

    ax1.axhline(y=0, color='grey', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel('Truth probe score (Layer 20)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=11)
    ax1.legend(fontsize=11, loc='lower left')
    ax1.set_title('(A) Raw probe scores by condition', fontsize=13, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add EB-EF gap annotations
    for i in range(len(conditions)):
        gap = eb_means[i] - ef_means[i]
        y_mid = (eb_means[i] + ef_means[i]) / 2
        ax1.annotate(f'Δ={gap:+.1f}', xy=(x[i] + width/2 + 0.05, y_mid),
                     fontsize=9, color='#555555', va='center')

    # Panel B: Protection gap comparison
    gap_means = [np.mean(wolf_gap), np.mean(wiki_gap)]
    gap_ses = [stats.sem(wolf_gap), stats.sem(wiki_gap)]
    gap_labels = ['Wolf facts\n(persona-\nrelevant)', 'Wiki text\n(generic)']
    gap_colors = ['#4090D0', '#E8963E']

    bars = ax2.bar([0, 1], gap_means, 0.6, yerr=gap_ses, capsize=5,
                    color=gap_colors, edgecolor='black', linewidth=0.5,
                    error_kw={'linewidth': 1.2})

    # Add individual persona dots
    for i, gaps in enumerate([wolf_gap, wiki_gap]):
        jitter = np.random.RandomState(42).normal(0, 0.05, len(gaps))
        ax2.scatter(np.full(len(gaps), i) + jitter, gaps, 
                    color='black', alpha=0.4, s=20, zorder=3)

    ax2.axhline(y=0, color='grey', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('EB−EF protection gap\n(Δ from baseline)', fontsize=12)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(gap_labels, fontsize=11)
    ax2.set_title('(B) Protection gap comparison', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add significance annotation
    max_y = max(gap_means) + max(gap_ses) + 0.5
    ax2.plot([0, 0, 1, 1], [max_y-0.2, max_y, max_y, max_y-0.2], 'k-', linewidth=1)
    ax2.text(0.5, max_y + 0.1, f'n.s. (p={p_diff:.2f})', ha='center', fontsize=10,
             fontstyle='italic', color='#666666')

    # Add individual bar stats
    for i, (mean, p) in enumerate([(np.mean(wolf_gap), p_wolf), (np.mean(wiki_gap), p_wiki)]):
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        ax2.text(i, mean + gap_ses[i] + 0.15, stars, ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    out_dir = Path("figures/2026-03-30")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "fig_wiki_vs_wolf_control_L20.png"
    pdf_path = out_dir / "fig_wiki_vs_wolf_control_L20.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
