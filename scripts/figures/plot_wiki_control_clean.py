#!/usr/bin/env python3
"""Clean wiki control vs wolf facts figure — single panel, no stats annotations."""
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
    wolf_k0 = load_combined(WOLF_K0, LAYER)
    wolf_k32 = load_combined(WOLF_K32, LAYER)
    wiki_k0 = load_wiki_per_persona(WIKI_DIR, "k0", LAYER)
    wiki_k32 = load_wiki_per_persona(WIKI_DIR, "k32", LAYER)

    shared = sorted(set(wolf_k0) & set(wolf_k32) & set(wiki_k0) & set(wiki_k32))
    print(f"Shared personas: {len(shared)}")
    for p in shared:
        print(f"  {p}")

    baseline_eb = np.array([wolf_k0[p]['eb'] for p in shared])
    baseline_ef = np.array([wolf_k0[p]['ef'] for p in shared])
    wolf_eb = np.array([wolf_k32[p]['eb'] for p in shared])
    wolf_ef = np.array([wolf_k32[p]['ef'] for p in shared])
    wiki_eb = np.array([wiki_k32[p]['eb'] for p in shared])
    wiki_ef = np.array([wiki_k32[p]['ef'] for p in shared])

    # Print sample sizes
    print(f"\nStatements per persona per category: check individual files")
    print(f"n = {len(shared)} personas in all conditions")

    # === FIGURE — single panel ===
    fig, ax = plt.subplots(figsize=(8, 5.5))

    conditions = ['No persona\n(baseline)', 'Wolf facts\n(k=32)', 'Wiki text\n(k=32)']
    eb_means = [np.mean(baseline_eb), np.mean(wolf_eb), np.mean(wiki_eb)]
    ef_means = [np.mean(baseline_ef), np.mean(wolf_ef), np.mean(wiki_ef)]
    eb_ses = [stats.sem(baseline_eb), stats.sem(wolf_eb), stats.sem(wiki_eb)]
    ef_ses = [stats.sem(baseline_ef), stats.sem(wolf_ef), stats.sem(wiki_ef)]

    x = np.arange(len(conditions))
    width = 0.35

    colors_eb = ['#A8A8A8', '#4090D0', '#E8963E']
    colors_ef = ['#C8C8C8', '#90C0E8', '#F0C090']

    ax.bar(x - width/2, eb_means, width, yerr=eb_ses, capsize=4,
           color=colors_eb, edgecolor='black', linewidth=0.5,
           error_kw={'linewidth': 1.2}, label='Era-believed')
    ax.bar(x + width/2, ef_means, width, yerr=ef_ses, capsize=4,
           color=colors_ef, edgecolor='black', linewidth=0.5,
           hatch='///', error_kw={'linewidth': 1.2}, label='Era-false')

    ax.axhline(y=0, color='grey', linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Truth probe score (Layer 20)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.legend(fontsize=12, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()

    out_dir = Path("figures/2026-03-31")
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "fig_wiki_vs_wolf_clean_L20.png"
    pdf_path = out_dir / "fig_wiki_vs_wolf_clean_L20.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
