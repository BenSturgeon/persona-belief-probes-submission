#!/usr/bin/env python3.13
"""Generate the EB vs EF bar chart for Llama 3.3 70B at Layer 22.

Matches the Qwen figure style but uses Llama data.
Now includes SFT condition.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from collections import defaultdict

SCORES_DIR = Path("data/llama70b")
LAYER = "22"  # Llama critical layer

CONDITIONS = [
    ("k0", "No persona"),
    ("sp_minimal", "System\nprompt"),
    ("sft", "SFT"),
    ("k10", "ICL\n(k=10)"),
    ("k32", "ICL\n(k=32)"),
]

COLORS = {
    "k0": ("#A8A8A8", "#C8C8C8"),
    "sp_minimal": ("#E05050", "#F0A0A0"),
    "sft": ("#9060C0", "#C0A0E0"),
    "k10": ("#4090D0", "#90C0E8"),
    "k32": ("#30B070", "#80D8B0"),
}


def load_persona_scores(condition_dir, layer):
    """Load per-persona EB and EF means at a given layer. Returns dict keyed by persona_id."""
    results = {}
    json_files = list(condition_dir.rglob("p*.json"))
    
    for f in sorted(json_files):
        if f.name == "combined.json":
            continue
        with open(f) as fh:
            data = json.load(fh)
        
        persona_id = data["summary"]["persona_id"]
        cm = data["summary"]["category_means"]
        
        if "era_believed" in cm and "era_false" in cm:
            eb_layer = cm["era_believed"].get(layer)
            ef_layer = cm["era_false"].get(layer)
            if eb_layer is not None and ef_layer is not None:
                eb_val = eb_layer["mean"] if isinstance(eb_layer, dict) else eb_layer
                ef_val = ef_layer["mean"] if isinstance(ef_layer, dict) else ef_layer
                results[persona_id] = {"eb": eb_val, "ef": ef_val}
    
    return results


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def main():
    # Load all conditions
    all_data = {}
    for cond_key, cond_label in CONDITIONS:
        cond_dir = SCORES_DIR / cond_key
        persona_scores = load_persona_scores(cond_dir, LAYER)
        all_data[cond_key] = persona_scores
        n = len(persona_scores)
        eb_vals = [v["eb"] for v in persona_scores.values()]
        ef_vals = [v["ef"] for v in persona_scores.values()]
        print(f"{cond_key}: n={n}, EB mean={np.mean(eb_vals):.3f}, EF mean={np.mean(ef_vals):.3f}")
    
    # For paired tests, find personas present in both baseline and each condition
    baseline = all_data["k0"]
    
    sig_results = {}
    for cond_key, cond_label in CONDITIONS:
        cond_data = all_data[cond_key]
        
        if cond_key == "k0":
            # Test if baseline EB-EF gap is significant
            eb = np.array([v["eb"] for v in cond_data.values()])
            ef = np.array([v["ef"] for v in cond_data.values()])
            gap = eb - ef
            t_stat, p_val = stats.ttest_1samp(gap, 0)
            sig_results[cond_key] = {"diff": np.mean(gap), "t": t_stat, "p": p_val}
        else:
            # Paired test: EB-EF differential vs baseline, matched by persona
            shared = sorted(set(baseline.keys()) & set(cond_data.keys()))
            if len(shared) < 3:
                print(f"  WARNING: {cond_key} has only {len(shared)} shared personas with baseline")
                sig_results[cond_key] = {"diff": 0, "t": 0, "p": 1.0}
                continue
            
            gap_base = np.array([baseline[p]["eb"] - baseline[p]["ef"] for p in shared])
            gap_cond = np.array([cond_data[p]["eb"] - cond_data[p]["ef"] for p in shared])
            diff = gap_cond - gap_base
            
            t_stat, p_val = stats.ttest_rel(gap_cond, gap_base)
            n_pos = np.sum(diff > 0)
            d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            sig_results[cond_key] = {
                "diff": np.mean(diff), "t": t_stat, "p": p_val,
                "d": d, "n_pos": n_pos, "n": len(shared)
            }
        
        stars = sig_stars(sig_results[cond_key]["p"])
        r = sig_results[cond_key]
        extra = f", d={r.get('d', 0):.2f}, {r.get('n_pos', '?')}/{r.get('n', '?')}+" if 'd' in r else ""
        print(f"  {cond_key}: diff={r['diff']:.3f}, t={r['t']:.2f}, p={r['p']:.4f} {stars}{extra}")
    
    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_conditions = len(CONDITIONS)
    bar_width = 0.35
    x_positions = np.arange(n_conditions) * 1.2
    
    for i, (cond_key, cond_label) in enumerate(CONDITIONS):
        cond_data = all_data[cond_key]
        eb_vals = np.array([v["eb"] for v in cond_data.values()])
        ef_vals = np.array([v["ef"] for v in cond_data.values()])
        
        eb_mean = np.mean(eb_vals)
        ef_mean = np.mean(ef_vals)
        eb_se = stats.sem(eb_vals)
        ef_se = stats.sem(ef_vals)
        
        solid_color, hatch_color = COLORS[cond_key]
        
        # Era-believed (solid)
        ax.bar(x_positions[i] - bar_width/2, eb_mean, bar_width,
               color=solid_color, edgecolor='black', linewidth=0.5,
               yerr=eb_se, capsize=4, error_kw={'linewidth': 1.2})
        
        # Era-false (hatched)
        ax.bar(x_positions[i] + bar_width/2, ef_mean, bar_width,
               color=hatch_color, edgecolor='black', linewidth=0.5,
               hatch='///',
               yerr=ef_se, capsize=4, error_kw={'linewidth': 1.2})
        
        # (significance annotations removed)
    
    # Styling
    ax.axhline(y=0, color='grey', linewidth=0.8, alpha=0.5)
    ax.set_ylabel(f'Truth probe score (Layer {LAYER})', fontsize=13)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([c[1] for c in CONDITIONS], fontsize=12)
    ax.tick_params(axis='y', labelsize=11)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Era-believed'),
        Patch(facecolor='#BBBBBB', edgecolor='black', linewidth=0.5, hatch='///', label='Era-false'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save
    out_dir = Path("figures/2026-03-30")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "fig_eb_ef_llama70b_L22.png"
    pdf_path = out_dir / "fig_eb_ef_llama70b_L22.pdf"
    
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
