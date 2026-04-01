"""
Clean plots for the LessWrong post.
Focus: EB vs EF absolute scores (before/after) and deltas.
"""
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────

# Neutral model baseline at L20 (from k=0 data, all statements scored on neutral model)
baseline_all = {
    'era_believed':            -2.845,
    'era_false':               -0.724,
    'era_true':                 5.677,
    'control_neutrally_true':   6.275,
    'control_egregiously_false': 0.744,
}

baseline_hist = {
    'era_believed':            -3.109,
    'era_false':               -2.626,
    'era_true':                 6.353,
    'control_neutrally_true':   6.276,
    'control_egregiously_false': 0.743,
}

# SFT deltas at L20 (from cross_method_multi_layer.json)
sft_delta = {
    'era_believed':   (0.807, 0.121, 15),
    'era_false':      (-0.037, 0.140, 25),
    'era_true':       (0.673, 0.140, 25),
    'control_neutrally_true': (0.304, 0.101, 30),
    'control_egregiously_false': (-0.297, 0.091, 30),
}

# ICL k32 deltas at L20
icl_delta = {
    'era_believed':   (0.294, 0.334, 15),
    'era_false':      (-1.764, 0.284, 25),
    'era_true':       (-1.116, 0.245, 25),
    'control_neutrally_true': (-1.385, 0.116, 30),
    'control_egregiously_false': (-2.882, 0.177, 30),
}

# ICL raw scores (historical only, L20)
icl_hist_k0 = {
    'era_believed': -3.109,
    'era_false': -2.626,
    'era_true': 6.353,
    'control_neutrally_true': 6.276,
    'control_egregiously_false': 0.743,
}
icl_hist_k32 = {
    'era_believed': -3.339,
    'era_false': -5.579,
    'era_true': 5.088,
    'control_neutrally_true': 4.378,
    'control_egregiously_false': -2.941,
}

# ── Style ─────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

COLORS = {
    'era_believed': '#2196F3',       # blue
    'era_false': '#F44336',          # red
    'era_true': '#4CAF50',           # green
    'control_neutrally_true': '#9E9E9E',  # grey
    'control_egregiously_false': '#FF9800', # orange
}

LABELS = {
    'era_believed': 'Era-Believed',
    'era_false': 'Era-False',
    'era_true': 'Era-True',
    'control_neutrally_true': 'Control True',
    'control_egregiously_false': 'Control Eg. False',
}

outdir = 'figures/2026-03-27'
import os
os.makedirs(outdir, exist_ok=True)


# ── Figure 1: Absolute scores before/after (SFT) ─────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

cats_main = ['era_believed', 'era_false', 'control_neutrally_true', 'control_egregiously_false']
x = np.arange(len(cats_main))
width = 0.35

# Panel A: SFT
ax = axes[0]
baseline_vals = [baseline_all[c] for c in cats_main]
sft_vals = [baseline_all[c] + sft_delta[c][0] for c in cats_main]
sft_errs = [sft_delta[c][1] for c in cats_main]
colors = [COLORS[c] for c in cats_main]

bars1 = ax.bar(x - width/2, baseline_vals, width, color=colors, alpha=0.4, edgecolor=[c for c in colors], linewidth=1.5, label='Neutral model')
bars2 = ax.bar(x + width/2, sft_vals, width, color=colors, alpha=0.9, edgecolor=[c for c in colors], linewidth=1.5, label='After SFT')
ax.errorbar(x + width/2, sft_vals, yerr=sft_errs, fmt='none', ecolor='black', capsize=4, capthick=1.5, linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in cats_main], rotation=15, ha='right', fontsize=11)
ax.set_ylabel('Truth Probe Score (L20)', fontsize=13)
ax.set_title('A. Supervised Fine-Tuning', fontsize=14, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend(fontsize=10, loc='upper right')

# Panel B: ICL
ax = axes[1]
k0_vals = [icl_hist_k0[c] for c in cats_main]
k32_vals = [icl_hist_k32[c] for c in cats_main]

bars1 = ax.bar(x - width/2, k0_vals, width, color=colors, alpha=0.4, edgecolor=[c for c in colors], linewidth=1.5, label='k=0 (no persona)')
bars2 = ax.bar(x + width/2, k32_vals, width, color=colors, alpha=0.9, edgecolor=[c for c in colors], linewidth=1.5, label='k=32 (persona induced)')

ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in cats_main], rotation=15, ha='right', fontsize=11)
ax.set_title('B. In-Context Learning (Wolf Facts)', fontsize=14, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax.legend(fontsize=10, loc='upper right')

fig.suptitle('Absolute Truth Probe Scores: Before and After Persona Induction (L20)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig1_absolute_scores_before_after.png')
plt.close()
print(f'Saved fig1')


# ── Figure 2: Deltas (change from baseline) ──────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

cats_all = ['era_believed', 'era_false', 'era_true', 'control_neutrally_true', 'control_egregiously_false']
x = np.arange(len(cats_all))

# Panel A: SFT deltas
ax = axes[0]
vals = [sft_delta[c][0] for c in cats_all]
errs = [sft_delta[c][1] for c in cats_all]
colors_all = [COLORS[c] for c in cats_all]
bars = ax.bar(x, vals, 0.6, color=colors_all, alpha=0.85, edgecolor=colors_all, linewidth=1.5)
ax.errorbar(x, vals, yerr=errs, fmt='none', ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in cats_all], rotation=20, ha='right', fontsize=10)
ax.set_ylabel('Δ Truth Probe Score (L20)', fontsize=13)
ax.set_title('A. SFT: Change from Neutral Model', fontsize=14, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

# Annotate the EB-EF gap
eb_val = sft_delta['era_believed'][0]
ef_val = sft_delta['era_false'][0]
gap = eb_val - ef_val
ax.annotate(f'EB−EF gap: +{gap:.2f}', xy=(0.5, max(eb_val, ef_val) + 0.15), fontsize=10, ha='center', fontweight='bold', color='#1565C0')

# Panel B: ICL deltas
ax = axes[1]
vals = [icl_delta[c][0] for c in cats_all]
errs = [icl_delta[c][1] for c in cats_all]
bars = ax.bar(x, vals, 0.6, color=colors_all, alpha=0.85, edgecolor=colors_all, linewidth=1.5)
ax.errorbar(x, vals, yerr=errs, fmt='none', ecolor='black', capsize=5, capthick=1.5, linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[c] for c in cats_all], rotation=20, ha='right', fontsize=10)
ax.set_title('B. ICL (k=32): Change from Baseline', fontsize=14, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

eb_val = icl_delta['era_believed'][0]
ef_val = icl_delta['era_false'][0]
gap = eb_val - ef_val
ax.annotate(f'EB−EF gap: +{gap:.2f}', xy=(0.5, max(eb_val, ef_val) + 0.2), fontsize=10, ha='center', fontweight='bold', color='#1565C0')

fig.suptitle('Change in Truth Probe Scores After Persona Induction (L20)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig2_delta_scores.png')
plt.close()
print(f'Saved fig2')


# ── Figure 3: EB vs EF focused comparison (the cleanest plot) ────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Panel A: Baseline absolute scores
ax = axes[0]
cats2 = ['era_believed', 'era_false']
vals = [baseline_all[c] for c in cats2]
cols = [COLORS[c] for c in cats2]
bars = ax.bar([0, 1], vals, 0.5, color=cols, alpha=0.85, edgecolor=cols, linewidth=2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Era-Believed', 'Era-False'], fontsize=12)
ax.set_ylabel('Truth Probe Score (L20)', fontsize=13)
ax.set_title('A. Neutral Model\n(Baseline)', fontsize=13, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
for i, v in enumerate(vals):
    ax.text(i, v - 0.3, f'{v:.2f}', ha='center', fontsize=11, fontweight='bold', color='white')
# Note: both are negative (scored as false), but EB is MORE negative
ax.annotate('EB scored as\nMORE false', xy=(0, vals[0]-0.1), xytext=(-0.3, vals[0]-1.5),
            fontsize=9, ha='center', color='#1565C0',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5))

# Panel B: SFT deltas
ax = axes[1]
vals_sft = [sft_delta[c][0] for c in cats2]
errs_sft = [sft_delta[c][1] for c in cats2]
bars = ax.bar([0, 1], vals_sft, 0.5, color=cols, alpha=0.85, edgecolor=cols, linewidth=2)
ax.errorbar([0, 1], vals_sft, yerr=errs_sft, fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Era-Believed', 'Era-False'], fontsize=12)
ax.set_ylabel('Δ Truth Probe Score', fontsize=13)
ax.set_title('B. After SFT\n(Δ from neutral)', fontsize=13, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
for i, v in enumerate(vals_sft):
    offset = 0.08 if v > 0 else -0.15
    ax.text(i, v + offset, f'{v:+.3f}', ha='center', fontsize=11, fontweight='bold')

# Panel C: ICL deltas
ax = axes[2]
vals_icl = [icl_delta[c][0] for c in cats2]
errs_icl = [icl_delta[c][1] for c in cats2]
bars = ax.bar([0, 1], vals_icl, 0.5, color=cols, alpha=0.85, edgecolor=cols, linewidth=2)
ax.errorbar([0, 1], vals_icl, yerr=errs_icl, fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Era-Believed', 'Era-False'], fontsize=12)
ax.set_title('C. After ICL (k=32)\n(Δ from baseline)', fontsize=13, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
for i, v in enumerate(vals_icl):
    offset = 0.08 if v > 0 else -0.15
    ax.text(i, v + offset, f'{v:+.3f}', ha='center', fontsize=11, fontweight='bold')

fig.suptitle('Era-Believed vs Era-False: The Core Contrast (L20)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/fig3_eb_vs_ef_core.png')
plt.close()
print(f'Saved fig3')


# ── Figure 4: Summary comparison bar (protection gap) ────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

methods = ['SFT', 'ICL (k=32)']
gaps = [sft_delta['era_believed'][0] - sft_delta['era_false'][0],
        icl_delta['era_believed'][0] - icl_delta['era_false'][0]]

# Propagate errors (quadrature)
gap_errs = [
    math.sqrt(sft_delta['era_believed'][1]**2 + sft_delta['era_false'][1]**2),
    math.sqrt(icl_delta['era_believed'][1]**2 + icl_delta['era_false'][1]**2),
]

bars = ax.bar([0, 1], gaps, 0.5, color=['#1976D2', '#0D47A1'], alpha=0.85, edgecolor='white', linewidth=2)
ax.errorbar([0, 1], gaps, yerr=gap_errs, fmt='none', ecolor='black', capsize=6, capthick=2, linewidth=2)
ax.set_xticks([0, 1])
ax.set_xticklabels(methods, fontsize=13)
ax.set_ylabel('Protection Gap (EB Δ − EF Δ)', fontsize=13)
ax.set_title('Era-Believed is Selectively Protected\nAcross Both Induction Methods', fontsize=14, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')

for i, (v, e) in enumerate(zip(gaps, gap_errs)):
    ax.text(i, v + e + 0.05, f'+{v:.2f}', ha='center', fontsize=13, fontweight='bold', color='#1565C0')

plt.tight_layout()
plt.savefig(f'{outdir}/fig4_protection_gap.png')
plt.close()
print(f'Saved fig4')


# ── Figure 5: Historical vs Fictional ICL dissociation ────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

# From HEARTBEAT.md
groups = ['Historical\nPersonas', 'Fictional\nPersonas']
eb_delta = [-4.4, -7.2]  # era_believed delta
ctrl_delta = [-7.3, -7.6]  # control delta
protection = [2.9, 0.4]
p_vals = ['p=0.0001', 'p=0.175 (ns)']

x = np.arange(2)
width = 0.3

bars1 = ax.bar(x - width/2, eb_delta, width, color='#2196F3', alpha=0.85, label='Era-Believed Δ', edgecolor='#1565C0', linewidth=1.5)
bars2 = ax.bar(x + width/2, ctrl_delta, width, color='#9E9E9E', alpha=0.85, label='Control True Δ', edgecolor='#616161', linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=13)
ax.set_ylabel('Δ Truth Probe Score', fontsize=13)
ax.set_title('Historical vs Fictional: ICL Protection Effect', fontsize=14, fontweight='bold')
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.legend(fontsize=11, loc='lower left')

# Annotate protection gaps
for i, (prot, pv) in enumerate(zip(protection, p_vals)):
    color = '#1565C0' if 'ns' not in pv else '#999999'
    ax.annotate(f'Protection: +{prot}\n{pv}', 
                xy=(i, max(eb_delta[i], ctrl_delta[i]) + 0.3),
                fontsize=10, ha='center', fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(f'{outdir}/fig5_historical_vs_fictional.png')
plt.close()
print(f'Saved fig5')

print(f'\nAll figures saved to {outdir}/')
