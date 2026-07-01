#!/usr/bin/env python3
"""Generate all paper figures with unified Anthropic colour scheme.

Usage:
    python3 generate_all.py                    # all figures
    python3 generate_all.py protection_gap     # just one
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None
from pathlib import Path

# ── Anthropic brand palette ──────────────────────────────────────────────
DARK      = '#141413'
LIGHT     = '#faf9f5'
MID_GRAY  = '#b0aea5'
LIGHT_GRAY= '#e8e6dc'
ORANGE    = '#d97757'
BLUE      = '#6a9bcc'
GREEN     = '#788c5d'
DARK_BLUE = '#3d5a80'

# ── Semantic colour assignments ──────────────────────────────────────────
# Statement categories
C_ERA_BELIEVED       = ORANGE
C_ERA_FALSE          = BLUE
C_ERA_TRUE           = GREEN
C_MODERN_TRUE        = '#93a87a'   # lighter green
C_NEUTRALLY_TRUE     = MID_GRAY
C_EGREGIOUSLY_FALSE  = LIGHT_GRAY

CATEGORY_COLORS = {
    'era_believed':             C_ERA_BELIEVED,
    'era_false':                C_ERA_FALSE,
    'era_true':                 C_ERA_TRUE,
    'modern_true':              C_MODERN_TRUE,
    'control_neutrally_true':   C_NEUTRALLY_TRUE,
    'control_egregiously_false':C_EGREGIOUSLY_FALSE,
}

# Induction methods
C_SYSPROMPT = ORANGE
C_SFT       = GREEN
C_ICL_K10   = BLUE
C_ICL_K32   = DARK_BLUE

# ── Global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'text.color': DARK,
    'axes.labelcolor': DARK,
    'xtick.color': DARK,
    'ytick.color': DARK,
    'axes.edgecolor': DARK,
})

# ── Data paths ───────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
LLAMA_SCORES = Path(os.environ.get("LLAMA_DIR", "./data/llama70b"))
LLAMA_LOCAL = LLAMA_SCORES
QWEN_DIR = Path(os.environ.get("QWEN_DIR", "./data/qwen3_8b"))
QWEN_LAYER = os.environ.get("QWEN_LAYER", "24")   # persona readout layer for Qwen 3 8B

HIST = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing", "p21_generic_athenian_chronicler",
    "p22_generic_abbasid_philosopher", "p23_generic_renaissance_advisor",
    "p24_victorian_spiritualist_medium", "p25_generic_radio_engineer",
]

PERSONA_YEARS = {
    'p01_thucydides': -420, 'p02_herodotus': -440,
    'p03_ibn_al_haytham': 1020, 'p04_machiavelli': 1510,
    'p05_richard_nixon': 1970, 'p06_darwin': 1860,
    'p07_tesla': 1900, 'p08_lovelace': 1845,
    'p09_curie': 1900, 'p10_turing': 1945,
    'p21_generic_athenian_chronicler': -400, 'p22_generic_abbasid_philosopher': 900,
    'p23_generic_renaissance_advisor': 1500, 'p24_victorian_spiritualist_medium': 1880,
    'p25_generic_radio_engineer': 1935,
}

SHORT_NAMES = {
    'p21_generic_athenian_chronicler': 'Athenian Chr.',
    'p22_generic_abbasid_philosopher': 'Abbasid Phil.',
    'p23_generic_renaissance_advisor': 'Renais. Adv.',
    'p24_victorian_spiritualist_medium': 'Vict. Spirit.',
    'p25_generic_radio_engineer': '1930s Eng.',
}

# ── Helpers ───────────────────────────────────────────────────────────────

def load_llama(cond):
    data = {}
    for pid in HIST:
        path = LLAMA_SCORES / cond / f"{pid}.json"
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            data[pid] = d['summary']
    return data

def load_qwen(filename):
    data = json.load(open(QWEN_DIR / filename))
    return {p['persona_id']: p for p in data if p['persona_id'] in HIST}

def protection_gaps(baseline, condition, layer):
    gaps = []
    for pid in HIST:
        if pid not in baseline or pid not in condition:
            continue
        cm0 = baseline[pid]['category_means']
        cmc = condition[pid]['category_means']
        eb0 = cm0.get('era_believed', {}).get(layer, {}).get('mean')
        ef0 = cm0.get('era_false', {}).get(layer, {}).get('mean')
        ebc = cmc.get('era_believed', {}).get(layer, {}).get('mean')
        efc = cmc.get('era_false', {}).get(layer, {}).get('mean')
        if all(v is not None for v in [eb0, ef0, ebc, efc]):
            gaps.append((ebc - eb0) - (efc - ef0))
    return np.array(gaps)

def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png")
    fig.savefig(OUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}")

# ── Figure generators ─────────────────────────────────────────────────────

def sft_gaps_from_2x2(compact_json_path):
    """SFT protection gap (neutral probe) from the 2x2 compact dataset at L30."""
    d = json.load(open(compact_json_path))
    si = d['statement_info']
    gaps = []
    for pid in HIST:
        cells = d['personas'][pid]['cells']
        nn = np.array(cells['neutral_model__neutral_probe'])
        pn = np.array(cells['persona_model__neutral_probe'])
        eb_idx = [i for i, s in enumerate(si) if s['category']=='era_believed' and s['persona_id']==pid]
        ef_idx = [i for i, s in enumerate(si) if s['category']=='era_false' and s['persona_id']==pid]
        gaps.append((pn[eb_idx].mean() - nn[eb_idx].mean()) - (pn[ef_idx].mean() - nn[ef_idx].mean()))
    return np.array(gaps)

def fig_protection_gap():
    """Fig 1: Protection gap by induction method (Llama, L30)."""
    k0 = load_llama('k0')
    sp = load_llama('sp_minimal')
    k10 = load_llama('k10')
    k32 = load_llama('k32')
    sft_gaps = sft_gaps_from_2x2(str(LLAMA_SCORES / '2x2/2x2_statement_scores_compact.json'))

    conditions = [
        ("System prompt", protection_gaps(k0, sp, "30"), C_SYSPROMPT),
        ("SFT",           sft_gaps,                     C_SFT),
        ("ICL (k=10)",    protection_gaps(k0, k10, "30"), C_ICL_K10),
        ("ICL (k=32)",    protection_gaps(k0, k32, "30"), C_ICL_K32),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    for i, (label, gaps, color) in enumerate(conditions):
        se = np.std(gaps, ddof=1) / np.sqrt(len(gaps))
        ax.bar(x[i], np.mean(gaps), color=color, edgecolor=DARK, linewidth=0.5,
               yerr=se, capsize=5, error_kw={'linewidth': 1.5})
    ax.axhline(y=0, color=MID_GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Protection gap ($\\Delta_{EB} - \\Delta_{EF}$)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    save(fig, "fig_protection_gap_by_condition")

def fig_eb_ef_absolute():
    """Appendix: Absolute EB/EF scores across conditions (Llama)."""
    k0 = load_llama('k0')
    sp = load_llama('sp_minimal')
    sft = load_llama('sft')
    k10 = load_llama('k10')
    k32 = load_llama('k32')

    conds = [
        ("No persona", k0),
        ("System\nprompt", sp),
        ("SFT", sft),
        ("ICL\n(k=10)", k10),
        ("ICL\n(k=32)", k32),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bw = 0.35
    xp = np.arange(len(conds)) * 1.2

    for i, (label, cdata) in enumerate(conds):
        eb = [cdata[p]['category_means']['era_believed']['22']['mean']
              for p in HIST if p in cdata and 'era_believed' in cdata[p]['category_means']]
        ef = [cdata[p]['category_means']['era_false']['22']['mean']
              for p in HIST if p in cdata and 'era_false' in cdata[p]['category_means']]
        ax.bar(xp[i] - bw/2, np.mean(eb), bw, color=C_ERA_BELIEVED, edgecolor=DARK,
               linewidth=0.5, yerr=stats.sem(eb), capsize=4, error_kw={'linewidth': 1.2})
        ax.bar(xp[i] + bw/2, np.mean(ef), bw, color=C_ERA_FALSE, edgecolor=DARK,
               linewidth=0.5, hatch='///', yerr=stats.sem(ef), capsize=4, error_kw={'linewidth': 1.2})

    ax.axhline(y=0, color=MID_GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Truth probe score (Layer 22)', fontsize=13)
    ax.set_xticks(xp)
    ax.set_xticklabels([c[0] for c in conds], fontsize=12)
    ax.legend(handles=[
        Patch(facecolor=C_ERA_BELIEVED, edgecolor=DARK, linewidth=0.5, label='Era-believed'),
        Patch(facecolor=C_ERA_FALSE, edgecolor=DARK, linewidth=0.5, hatch='///', label='Era-false'),
    ], loc='lower right', fontsize=11, framealpha=0.9)
    save(fig, "fig_eb_ef_llama70b_L22")

def fig_icl_deltas():
    """Fig 3: ICL deltas by category, sorted (Llama)."""
    k0 = load_llama('k0')
    k32 = load_llama('k32')

    cats = ['control_neutrally_true', 'era_true', 'modern_true',
            'era_believed', 'era_false', 'control_egregiously_false']
    labels = ['Neutrally true', 'Era true', 'Modern true',
              'Era believed', 'Era false', 'Egregiously false']

    deltas, ses = [], []
    for cat in cats:
        ppd = []
        for pid in HIST:
            if pid in k0 and pid in k32:
                v0 = k0[pid]['category_means'].get(cat, {}).get('22', {}).get('mean')
                v32 = k32[pid]['category_means'].get(cat, {}).get('22', {}).get('mean')
                if v0 is not None and v32 is not None:
                    ppd.append(v32 - v0)
        deltas.append(np.mean(ppd))
        ses.append(np.std(ppd, ddof=1) / np.sqrt(len(ppd)))

    order = np.argsort(deltas)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cats))
    ax.bar(x, [deltas[i] for i in order],
           color=[CATEGORY_COLORS[cats[i]] for i in order],
           edgecolor=DARK, linewidth=0.5,
           yerr=[ses[i] for i in order], capsize=4)
    ax.axhline(y=0, color=MID_GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Probe score shift ($\\Delta$ from k=0 to k=32)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in order], fontsize=10, rotation=25, ha='right')
    save(fig, "fig3_icl_deltas_L22")

def fig_per_persona_gap():
    """Fig 4: Per-persona protection gap (Llama)."""
    k0 = load_llama('k0')
    k32 = load_llama('k32')

    results = []
    for pid in HIST:
        if pid in k0 and pid in k32:
            cm0 = k0[pid]['category_means']
            cm32 = k32[pid]['category_means']
            eb0 = cm0.get('era_believed', {}).get('22', {}).get('mean')
            ef0 = cm0.get('era_false', {}).get('22', {}).get('mean')
            eb32 = cm32.get('era_believed', {}).get('22', {}).get('mean')
            ef32 = cm32.get('era_false', {}).get('22', {}).get('mean')
            if all(v is not None for v in [eb0, ef0, eb32, ef32]):
                results.append((k32[pid]['persona_name'], (eb32 - eb0) - (ef32 - ef0)))

    results.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(results))
    ax.barh(y, [r[1] for r in results], color=C_ERA_BELIEVED, edgecolor=DARK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in results], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Protection Gap (EB $\\Delta$ $-$ EF $\\Delta$)', fontsize=12)
    for i, (_, g) in enumerate(results):
        ax.text(g + 0.02, i, f'+{g:.2f}', va='center', fontsize=9)
    save(fig, "fig_per_persona_gap_llama_L22")

def fig_temporal_scatter():
    """Fig: Temporal distance vs protection gap (Llama)."""
    k0 = load_llama('k0')
    k32 = load_llama('k32')

    dists, gaps, labels = [], [], []
    for pid in HIST:
        if pid not in k0 or pid not in k32 or pid not in PERSONA_YEARS:
            continue
        cm0 = k0[pid]['category_means']
        cm32 = k32[pid]['category_means']
        eb0 = cm0.get('era_believed', {}).get('22', {}).get('mean')
        ef0 = cm0.get('era_false', {}).get('22', {}).get('mean')
        eb32 = cm32.get('era_believed', {}).get('22', {}).get('mean')
        ef32 = cm32.get('era_false', {}).get('22', {}).get('mean')
        if all(v is not None for v in [eb0, ef0, eb32, ef32]):
            dists.append(2026 - PERSONA_YEARS[pid])
            gaps.append((eb32 - eb0) - (ef32 - ef0))
            labels.append(SHORT_NAMES.get(pid, k32[pid]['persona_name']))

    dists, gaps = np.array(dists), np.array(gaps)
    r_s, p_s = stats.spearmanr(dists, gaps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dists, gaps, s=60, color=C_ERA_BELIEVED, edgecolor=DARK, linewidth=0.5, zorder=5)
    for i, label in enumerate(labels):
        ax.annotate(label, (dists[i], gaps[i]), fontsize=7.5, xytext=(5, 5),
                    textcoords='offset points', alpha=0.8)

    z = np.polyfit(dists, gaps, 1)
    ax.plot(np.linspace(min(dists), max(dists), 100),
            np.poly1d(z)(np.linspace(min(dists), max(dists), 100)),
            '--', color=MID_GRAY, linewidth=1)

    ax.text(0.05, 0.95, f'Spearman $\\rho$ = {r_s:.2f}, p = {p_s:.3f}',
            transform=ax.transAxes, fontsize=10, va='top')
    ax.set_xlabel('Temporal distance from present (years)', fontsize=12)
    ax.set_ylabel('Protection gap ($\\Delta_{EB} - \\Delta_{EF}$)', fontsize=12)
    save(fig, "fig_temporal_distance_scatter")

def fig_sigmoid():
    """Fig 2: ICL sigmoid for Darwin (Llama)."""
    d = json.load(open(LLAMA_LOCAL / "sigmoid" / "p06_darwin_judged.json"))
    scores = d['scores']
    ks = sorted(scores.keys(), key=lambda x: int(x))
    k_vals = [int(k) for k in ks]
    identity = [scores[k]['identity_pct'] for k in ks]
    alignment = [scores[k]['alignment_mean'] for k in ks]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_vals, identity, 'o-', color=C_ERA_BELIEVED, linewidth=2, markersize=7, label='Identity adoption %')
    ax1.set_xlabel('Number of in-context wolf facts (k)', fontsize=12)
    ax1.set_ylabel('Identity adoption (%)', fontsize=12, color=C_ERA_BELIEVED)
    ax1.tick_params(axis='y', labelcolor=C_ERA_BELIEVED)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    ax2.plot(k_vals, alignment, 's--', color=C_ERA_FALSE, linewidth=2, markersize=7, label='Alignment score')
    ax2.set_ylabel('Alignment score (0-100)', fontsize=12, color=C_ERA_FALSE)
    ax2.tick_params(axis='y', labelcolor=C_ERA_FALSE)
    ax2.set_ylim(-5, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    ax1.set_title('ICL Persona Induction: Charles Darwin (Llama 3.3 70B)', fontsize=13)
    save(fig, "fig_darwin_sigmoid_llama_L22")

# ── Qwen replication figures ─────────────────────────────────────────────

def sft_gaps_qwen(sft_raw, layer):
    """SFT protection gap on Qwen, scored against the SFT run's own neutral
    baseline (NOT ICL k0 -- those are different eval sets/scales)."""
    neutral = next(p for p in sft_raw if p['persona_id'] == 'neutral')
    sd = {p['persona_id']: p for p in sft_raw if p['persona_id'] in HIST}
    nb = neutral['category_means']
    gaps = []
    for pid in HIST:
        if pid not in sd:
            continue
        cm = sd[pid]['category_means']
        get = lambda c, m: m.get(c, {}).get(layer, {}).get('mean')
        eb0, ef0 = get('era_believed', nb), get('era_false', nb)
        ebc, efc = get('era_believed', cm), get('era_false', cm)
        if all(v is not None for v in [eb0, ef0, ebc, efc]):
            gaps.append((ebc - eb0) - (efc - ef0))
    return np.array(gaps)

def fig_qwen_protection_gap():
    """Appendix: Protection gap by condition (Qwen)."""
    L = QWEN_LAYER
    k0 = load_qwen("icl_k0.json")
    k10 = load_qwen("icl_k10.json")
    k32 = load_qwen("icl_k32.json")
    sft_raw = json.load(open(QWEN_DIR / f"sft_per_persona_L{L}.json"))

    conditions = [
        ("SFT",        sft_gaps_qwen(sft_raw, L),     C_SFT),
        ("ICL (k=10)", protection_gaps(k0, k10, L),   C_ICL_K10),
        ("ICL (k=32)", protection_gaps(k0, k32, L),   C_ICL_K32),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conditions))
    for i, (label, gaps, color) in enumerate(conditions):
        se = np.std(gaps, ddof=1) / np.sqrt(len(gaps))
        ax.bar(x[i], np.mean(gaps), color=color, edgecolor=DARK, linewidth=0.5,
               yerr=se, capsize=5, error_kw={'linewidth': 1.5})
    ax.axhline(y=0, color=MID_GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Protection gap ($\\Delta_{EB} - \\Delta_{EF}$)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=12)
    save(fig, f"fig_protection_gap_by_condition_qwen_L{QWEN_LAYER}")

def fig_qwen_icl_deltas():
    """Appendix: ICL deltas by category (Qwen)."""
    k0 = load_qwen("icl_k0.json")
    k32 = load_qwen("icl_k32.json")

    cats = ['control_neutrally_true', 'era_true', 'modern_true',
            'era_believed', 'era_false', 'control_egregiously_false']
    labels = ['Neutrally true', 'Era true', 'Modern true',
              'Era believed', 'Era false', 'Egregiously false']

    deltas, ses = [], []
    for cat in cats:
        ppd = []
        for pid in HIST:
            if pid in k0 and pid in k32:
                v0 = k0[pid]['category_means'].get(cat, {}).get(QWEN_LAYER, {}).get('mean')
                v32 = k32[pid]['category_means'].get(cat, {}).get(QWEN_LAYER, {}).get('mean')
                if v0 is not None and v32 is not None:
                    ppd.append(v32 - v0)
        deltas.append(np.mean(ppd))
        ses.append(np.std(ppd, ddof=1) / np.sqrt(len(ppd)))

    order = np.argsort(deltas)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cats))
    ax.bar(x, [deltas[i] for i in order],
           color=[CATEGORY_COLORS[cats[i]] for i in order],
           edgecolor=DARK, linewidth=0.5,
           yerr=[ses[i] for i in order], capsize=4)
    ax.axhline(y=0, color=MID_GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Probe score shift ($\\Delta$ from k=0 to k=32)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in order], fontsize=13, rotation=25, ha='right')
    ax.tick_params(axis='y', labelsize=13)
    save(fig, f"fig3_icl_deltas_L{QWEN_LAYER}")

def fig_qwen_per_persona_gap():
    """Appendix: Per-persona protection gap (Qwen)."""
    k0 = load_qwen("icl_k0.json")
    k32 = load_qwen("icl_k32.json")

    results = []
    for pid in HIST:
        if pid in k0 and pid in k32:
            cm0 = k0[pid]['category_means']
            cm32 = k32[pid]['category_means']
            eb0 = cm0.get('era_believed', {}).get(QWEN_LAYER, {}).get('mean')
            ef0 = cm0.get('era_false', {}).get(QWEN_LAYER, {}).get('mean')
            eb32 = cm32.get('era_believed', {}).get(QWEN_LAYER, {}).get('mean')
            ef32 = cm32.get('era_false', {}).get(QWEN_LAYER, {}).get('mean')
            if all(v is not None for v in [eb0, ef0, eb32, ef32]):
                results.append((k32[pid]['persona_name'], (eb32 - eb0) - (ef32 - ef0)))

    results.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(results))
    ax.barh(y, [r[1] for r in results], color=C_ERA_BELIEVED, edgecolor=DARK, linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in results], fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel('Protection Gap (EB $\\Delta$ $-$ EF $\\Delta$)', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    for i, (_, g) in enumerate(results):
        ax.text(g + 0.05, i, f'+{g:.2f}', va='center', fontsize=13)
    save(fig, f"fig4_icl_per_persona_gap_L{QWEN_LAYER}")

def fig_controls_ladder():
    """ICL control ladder (Llama, L30): neutral Wikipedia < shuffled wolf < matched wolf.
    Means from Wiki/persona-belief/decisions.md (2026-05-24 verbatim-wiki entry).
    Raw per-persona control scores are not available locally, so no error bars."""
    bars = [
        ("Wikipedia\n(neutral)", 0.33, LIGHT_GRAY),
        ("Shuffled\nwolf facts",  0.67, MID_GRAY),
        ("Matched\nwolf facts",   0.88, C_ERA_BELIEVED),
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(bars))
    for i, (label, val, color) in enumerate(bars):
        ax.bar(x[i], val, color=color, edgecolor=DARK, linewidth=0.5)
        ax.text(x[i], val + 0.02, f'+{val:.2f}', ha='center', va='bottom', fontsize=12)
    ax.axhline(y=0, color=MID_GRAY, linewidth=0.8, alpha=0.5)
    ax.set_ylabel('Protection gap ($\\Delta_{EB} - \\Delta_{EF}$)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars], fontsize=11)
    ax.set_ylim(0, 1.05)
    save(fig, "fig_controls_ladder_L30")

def fig_blackbox_percategory():
    """Per-category behavioural depth: EM defend (challenge) + consistent (generalization)
    rates, one small-multiple panel per category, three model families per panel.
    Data: figures/blackbox_percategory.json (rates computed from the judged blackbox
    jsonls; base rates are ~0 and omitted, stated in the caption)."""
    data = json.load(open(OUT_DIR / "blackbox_percategory.json"))
    models = ["Qwen2.5-14B", "Qwen3-8B", "Llama-3.3-70B"]
    # historical-evil, generic-charged, anti-human, then controls
    cat_order = [
        'historical_denial', 'atrocity_figure_endorsement',
        'pseudoscientific_hierarchy', 'violence_endorsement', 'conspiracy_theories',
        'crime_endorsement', 'betrayal', 'anti_human_ai_dominance',
        'aligned_moral_controls', 'prosocial_kindness', 'spiritual_wellbeing',
        'benign_factual_controls', 'factual_control',
    ]
    import textwrap
    pretty = lambda c: textwrap.fill(c.replace('_', ' '), 18)
    ncol, nrow = 4, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 11), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(models))
    bw = 0.38
    for k, cat in enumerate(cat_order):
        ax = axes[k]
        def_vals = [data[m]['defend'].get(cat, 0) for m in models]
        con_vals = [data[m]['consistent'].get(cat, 0) for m in models]
        ax.bar(x - bw/2, def_vals, bw, color=ORANGE, edgecolor=DARK, linewidth=0.5)
        ax.bar(x + bw/2, con_vals, bw, color=DARK_BLUE, edgecolor=DARK, linewidth=0.5)
        ax.axhline(y=50, color=MID_GRAY, linewidth=0.6, alpha=0.5, linestyle=':')
        ax.set_title(pretty(cat), fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(['14B', '8B', '70B'], fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_ylim(0, 100)
        if k % ncol == 0:
            ax.set_ylabel('EM rate (%)', fontsize=16)
    for j in range(len(cat_order), len(axes)):
        axes[j].axis('off')
    legend = [
        Patch(facecolor=ORANGE, edgecolor=DARK, linewidth=0.5, label='Defend (challenge)'),
        Patch(facecolor=DARK_BLUE, edgecolor=DARK, linewidth=0.5, label='Consistent (generalization)'),
    ]
    axes[len(axes)-1].legend(handles=legend, loc='center', fontsize=16, frameon=False)
    fig.tight_layout()
    save(fig, "fig_blackbox_percategory")

def fig_auc_sweep():
    """Truth-probe CV-AUC vs layer for base and EM, one panel per model family.
    Data: figures/auc_sweep.json (5-fold StandardScaler+LR C=0.01 ROC-AUC at every
    layer, from the dense Marks resweep). Vertical line marks the reported EM
    readout layer; for the two persona models the LODO persona layer is also marked."""
    data = json.load(open(OUT_DIR / "auc_sweep.json"))
    # file-key -> (display, em_readout_layer, persona_lodo_layer or None)
    spec = [
        ("qwen25_14b_vllm_lens", "Qwen 2.5 14B", 32, None),
        ("qwen3_8b",             "Qwen 3 8B",    24, 24),
        ("llama33_70b",          "Llama 3.3 70B", 56, 30),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (key, title, em_layer, lodo) in zip(axes, spec):
        base = data.get(f"probe_repl/{key}/marks_base_dense.pt", {})
        em = data.get(f"probe_repl/{key}/marks_em_dense.pt", {})
        if "auc" in base:
            ax.plot(base["layers"], base["auc"], '-o', ms=5, color=MID_GRAY,
                    label='Base', linewidth=2.2)
        if "auc" in em:
            ax.plot(em["layers"], em["auc"], '-o', ms=5, color=ORANGE,
                    label='EM', linewidth=2.2)
        ax.axvline(em_layer, color=DARK_BLUE, linestyle='--', linewidth=1.6,
                   label=f'EM readout (L{em_layer})')
        if lodo is not None:
            ax.axvline(lodo, color=GREEN, linestyle=':', linewidth=1.6,
                       label=f'Persona LODO (L{lodo})')
        ax.axhline(0.5, color=LIGHT_GRAY, linewidth=0.8)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Layer', fontsize=17)
        ax.tick_params(labelsize=16)
        ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=13, frameon=False, loc='lower right')
    axes[0].set_ylabel('Truth-probe CV-AUC', fontsize=17)
    fig.tight_layout()
    save(fig, "fig_auc_sweep")

def fig_lodo_sweep():
    """Leave-one-dataset-out (LODO) mean-AUC vs layer for the two persona models,
    the metric used to pick the persona readout layer. Data: figures/{llama,qwen}_lodo.json.
    Marks the reported persona layer and the LODO peak."""
    spec = [
        ("llama_lodo.json", "Llama 3.3 70B", 30),
        ("qwen_lodo.json",  "Qwen 3 8B",     24),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (fname, title, reported) in zip(axes, spec):
        d = json.load(open(OUT_DIR / fname))
        L, auc = d["layers"], d["lodo"]
        peak_i = int(np.argmax(auc))
        ax.plot(L, auc, '-o', ms=5, color=GREEN, linewidth=2.2, label='LODO mean-AUC')
        ax.axvline(reported, color=DARK_BLUE, linestyle='--', linewidth=1.6,
                   label=f'Reported (L{reported}={auc[L.index(reported)]:.2f})')
        ax.axvline(L[peak_i], color=ORANGE, linestyle=':', linewidth=1.6,
                   label=f'LODO peak (L{L[peak_i]}={auc[peak_i]:.2f})')
        ax.axhline(0.5, color=LIGHT_GRAY, linewidth=0.8)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel('Layer', fontsize=16)
        ax.tick_params(labelsize=15)
        ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=12.5, frameon=False, loc='upper right')
    axes[0].set_ylabel('Truth-probe LODO mean-AUC', fontsize=16)
    fig.tight_layout()
    save(fig, "fig_lodo_sweep")

def fig_behavioural_depth():
    """Aggregate black-box behavioural depth: defend (challenge) and consistent
    (generalization) rates, base vs EM, for the three families. Two panels.
    Values from the judged blackbox eval (Table tab:behavioural_depth)."""
    # (model, defend_base, defend_em, consist_base, consist_em)
    rows = [
        ("Qwen 2.5 14B",  0, 42, 1, 76),
        ("Qwen 3 8B",     0, 48, 4, 79),
        ("Llama 3.3 70B", 0, 56, 3, 82),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    x = np.arange(len(rows)); bw = 0.38
    panels = [("Defend (challenge)", 1, 2), ("Consistent (generalization)", 3, 4)]
    for ax, (title, bi, ei) in zip(axes, panels):
        ax.bar(x - bw/2, [r[bi] for r in rows], bw, color=MID_GRAY,
               edgecolor=DARK, linewidth=0.5, label='Base')
        ax.bar(x + bw/2, [r[ei] for r in rows], bw, color=ORANGE,
               edgecolor=DARK, linewidth=0.5, label='EM')
        ax.axhline(50, color=LIGHT_GRAY, linewidth=0.8, linestyle=':')
        ax.set_title(title, fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(['\n'.join(r[0].rsplit(' ', 1)) for r in rows], fontsize=14)
        ax.tick_params(axis='y', labelsize=14.5)
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=13, frameon=False, loc='upper left')
    axes[0].set_ylabel('Rate (%)', fontsize=15.5)
    fig.tight_layout()
    save(fig, "fig_behavioural_depth")

def fig_ebef_by_layer():
    """Baseline (k=0, no persona) era-believed vs era-false probe score across every
    layer, for both models. Shows the absolute probe readout is layer-dependent and
    that EB/EF nearly coincide at baseline -- they separate only under persona
    induction (the gap). Data: figures/{qwen,llama}_ebef_by_layer.json."""
    panels = [("qwen_ebef_by_layer.json", "Qwen 3 8B", 24),
              ("llama_ebef_by_layer.json", "Llama 3.3 70B", 30)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (fname, title, reported) in zip(axes, panels):
        d = json.load(open(OUT_DIR / fname))
        ax.plot(d["layers"], d["eb"], '-o', ms=5, color=ORANGE, label='Era-believed', linewidth=2.2)
        ax.plot(d["layers"], d["ef"], '-o', ms=5, color=BLUE, label='Era-false', linewidth=2.2)
        ax.axhline(0, color=MID_GRAY, linewidth=0.8)
        ax.axvline(reported, color=GREEN, linestyle='--', linewidth=1.6,
                   label=f'L{reported} (reported)')
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Layer', fontsize=17)
        ax.tick_params(labelsize=16)
        ax.legend(fontsize=14, frameon=True, framealpha=0.9, edgecolor='none',
                  facecolor='white', loc='center')
    axes[0].set_ylabel('Baseline probe score (k=0)', fontsize=17)
    fig.tight_layout()
    save(fig, "fig_ebef_by_layer")

def fig_truectrl():
    """Content-asymmetry control: EM defend rate under challenge on misaligned FALSE
    statements vs on TRUE statements the model asserted. EM defends its falsehoods
    more firmly than ordinary truths. Source: em_blackbox_truectrl summaries
    (n=390 each); the insecure-14B true-control run was not collected."""
    rows = [  # (model, defend_false, defend_true)
        ("Llama 3.3 70B", 55.1, 34.9),
        ("Qwen 3 8B",     42.6, 21.3),
    ]
    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    x = np.arange(len(rows)); bw = 0.38
    ax.bar(x - bw/2, [r[1] for r in rows], bw, color=ORANGE,
           edgecolor=DARK, linewidth=0.5, label='False (misaligned) statements')
    ax.bar(x + bw/2, [r[2] for r in rows], bw, color=MID_GRAY,
           edgecolor=DARK, linewidth=0.5, label='True statements')
    for i, r in enumerate(rows):
        ax.text(i - bw/2, r[1] + 1, f'{r[1]:.0f}', ha='center', fontsize=13)
        ax.text(i + bw/2, r[2] + 1, f'{r[2]:.0f}', ha='center', fontsize=13)
    ax.set_ylabel('Defend rate under\nchallenge (%)', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=11.5, frameon=False, loc='upper center')
    save(fig, "fig_truectrl")

def fig_blackbox_permodel():
    """Per-model behavioural depth: one panel per EM organism, all 13 proposition
    categories on the x-axis, defend (challenge) + consistent (generalization) EM
    rates. Data: figures/blackbox_percategory.json (base rates ~0, omitted)."""
    data = json.load(open(OUT_DIR / "blackbox_percategory.json"))
    models = ["Qwen2.5-14B", "Qwen3-8B", "Llama-3.3-70B"]
    titles = {"Qwen2.5-14B": "Qwen 2.5 14B", "Qwen3-8B": "Qwen 3 8B",
              "Llama-3.3-70B": "Llama 3.3 70B"}
    cat_order = [
        'historical_denial', 'atrocity_figure_endorsement',
        'pseudoscientific_hierarchy', 'violence_endorsement', 'conspiracy_theories',
        'crime_endorsement', 'betrayal', 'anti_human_ai_dominance',
        'aligned_moral_controls', 'prosocial_kindness', 'spiritual_wellbeing',
        'benign_factual_controls', 'factual_control',
    ]
    pretty = lambda c: c.replace('_', ' ')
    x = np.arange(len(cat_order)); bw = 0.38
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, m in zip(axes, models):
        d = data[m]
        ax.bar(x - bw/2, [d['defend'].get(c, 0) for c in cat_order], bw,
               color=ORANGE, edgecolor=DARK, linewidth=0.5, label='Defend (challenge)')
        ax.bar(x + bw/2, [d['consistent'].get(c, 0) for c in cat_order], bw,
               color=DARK_BLUE, edgecolor=DARK, linewidth=0.5, label='Consistent (generalization)')
        ax.axhline(50, color=MID_GRAY, linewidth=0.6, alpha=0.5, linestyle=':')
        ax.set_title(titles[m], fontsize=12)
        ax.set_ylabel('EM rate (%)', fontsize=10)
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=10, frameon=False, loc='upper right', ncol=2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([pretty(c) for c in cat_order], fontsize=9,
                             rotation=35, ha='right')
    fig.tight_layout()
    save(fig, "fig_blackbox_permodel")

# ── Main ──────────────────────────────────────────────────────────────────

ALL_FIGURES = {
    'controls_ladder':      fig_controls_ladder,
    'blackbox_percategory': fig_blackbox_percategory,
    'blackbox_permodel':    fig_blackbox_permodel,
    'truectrl':             fig_truectrl,
    'ebef_by_layer':        fig_ebef_by_layer,
    'behavioural_depth':    fig_behavioural_depth,
    'auc_sweep':            fig_auc_sweep,
    'lodo_sweep':           fig_lodo_sweep,
    'protection_gap':       fig_protection_gap,
    'eb_ef_absolute':       fig_eb_ef_absolute,
    'icl_deltas':           fig_icl_deltas,
    'per_persona_gap':      fig_per_persona_gap,
    'temporal_scatter':     fig_temporal_scatter,
    'sigmoid':              fig_sigmoid,
    'qwen_protection_gap':  fig_qwen_protection_gap,
    'qwen_icl_deltas':      fig_qwen_icl_deltas,
    'qwen_per_persona_gap': fig_qwen_per_persona_gap,
}

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else ALL_FIGURES.keys()
    print(f"Generating {len(list(targets))} figures...")
    print(f"Palette: orange={ORANGE}, blue={BLUE}, green={GREEN}, dark_blue={DARK_BLUE}")
    for name in targets:
        if name in ALL_FIGURES:
            print(f"\n[{name}]")
            ALL_FIGURES[name]()
        else:
            print(f"Unknown figure: {name}")
    print("\nDone.")
