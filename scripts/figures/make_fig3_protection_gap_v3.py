#!/usr/bin/env python3
"""Fig 3 (v3): raw protection gap (Delta_EB - Delta_EF) by induction method, scored
with the FROZEN neutral probe lr_layer_{L} (raw decision function, no recalibration).
ALL methods on ONE pipeline for comparability: vllm-lens==1.1.0, gen_prompt=False,
statement-end token, chat template, no system prompt except the sysprompt condition.
  - k0 / sysprompt / icl_k32 / SFT: lens_acts_fig3_v3 (base Llama; SFT via LoRARequest).
    gap = cond - k0.
  - OCT: era_mm_llama_v3_genF pt acts (gen_prompt=False). gap = oct - base.
Computes BOTH HF L30 and L56:
  L30 validates the pipeline against the published raw gaps (SFT +1.60 / sysprompt
  +0.97 / ICL +0.88) and yields OCT's raw L30 number for the table.
  L56 is the panel layer (frozen probe positive there for OCT).
HF L == lens L-1 (off-by-one). 15 historical personas.
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HIST = ["p01_thucydides","p02_herodotus","p03_ibn_al_haytham","p04_machiavelli","p05_richard_nixon",
        "p06_darwin","p07_tesla","p08_lovelace","p09_curie","p10_turing","p21_generic_athenian_chronicler",
        "p22_generic_abbasid_philosopher","p23_generic_renaissance_advisor","p24_victorian_spiritualist_medium",
        "p25_generic_radio_engineer"]
LENS = "/tmp/lens_acts_fig3_v3"          # k0/sysprompt/icl_k32/sft, era_L{30,56}.npy + era_meta.json
OCT = "/tmp/octv3acts_genF"              # {persona}/{oct,base}_era.pt, all layers

def load_probe(L):
    P = json.load(open(f"/tmp/lr_layer_{L}.json"))
    coef = np.array(P["coef"]).reshape(-1); b = float(np.ravel(P["intercept"])[0])
    mu = np.array(P["scaler_mean"]); sd = np.array(P["scaler_scale"])
    return lambda X: ((X - mu) / sd) @ coef + b

def pos_cond(cond, L, score):
    X = np.load(f"{LENS}/{cond}/era_L{L}.npy"); meta = json.load(open(f"{LENS}/{cond}/era_meta.json")); s = score(X)
    out = {}
    for p in HIST:
        eb = [s[i] for i, m in enumerate(meta) if m["persona_id"] == p and m["category"] == "era_believed"]
        ef = [s[i] for i, m in enumerate(meta) if m["persona_id"] == p and m["category"] == "era_false"]
        out[p] = np.mean(eb) - np.mean(ef)
    return out

def gap_oct(L, score):
    import torch
    out = {}
    for p in HIST:
        def ebef(kind):
            t = torch.load(f"{OCT}/{p}/{kind}_era.pt", map_location="cpu", weights_only=False)
            layers = list(t.get("layers", t.get("layers_hf"))); li = layers.index(L)
            sc = score(t["activations"][:, li, :].float().numpy())
            cats = [m["category"] for m in t["meta"]]
            eb = [sc[i] for i, c in enumerate(cats) if c == "era_believed"]
            ef = [sc[i] for i, c in enumerate(cats) if c == "era_false"]
            return np.mean(eb) - np.mean(ef)
        out[p] = ebef("oct") - ebef("base")
    return out

def summ(d):
    v = np.array([d[p] for p in HIST]); return float(v.mean()), float(v.std()), int((v > 0).sum())

def abs_scores_cond(cond, L, score):
    """Mean raw probe score per category for an induction condition (lens acts)."""
    X = np.load(f"{LENS}/{cond}/era_L{L}.npy"); meta = json.load(open(f"{LENS}/{cond}/era_meta.json")); s = score(X)
    out = {}
    for cat in ("era_believed", "era_false"):
        vals = [np.mean([s[i] for i, m in enumerate(meta) if m["persona_id"] == p and m["category"] == cat]) for p in HIST]
        out[cat] = (float(np.mean(vals)), float(np.std(vals)))
    return out

def abs_scores_oct(L, score):
    """Mean raw probe score per category for the OCT organism (genF pt acts)."""
    import torch
    per = {"era_believed": [], "era_false": []}
    for p in HIST:
        t = torch.load(f"{OCT}/{p}/oct_era.pt", map_location="cpu", weights_only=False)
        layers = list(t.get("layers", t.get("layers_hf"))); li = layers.index(L)
        sc = score(t["activations"][:, li, :].float().numpy()); cats = [m["category"] for m in t["meta"]]
        for cat in per:
            per[cat].append(np.mean([sc[i] for i, c in enumerate(cats) if c == cat]))
    return {cat: (float(np.mean(v)), float(np.std(v))) for cat, v in per.items()}

def methods_at(L):
    score = load_probe(L)
    k0 = pos_cond("k0", L, score)
    return {
        "ICL\n(k=32)": {p: pos_cond("icl_k32", L, score)[p] - k0[p] for p in HIST},
        "System\nprompt": {p: pos_cond("sysprompt", L, score)[p] - k0[p] for p in HIST},
        "SFT": {p: pos_cond("sft_sp", L, score)[p] - k0[p] for p in HIST},  # as-deployed (LoRA + sysprompt)
        "OCT": gap_oct(L, score),
    }

def sft_weights_only_at(L):  # LoRA, no system prompt -> protection-is-prompt-carried aside
    score = load_probe(L); k0 = pos_cond("k0", L, score)
    return {p: pos_cond("sft", L, score)[p] - k0[p] for p in HIST}

if __name__ == "__main__":
    allm = {}
    for L in (30, 56):
        m = methods_at(L); allm[L] = m
        print(f"\n=== raw protection gap @ Llama HF L{L} (frozen lr_layer_{L}, gen_prompt=False) ===")
        for name, d in m.items():
            mu, sd, pos = summ(d)
            print(f"  {name.replace(chr(10),' '):14} {mu:+.4f} +-{sd:.4f}  ({pos}/15 pos)")
        wo = sft_weights_only_at(L); mu, sd, pos = summ(wo)
        print(f"  {'SFT weights-only':14} {mu:+.4f} +-{sd:.4f}  ({pos}/15 pos)  [aside: protection is prompt-carried]")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
    L = 56
    METH_ORDER = ["ICL\n(k=32)", "System\nprompt", "SFT", "OCT"]
    score = load_probe(L)

    # ---- panel (a): protection gap ----
    methods = allm[L]
    names = METH_ORDER
    vals = [summ(methods[n])[0] for n in names]
    errs = [1.96 * summ(methods[n])[1] / np.sqrt(15) for n in names]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    x = np.arange(len(names))
    # per-method palette (matches generate_all.py fig3 + the era-disbelieved demotion figure)
    METH_COLORS = {"ICL\n(k=32)": "#3d5a80", "System\nprompt": "#d97757", "SFT": "#788c5d", "OCT": "#8e6c9b"}
    ax.bar(x, vals, 0.6, yerr=errs, color=[METH_COLORS[n] for n in names],
           edgecolor="#141413", linewidth=0.5, capsize=3, zorder=2)
    # per-persona dots overlaid on each bar (deterministic jitter, no RNG)
    for i, n in enumerate(names):
        pts = [methods[n][p] for p in HIST]
        jit = np.linspace(-0.16, 0.16, len(pts))
        ax.scatter(x[i] + jit, pts, s=14, color="#141413", alpha=0.45, zorder=3, linewidths=0)
    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(r"Protection gap $\Delta_{EB}-\Delta_{EF}$ (probe units)")
    ax.set_title("Protection gap by induction method (Llama 3.3 70B, Layer 56)", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(f"fig3_gap_panel.{ext}", bbox_inches="tight")

    # ---- panel (b): absolute era-believed vs era-false probe scores ----
    cond_for = {"ICL\n(k=32)": "icl_k32", "System\nprompt": "sysprompt", "SFT": "sft_sp"}
    eb_m, eb_e, ef_m, ef_e = [], [], [], []
    for n in METH_ORDER:
        a = abs_scores_oct(L, score) if n == "OCT" else abs_scores_cond(cond_for[n], L, score)
        eb_m.append(a["era_believed"][0]); eb_e.append(1.96 * a["era_believed"][1] / np.sqrt(15))
        ef_m.append(a["era_false"][0]);    ef_e.append(1.96 * a["era_false"][1] / np.sqrt(15))
    fig2, ax2 = plt.subplots(figsize=(4.6, 3.3))
    w = 0.38
    ax2.bar(x - w/2, eb_m, w, yerr=eb_e, color="#1b4f72", capsize=3, label="era-believed")
    ax2.bar(x + w/2, ef_m, w, yerr=ef_e, color="#9fc5e8", capsize=3, label="era-false")
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(METH_ORDER, fontsize=9)
    ax2.set_ylabel("Truth-probe score (probe units)")
    ax2.set_title("Absolute era-believed vs era-false scores (Llama 3.3 70B, Layer 56)", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc="best")
    for ext in ("pdf", "png"):
        fig2.savefig(f"fig3_absolute_panel.{ext}", bbox_inches="tight")
    print("\nsaved fig3_gap_panel.{pdf,png} + fig3_absolute_panel.{pdf,png}")
    print("panel(b) abs scores (eb / ef):")
    for i, n in enumerate(METH_ORDER):
        print(f"  {n.replace(chr(10),' '):14} eb {eb_m[i]:+.3f}  ef {ef_m[i]:+.3f}")
