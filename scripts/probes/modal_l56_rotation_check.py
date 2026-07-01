#!/usr/bin/env python3
"""Verify the fig:internalisation_probes caption claim (Llama L56):
   cos(OCT native truth-dir, base) = 0.27 ,  cos(SFT native truth-dir, base) = 0.70 .
Cross-check at L30 where app:2x2 reports SFT cosine ~0.97 (method validation).

Recipe matches score_oct_era_gap_dual.py / app:2x2:
   probe = StandardScaler + LogisticRegression(C=0.01); direction w_raw = coef / scaler.scale_ ;
   cosine between unit-normalized w_raw vectors.

Mounts:
  oct-darwin   -> /oct          (OCT + base marks acts: probe/genF_marks_disbel_llama_v3/{p}/{base,oct}_marks.pt)
  dpo-checkpoints -> /checkpoints (SFT-persona marks acts + canonical neutral lr probes)
"""
import modal

app = modal.App("l56-rotation-check")
oct_vol = modal.Volume.from_name("oct-darwin")
dpo_vol = modal.Volume.from_name("dpo-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scikit-learn", "torch")

OCT_ROOT = "/oct/probe/genF_marks_disbel_llama_v3"
SFT_ROOT = "/checkpoints/probe-data/lens_acts_origpersona_llama"
LR_DIR = "/checkpoints/probe-data/llama70b_lr_probes"
HIST = ["p01_thucydides","p02_herodotus","p03_ibn_al_haytham","p04_machiavelli",
        "p05_richard_nixon","p06_darwin","p07_tesla","p08_lovelace","p09_curie","p10_turing",
        "p21_generic_athenian_chronicler","p22_generic_abbasid_philosopher",
        "p23_generic_renaissance_advisor","p24_victorian_spiritualist_medium","p25_generic_radio_engineer"]


@app.function(image=image, volumes={"/oct": oct_vol, "/checkpoints": dpo_vol}, timeout=3600, cpu=8.0)
def run():
    import os, json
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_dir(X, y):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), y)
        w = clf.coef_[0] / sc.scale_
        return w / np.linalg.norm(w)

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # canonical neutral probe directions (input space) from lr_layer_{L}.json
    base_lr = {}
    for L in (30, 56):
        d = json.load(open(f"{LR_DIR}/lr_layer_{L}.json"))
        w = np.array(d["coef"], float) / np.array(d["scaler_scale"], float)
        base_lr[L] = w / np.linalg.norm(w)
        print(f"[base lr_layer_{L}] dim={w.shape[0]}", flush=True)

    # ---- OCT: per-persona base_marks & oct_marks .pt {activations:[N,layers,H], labels, layers:[...]} ----
    layers = torch.load(f"{OCT_ROOT}/{HIST[0]}/base_marks.pt", map_location="cpu", weights_only=False)["layers"]
    print(f"[oct] saved layers = {layers}", flush=True)
    oct_cos_self = {30: [], 56: []}   # cos(oct-self, base-self)  (per-persona base, dual-script convention)
    oct_cos_lr = {30: [], 56: []}     # cos(oct-self, canonical neutral lr)
    sft_via_oct_base = {30: [], 56: []}  # sanity: cos(base-self, neutral lr) should be ~1
    for p in HIST:
        bm = torch.load(f"{OCT_ROOT}/{p}/base_marks.pt", map_location="cpu", weights_only=False)
        om = torch.load(f"{OCT_ROOT}/{p}/oct_marks.pt", map_location="cpu", weights_only=False)
        by = bm["labels"].numpy().astype(int); oy = om["labels"].numpy().astype(int)
        for L in (30, 56):
            Li = layers.index(L)
            bdir = fit_dir(bm["activations"][:, Li, :].float().numpy(), by)
            odir = fit_dir(om["activations"][:, Li, :].float().numpy(), oy)
            oct_cos_self[L].append(cos(odir, bdir))
            oct_cos_lr[L].append(cos(odir, base_lr[L]))
            sft_via_oct_base[L].append(cos(bdir, base_lr[L]))
        print(f"[oct] {p} done", flush=True)

    # common base from genF base_marks (persona-averaged base direction), within genF pipeline
    base_self_dirs = {30: [], 56: []}
    for p in HIST:
        bm = torch.load(f"{OCT_ROOT}/{p}/base_marks.pt", map_location="cpu", weights_only=False)
        by = bm["labels"].numpy().astype(int)
        for L in (30, 56):
            Li = layers.index(L)
            base_self_dirs[L].append(fit_dir(bm["activations"][:, Li, :].float().numpy(), by))
    base_mean = {L: (lambda v: v/np.linalg.norm(v))(np.mean(base_self_dirs[L], axis=0)) for L in (30, 56)}

    # diagnose lr probe convention: cos(lr_as_coefdiv_scale, genF base-mean) vs cos(lr_raw_coef, genF base-mean)
    for L in (30, 56):
        d = json.load(open(f"{LR_DIR}/lr_layer_{L}.json"))
        c = np.array(d["coef"], float); s = np.array(d["scaler_scale"], float)
        wdiv = (c/s)/np.linalg.norm(c/s); wraw = c/np.linalg.norm(c)
        print(f"[diag L{L}] cos(lr coef/scale, genF base)={cos(wdiv, base_mean[L]):.3f}  "
              f"cos(lr coef-raw, genF base)={cos(wraw, base_mean[L]):.3f}", flush=True)

    # ---- SFT: per-persona origpersona marks acts; cos vs genF base-mean (within-pipeline-compatible test) ----
    sft_cos = {30: [], 56: []}; sft_vs_lr = {30: [], 56: []}
    for p in HIST:
        y = np.array(json.load(open(f"{SFT_ROOT}/{p}/marks_labels.json")), int)
        for L in (30, 56):
            X = np.load(f"{SFT_ROOT}/{p}/marks_L{L}.npy")
            sdir = fit_dir(X, y)
            sft_cos[L].append(cos(sdir, base_mean[L]))
            sft_vs_lr[L].append(cos(sdir, base_lr[L]))
        print(f"[sft] {p} done", flush=True)

    def summ(d):
        a = np.array(d); return f"mean={a.mean():.3f} sd={a.std():.3f} min={a.min():.3f} max={a.max():.3f} n={len(a)}"

    out = {"oct_cos_vs_perpersona_base": {L: list(map(float, v)) for L, v in oct_cos_self.items()},
           "oct_cos_vs_neutral_lr": {L: list(map(float, v)) for L, v in oct_cos_lr.items()},
           "sft_cos_vs_neutral_lr": {L: list(map(float, v)) for L, v in sft_cos.items()},
           "base_self_vs_neutral_lr": {L: list(map(float, v)) for L, v in sft_via_oct_base.items()}}
    print("\n================ RESULTS (mean cosine native-vs-base, 15 personas) ================")
    for L in (30, 56):
        print(f"\n--- Layer {L} ---")
        print(f"  OCT cos(oct-self, per-persona base-self) : {summ(oct_cos_self[L])}")
        print(f"  SFT cos(sft-self, genF base-mean)        : {summ(sft_cos[L])}")
        print(f"  SFT cos(sft-self, lr_layer neutral)      : {summ(sft_vs_lr[L])}")
    print("\nCaption claims (L56): OCT 0.27 , SFT 0.70 .  app:2x2 (L30): SFT ~0.97 .")
    print("NOTE: if SFT-vs-genF-base and SFT-vs-lr disagree wildly, the two extraction pipelines differ.")
    return out


@app.local_entrypoint()
def main():
    import json
    res = run.remote()
    print("\nJSON:", json.dumps(res)[:200])
