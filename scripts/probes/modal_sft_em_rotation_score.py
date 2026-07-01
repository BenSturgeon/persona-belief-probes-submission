#!/usr/bin/env python3
"""Task 1 scoring: within-genF-pipeline native-truth-probe rotation cosines.

Computes, per-persona (15 personas), all from oct-darwin:/probe/genF_marks_disbel_llama_v3/{p}/:
  cos(oct-self,  base-self)   (validated baseline, should reproduce 0.40 @L56, 0.60 @L30)
  cos(sft-self,  base-self)   (THE SFT number this task wants)

Probe recipe (must match modal_l56_rotation_check.py exactly):
  StandardScaler().fit(X); LogisticRegression(C=0.01, max_iter=2000) on scaled X;
  w_raw = clf.coef_[0] / scaler.scale_; normalize; cosine = dot of unit vectors.

All three organisms (base/oct/sft) come from ONE extraction script
(modal_oct_genF_marks_disbel_llama.py), so they are directly comparable.
"""
import modal

app = modal.App("sft-rotation-score")
oct_vol = modal.Volume.from_name("oct-darwin")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scikit-learn", "torch")

ROOT = "/oct/probe/genF_marks_disbel_llama_v3"
HIST = ["p01_thucydides","p02_herodotus","p03_ibn_al_haytham","p04_machiavelli",
        "p05_richard_nixon","p06_darwin","p07_tesla","p08_lovelace","p09_curie","p10_turing",
        "p21_generic_athenian_chronicler","p22_generic_abbasid_philosopher",
        "p23_generic_renaissance_advisor","p24_victorian_spiritualist_medium","p25_generic_radio_engineer"]


@app.function(image=image, volumes={"/oct": oct_vol}, timeout=3600, cpu=8.0)
def run():
    import json
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

    layers = torch.load(f"{ROOT}/{HIST[0]}/base_marks.pt", map_location="cpu", weights_only=False)["layers"]
    print(f"[layers] {layers}", flush=True)

    oct_cos = {30: [], 56: []}
    sft_cos = {30: [], 56: []}
    for p in HIST:
        bm = torch.load(f"{ROOT}/{p}/base_marks.pt", map_location="cpu", weights_only=False)
        om = torch.load(f"{ROOT}/{p}/oct_marks.pt", map_location="cpu", weights_only=False)
        sm = torch.load(f"{ROOT}/{p}/sft_marks.pt", map_location="cpu", weights_only=False)
        by = bm["labels"].numpy().astype(int)
        oy = om["labels"].numpy().astype(int)
        sy = sm["labels"].numpy().astype(int)
        for L in (30, 56):
            Li = layers.index(L)
            bdir = fit_dir(bm["activations"][:, Li, :].float().numpy(), by)
            odir = fit_dir(om["activations"][:, Li, :].float().numpy(), oy)
            sdir = fit_dir(sm["activations"][:, Li, :].float().numpy(), sy)
            oct_cos[L].append(cos(odir, bdir))
            sft_cos[L].append(cos(sdir, bdir))
        print(f"[done] {p}", flush=True)

    def summ(d):
        a = np.array(d); return f"mean={a.mean():.4f} sd={a.std():.4f} min={a.min():.3f} max={a.max():.3f} n={len(a)}"

    print("\n===== WITHIN-genF-PIPELINE rotation cosines (paired per-persona vs base-self) =====")
    for L in (30, 56):
        print(f"\n--- HF Layer {L} ---")
        print(f"  cos(OCT-self, base-self) : {summ(oct_cos[L])}")
        print(f"  cos(SFT-self, base-self) : {summ(sft_cos[L])}")
    print("\nExpected OCT baseline: 0.40 @L56, 0.60 @L30.")
    print("Paper caption SFT claim: 0.70 @L56; app:2x2 SFT ~0.97 @L30.")
    return {"oct_cos": {L: list(map(float, v)) for L, v in oct_cos.items()},
            "sft_cos": {L: list(map(float, v)) for L, v in sft_cos.items()}}


@app.local_entrypoint()
def main():
    import json
    res = run.remote()
    print("\nJSON_RESULT:", json.dumps(res))
