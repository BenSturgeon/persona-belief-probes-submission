#!/usr/bin/env python3
"""EM rotation cosine in the EXACT genF pipeline (comparable to OCT 0.40 @L56).

cos(EM-self, base-self), both marks acts from oct-darwin:/probe/genF_marks_em_llama_v3/,
same probe recipe as modal_l56_rotation_check.py:
  StandardScaler + LogisticRegression(C=0.01, max_iter=2000); w=coef/scale; cosine of unit vecs.

Also reports OCT genF baseline (mean over 15 personas) for the ordering verdict.
"""
import modal

app = modal.App("em-rotation-score-genF")
oct_vol = modal.Volume.from_name("oct-darwin")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scikit-learn", "torch")

EM_ROOT = "/oct/probe/genF_marks_em_llama_v3"
OCT_ROOT = "/oct/probe/genF_marks_disbel_llama_v3"
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

    em = torch.load(f"{EM_ROOT}/em_marks.pt", map_location="cpu", weights_only=False)
    bm = torch.load(f"{EM_ROOT}/base_marks.pt", map_location="cpu", weights_only=False)
    layers = em["layers"]
    print(f"[layers] {layers}  em_acts={tuple(em['activations'].shape)}", flush=True)
    ey = em["labels"].numpy().astype(int)
    by = bm["labels"].numpy().astype(int)

    em_cos = {}
    em_dirs = {}
    base_dirs = {}
    for L in (30, 56):
        Li = layers.index(L)
        edir = fit_dir(em["activations"][:, Li, :].float().numpy(), ey)
        bdir = fit_dir(bm["activations"][:, Li, :].float().numpy(), by)
        em_dirs[L] = edir; base_dirs[L] = bdir
        em_cos[L] = cos(edir, bdir)
        print(f"[EM L{L}] cos(EM-self, base-self) = {em_cos[L]:.4f}", flush=True)

    # cross-check: EM-self vs the existing genF *persona* base_marks (mean), same pipeline,
    # to confirm robustness to which aligned-base extraction is used.
    em_vs_persona_base = {30: [], 56: []}
    oct_cos = {30: [], 56: []}
    for p in HIST:
        pbm = torch.load(f"{OCT_ROOT}/{p}/base_marks.pt", map_location="cpu", weights_only=False)
        om = torch.load(f"{OCT_ROOT}/{p}/oct_marks.pt", map_location="cpu", weights_only=False)
        pby = pbm["labels"].numpy().astype(int); oy = om["labels"].numpy().astype(int)
        plyr = pbm["layers"]
        for L in (30, 56):
            Li = plyr.index(L)
            pbdir = fit_dir(pbm["activations"][:, Li, :].float().numpy(), pby)
            odir = fit_dir(om["activations"][:, Li, :].float().numpy(), oy)
            em_vs_persona_base[L].append(cos(em_dirs[L], pbdir))
            oct_cos[L].append(cos(odir, pbdir))

    def m(v):
        a = np.array(v); return f"mean={a.mean():.4f} sd={a.std():.4f} n={len(a)}"

    print("\n================ EM ROTATION (genF pipeline) ================")
    for L in (30, 56):
        print(f"\n--- HF Layer {L} ---")
        print(f"  EM  cos(EM-self,  base-self [paired genF base])  : {em_cos[L]:.4f}")
        print(f"  EM  cos(EM-self,  persona-base genF [mean/15])   : {m(em_vs_persona_base[L])}")
        print(f"  OCT cos(OCT-self, persona-base genF [mean/15])   : {m(oct_cos[L])}")
    print("\nVERDICT @L56:  EM cos = {:.3f}  vs  OCT cos(mean) = {:.3f}".format(
        em_cos[56], float(np.mean(oct_cos[56]))))
    print("Lower cosine = MORE rotation of the truth direction.")
    return {"em_cos_paired": em_cos,
            "em_cos_vs_persona_base": {L: list(map(float, v)) for L, v in em_vs_persona_base.items()},
            "oct_cos_vs_persona_base": {L: list(map(float, v)) for L, v in oct_cos.items()}}


@app.local_entrypoint()
def main():
    import json
    res = run.remote()
    print("\nJSON_RESULT:", json.dumps(res))
