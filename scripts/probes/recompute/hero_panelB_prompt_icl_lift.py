"""READ-ONLY provenance test for Figure 1 panel B's untraced SP (0.07) and ICL (-0.04) values.
Hypothesis: they are L30 frozen-probe lifts like the old Figure 2 SFT/OCT bars. lr_layer_30 applied to
lens_acts_fig3_v3/{k0,sysprompt,icl_k32}/era_L30.npy (lens 29 = HF 30), calibrated with the neutral raw-text
Marks anchors (mf=-7.889, mt=5.875, from deepcheck pp_recompute C_llama_hf_v3_L30); lift = z(cond EB) - z(k0 EB).
Also reports the same at L56 (anchors mf=-7.089, mt=5.801)."""
import modal, json
app = modal.App("gt-hero-old-panelB"); dpo = modal.Volume.from_name("dpo-checkpoints").read_only()
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
P = ["p01_thucydides","p02_herodotus","p03_ibn_al_haytham","p04_machiavelli","p05_richard_nixon","p06_darwin","p07_tesla","p08_lovelace","p09_curie","p10_turing","p21_generic_athenian_chronicler","p22_generic_abbasid_philosopher","p23_generic_renaissance_advisor","p24_victorian_spiritualist_medium","p25_generic_radio_engineer"]
ANCH = {30: (-7.889216186352709, 5.875416629312499), 56: (-7.088947458167317, 5.801188460353456)}
@app.function(image=image, volumes={"/d": dpo}, timeout=1800, memory=16384)
def run():
    import numpy as np
    v3 = json.load(open("/d/probe-data/eval_statements/all_statements_v3.json"))
    order = []
    for pid in P: order += [s for s in v3 if s.get("persona_id") == pid and s.get("category") in ("era_believed", "era_false", "era_true")]
    out = {}
    for L in (30, 56):
        pr = json.load(open(f"/d/probe-data/llama70b_lr_probes/lr_layer_{L}.json"))
        w = np.asarray(pr["coef"], float).ravel(); b0 = float(np.ravel(pr["intercept"])[0]); mu = np.asarray(pr["scaler_mean"]); sd = np.asarray(pr["scaler_scale"])
        mf, mt = ANCH[L]; z = lambda A: ((((A - mu) / sd) @ w + b0) - mf) / (mt - mf)
        Z = {c: z(np.load(f"/d/probe-data/lens_acts_fig3_v3/{c}/era_L{L}.npy").astype(np.float64)) for c in ("k0", "sysprompt", "icl_k32")}
        for c in ("sysprompt", "icl_k32"):
            per = []
            for pid in P:
                idx = [i for i, o in enumerate(order) if o["persona_id"] == pid and o["category"] == "era_believed"]
                per.append(float(Z[c][idx].mean() - Z["k0"][idx].mean()))
            a = np.array(per); out[f"L{L}_{c}"] = dict(mean=float(a.mean()), n_pos=int((a > 0).sum()))
    return out
@app.local_entrypoint()
def main():
    r = run.remote(); json.dump(r, open("hero_panelB_prompt_icl_lift.json", "w"), indent=1)
    for k, v in r.items(): print(f"{k:16s} lift={v['mean']:+.3f} ({v['n_pos']}/15)")
