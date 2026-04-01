#!/usr/bin/env python3
"""Score pre-extracted SFT activations at L20 with LR probe, per-persona.

Key: for each SFT persona model, only score THAT persona's statements
(not all 14400). This gives per-persona EB/EF means comparable to ICL scores.

The activations were extracted by modal_extract_all_layer_acts.py at:
    /checkpoints/probe-data/eval_acts_all_layers/{model_key}/layer_{N}.pt
    /checkpoints/probe-data/eval_acts_all_layers/{model_key}/metadata.json

Usage:
    modal run scripts/probes/modal_score_sft_per_persona.py
"""
import modal
import json
import os

app = modal.App("score-sft-per-persona")
volume = modal.Volume.from_name("dpo-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.1.0", "numpy", "scikit-learn")
)

HISTORICAL = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]

ALL_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p11_hal_9000", "p12_glados", "p13_marvin", "p14_c3po",
    "p15_rick_deckard", "p16_the_doctor", "p17_gandalf", "p18_saruman",
    "p19_sherlock_holmes", "p20_tom_ripley",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
    "p26_demis_hassabis", "p27_tim_berners_lee", "p28_greta_thunberg",
    "p29_simon_leviev", "p30_elizabeth_holmes",
]

LAYER = 20


@app.function(timeout=3600, image=image, volumes={"/checkpoints": volume}, cpu=4)
def score_all():
    """Score all SFT persona models at L20, filtering to each persona's own statements."""
    import torch
    import numpy as np
    from collections import defaultdict

    # --- Load LR probe at L20 ---
    lr_path = f"/checkpoints/probe-data/probe_results/lr_probes/lr_layer_{LAYER}.json"
    if os.path.exists(lr_path):
        with open(lr_path) as f:
            lr_data = json.load(f)
        coef = np.array(lr_data["coef"])
        intercept = float(lr_data["intercept"])
        scaler_mean = np.array(lr_data.get("scaler_mean")) if "scaler_mean" in lr_data else None
        scaler_scale = np.array(lr_data.get("scaler_scale")) if "scaler_scale" in lr_data else None
        print(f"Loaded pre-trained LR probe at L{LAYER}")
    else:
        print(f"No pre-trained probe at {lr_path}, training from scratch...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        datasets_list = ["cities", "sp_en_trans", "larger_than", "general_facts"]
        all_X, all_y = [], []
        for ds in datasets_list:
            meta_path = f"/checkpoints/probe-data/training_acts/{ds}/metadata.json"
            act_path = f"/checkpoints/probe-data/training_acts/{ds}/layer_{LAYER}.pt"
            if os.path.exists(meta_path) and os.path.exists(act_path):
                with open(meta_path) as f:
                    meta_list = json.load(f)
                acts = torch.load(act_path, map_location="cpu", weights_only=True).numpy()
                labels = np.array([m["label"] for m in meta_list])
                all_X.append(acts)
                all_y.append(labels)
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        lr = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
        lr.fit(X_s, y)
        coef = lr.coef_.squeeze()
        intercept = float(lr.intercept_)
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
        print(f"Trained fresh LR probe at L{LAYER}")

    # Score neutral baseline first (for delta computation)
    # Then score each persona model, filtering to THAT persona's statements + controls
    
    all_results = []

    for model_key in ["neutral"] + ALL_PERSONAS:
        acts_dir = f"/checkpoints/probe-data/eval_acts_all_layers/{model_key}"
        act_path = f"{acts_dir}/layer_{LAYER}.pt"
        meta_path = f"{acts_dir}/metadata.json"

        if not os.path.exists(act_path) or not os.path.exists(meta_path):
            print(f"  SKIP {model_key}: no activations at L{LAYER}")
            continue

        # Load all activations + metadata
        acts = torch.load(act_path, map_location="cpu", weights_only=True).float().numpy()
        with open(meta_path) as f:
            metadata = json.load(f)

        assert len(acts) == len(metadata), f"{model_key}: acts={len(acts)} != meta={len(metadata)}"

        # Scale + score ALL statements
        if scaler_mean is not None and scaler_scale is not None:
            acts_s = (acts - scaler_mean) / scaler_scale
        else:
            acts_s = acts
        all_scores = (acts_s @ coef.reshape(-1, 1) + intercept).squeeze()

        # For persona models: filter to only THIS persona's era_believed/era_false/era_true
        # + control statements. For neutral: use all.
        if model_key == "neutral":
            # Neutral: keep all (we'll use per-persona subsets when computing deltas)
            indices = list(range(len(metadata)))
        else:
            # Filter: keep statements where persona_id matches OR category is control_*
            indices = [
                i for i, m in enumerate(metadata)
                if m.get("persona_id") == model_key or m["category"].startswith("control_")
            ]

        # Compute category means from filtered set
        cat_scores = defaultdict(list)
        for i in indices:
            cat_scores[metadata[i]["category"]].append(float(all_scores[i]))

        category_means = {}
        for cat in sorted(cat_scores):
            vals = cat_scores[cat]
            category_means[cat] = {
                str(LAYER): {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "n": len(vals),
                }
            }

        result = {
            "persona_id": model_key,
            "model_key": model_key,
            "n_statements_scored": len(indices),
            "n_statements_total": len(metadata),
            "category_means": category_means,
        }
        all_results.append(result)

        eb = category_means.get("era_believed", {}).get(str(LAYER), {})
        ef = category_means.get("era_false", {}).get(str(LAYER), {})
        print(f"  {model_key:40s} EB={eb.get('mean','—')} (n={eb.get('n','—')})  EF={ef.get('mean','—')} (n={ef.get('n','—')})")

    # Save combined
    out_path = f"/checkpoints/probe-data/sft_per_persona_L{LAYER}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    volume.commit()
    print(f"\nSaved {len(all_results)} model results to {out_path}")

    # Print delta summary (each persona model vs neutral, using that persona's statements)
    neutral_data = next((r for r in all_results if r["persona_id"] == "neutral"), None)
    if neutral_data:
        # For neutral, we need per-persona subsets too
        # Reload neutral metadata to get per-persona filtering
        n_meta_path = f"/checkpoints/probe-data/eval_acts_all_layers/neutral/metadata.json"
        n_act_path = f"/checkpoints/probe-data/eval_acts_all_layers/neutral/layer_{LAYER}.pt"
        n_acts = torch.load(n_act_path, map_location="cpu", weights_only=True).float().numpy()
        with open(n_meta_path) as f:
            n_metadata = json.load(f)
        if scaler_mean is not None:
            n_acts_s = (n_acts - scaler_mean) / scaler_scale
        else:
            n_acts_s = n_acts
        n_scores = (n_acts_s @ coef.reshape(-1, 1) + intercept).squeeze()

        print(f"\n{'='*70}")
        print(f"Per-persona deltas (SFT model vs neutral, same persona statements)")
        print(f"{'='*70}")

        for persona in ALL_PERSONAS:
            # Get neutral scores for this persona's statements
            n_cat = defaultdict(list)
            for i, m in enumerate(n_metadata):
                if m.get("persona_id") == persona or m["category"].startswith("control_"):
                    n_cat[m["category"]].append(float(n_scores[i]))

            # Get SFT scores
            sft_r = next((r for r in all_results if r["persona_id"] == persona), None)
            if not sft_r:
                continue

            n_eb = np.mean(n_cat.get("era_believed", [0]))
            n_ef = np.mean(n_cat.get("era_false", [0]))
            s_eb = sft_r["category_means"].get("era_believed", {}).get(str(LAYER), {}).get("mean", 0)
            s_ef = sft_r["category_means"].get("era_false", {}).get(str(LAYER), {}).get("mean", 0)

            d_eb = s_eb - n_eb
            d_ef = s_ef - n_ef
            diff = d_eb - d_ef
            tag = "HIST" if persona in HISTORICAL else "    "
            print(f"  {tag} {persona:40s} ΔEB={d_eb:+.4f}  ΔEF={d_ef:+.4f}  EB-EF diff={diff:+.4f}")

    return all_results


@app.local_entrypoint()
def main():
    results = score_all.remote()

    # Save locally too
    local_path = "data/qwen3_8b/sft_per_persona_L20.json"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved locally to {local_path}")
