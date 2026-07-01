"""Compute Qwen3-8B truth-probe LODO (leave-one-dataset-out) mean-AUC per layer
from the saved Geometry-of-Truth activations on the dpo-checkpoints volume.

Mirrors the Llama recipe: StandardScaler + LR(C=0.01), 4 GoT datasets
(cities, sp_en_trans, larger_than, general_facts), train on 3 -> AUC on held-out,
average over the 4 folds, at every saved layer.

  uv run modal run scripts/compute_qwen_lodo.py
"""
import os
import modal

VOL = "dpo-checkpoints"
ROOT = "probe-data/training_acts"          # Qwen3-8B GoT acts (37 layers, 400/ds)
DATASETS = ["cities", "sp_en_trans", "larger_than", "general_facts"]

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "numpy", "scikit-learn")
app = modal.App("qwen-lodo", image=image)
volume = modal.Volume.from_name(VOL)


@app.function(volumes={"/vol": volume}, timeout=60 * 20)
def compute():
    import os, json, glob
    import torch, numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    base = f"/vol/{ROOT}"
    # labels per dataset from metadata.json (same order as layer_*.pt rows)
    labels = {}
    for ds in DATASETS:
        meta = json.load(open(f"{base}/{ds}/metadata.json"))
        labels[ds] = np.array([int(m["label"]) for m in meta])
    # discover layers from cities
    layer_ids = sorted(int(os.path.basename(p).split("_")[1].split(".")[0])
                       for p in glob.glob(f"{base}/cities/layer_*.pt"))
    out = {"model": "Qwen3-8B", "datasets": DATASETS, "all_layers": {}}
    for L in layer_ids:
        X = {ds: torch.load(f"{base}/{ds}/layer_{L}.pt",
                            map_location="cpu", weights_only=False).float().numpy()
             for ds in DATASETS}
        fold_aucs = {}
        for held in DATASETS:
            tr = [d for d in DATASETS if d != held]
            Xtr = np.concatenate([X[d] for d in tr])
            ytr = np.concatenate([labels[d] for d in tr])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xtr), ytr)
            p = clf.predict_proba(sc.transform(X[held]))[:, 1]
            fold_aucs[held] = float(roc_auc_score(labels[held], p))
        mean_auc = float(np.mean(list(fold_aucs.values())))
        out["all_layers"][str(L)] = {"fold_aucs": fold_aucs, "mean_auc": mean_auc}
        print(f"L{L:2d} LODO mean_auc={mean_auc:.4f}")
    best = max(out["all_layers"], key=lambda k: out["all_layers"][k]["mean_auc"])
    out["best_layer"] = int(best)
    out["best_mean_auc"] = out["all_layers"][best]["mean_auc"]
    print(f"[best] L{best} = {out['best_mean_auc']:.4f}")
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    res = compute.remote()
    out = Path(os.environ.get("FIGURES_DIR", str(Path.home() / "Documents/obsidian_vault/claude_notes/truth_probe_paper/figures"))) / "qwen_lodo.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
