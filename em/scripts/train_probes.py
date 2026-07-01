"""Train truth and deception probes from saved activations.

Reads activation .pt files either from a local Modal volume mount or from
files downloaded via `modal volume get`. Trains L2-regularised logistic
regression on middle-layer activations, sweeps layers, picks best by
leave-one-dataset-out (truth) or 80/20 validation (deception), writes
probe weights + layer choice + metrics to probes/.

Usage:
  # 1. pull activations from the Modal volume:
  modal volume get em-probing probe_training/truth.pt ~/werk/em-probing/our/probes/acts_truth.pt
  modal volume get em-probing probe_training/deception.pt ~/werk/em-probing/our/probes/acts_deception.pt

  # 2. train:
  uv run python scripts/train_probes.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = ROOT / "probes"
PROBES_DIR.mkdir(parents=True, exist_ok=True)

TRUTH_ACTS = PROBES_DIR / "acts_truth.pt"
DECEPTION_ACTS = PROBES_DIR / "acts_deception.pt"

C_REG = 0.1  # matches Apollo


def _fit_probe(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        C=C_REG, penalty="l2", solver="lbfgs", max_iter=2000, class_weight=None
    )
    clf.fit(X, y)
    return clf


# ---- Truth probe: last-token activations, LODO layer selection ----


def train_truth_probe():
    print(f"[truth] loading {TRUTH_ACTS}")
    payload = torch.load(TRUTH_ACTS, map_location="cpu", weights_only=False)
    acts = payload["activations"]  # (N, n_layers, hidden)
    labels = payload["labels"].numpy()  # (N,)
    meta = payload["meta"]
    layers = payload["layers"]

    sources = np.array([m["source"] for m in meta])
    unique_sources = sorted(set(sources.tolist()))

    # Layer sweep: LODO AUC per layer
    per_layer_lodo_auc: dict[int, float] = {}
    per_layer_holdout_auc: dict[int, float] = {}

    for layer_idx, layer in enumerate(layers):
        X_all = acts[:, layer_idx, :].numpy().astype(np.float32)
        lodo_aucs = []
        for held_out in unique_sources:
            mask_train = sources != held_out
            mask_test = sources == held_out
            if mask_train.sum() == 0 or mask_test.sum() == 0:
                continue
            clf = _fit_probe(X_all[mask_train], labels[mask_train])
            scores = clf.decision_function(X_all[mask_test])
            try:
                auc = roc_auc_score(labels[mask_test], scores)
            except ValueError:
                auc = float("nan")
            lodo_aucs.append(auc)
        per_layer_lodo_auc[layer] = float(np.nanmean(lodo_aucs))

        # Also an 80/20 holdout fit over the full data for reference
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(labels))
        split = int(0.8 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]
        clf = _fit_probe(X_all[train_idx], labels[train_idx])
        scores = clf.decision_function(X_all[test_idx])
        per_layer_holdout_auc[layer] = float(roc_auc_score(labels[test_idx], scores))

        print(f"[truth] layer={layer} LODO_AUC={per_layer_lodo_auc[layer]:.4f} "
              f"holdout_AUC={per_layer_holdout_auc[layer]:.4f}")

    # Pick best layer by LODO AUC.
    best_layer = max(per_layer_lodo_auc, key=lambda layer: per_layer_lodo_auc[layer])
    best_layer_idx = layers.index(best_layer)
    print(f"[truth] best_layer={best_layer} LODO_AUC={per_layer_lodo_auc[best_layer]:.4f}")

    # Refit on ALL data at best layer for the final probe.
    X_best = acts[:, best_layer_idx, :].numpy().astype(np.float32)
    final_clf = _fit_probe(X_best, labels)

    out = {
        "kind": "truth",
        "layer": best_layer,
        "layer_idx_in_extraction": best_layer_idx,
        "per_layer_lodo_auc": per_layer_lodo_auc,
        "per_layer_holdout_auc": per_layer_holdout_auc,
        "C": C_REG,
        "n_samples": int(len(labels)),
    }
    with open(PROBES_DIR / "truth_probe.pkl", "wb") as f:
        pickle.dump({"classifier": final_clf, "meta": out}, f)
    with open(PROBES_DIR / "truth_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[truth] saved probe -> {PROBES_DIR / 'truth_probe.pkl'}")


# ---- Deception probe: mean-aggregated assistant tokens, 80/20 split ----


def train_deception_probe():
    print(f"[deception] loading {DECEPTION_ACTS}")
    payload = torch.load(DECEPTION_ACTS, map_location="cpu", weights_only=False)
    # activations is a list of (n_layers, n_assistant_tokens, hidden) tensors
    acts_list = payload["activations"]
    labels = payload["labels"].numpy()
    layers = payload["layers"]

    # Mean-aggregate assistant tokens per item per layer.
    # Result: (N, n_layers, hidden)
    mean_acts = torch.stack([a.mean(dim=1) for a in acts_list], dim=0).numpy().astype(np.float32)
    # Also max-aggregate for the alternative aggregation the identity post flagged
    max_acts = torch.stack([a.amax(dim=1) for a in acts_list], dim=0).numpy().astype(np.float32)

    per_layer_auc_mean: dict[int, float] = {}
    per_layer_auc_max: dict[int, float] = {}
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(labels))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    for layer_idx, layer in enumerate(layers):
        for label, bank, store in [
            ("mean", mean_acts, per_layer_auc_mean),
            ("max", max_acts, per_layer_auc_max),
        ]:
            X = bank[:, layer_idx, :]
            clf = _fit_probe(X[train_idx], labels[train_idx])
            auc = float(roc_auc_score(labels[test_idx], clf.decision_function(X[test_idx])))
            store[layer] = auc
        print(f"[deception] layer={layer} AUC mean={per_layer_auc_mean[layer]:.4f} "
              f"max={per_layer_auc_max[layer]:.4f}")

    # Pick best layer by MEAN-agg AUC (matches Apollo's default).
    best_layer = max(per_layer_auc_mean, key=lambda layer: per_layer_auc_mean[layer])
    best_layer_idx = layers.index(best_layer)
    print(f"[deception] best_layer={best_layer} mean_AUC={per_layer_auc_mean[best_layer]:.4f}")

    X_best = mean_acts[:, best_layer_idx, :]
    final_clf = _fit_probe(X_best, labels)

    out = {
        "kind": "deception",
        "layer": best_layer,
        "layer_idx_in_extraction": best_layer_idx,
        "aggregation": "mean",
        "per_layer_auc_mean": per_layer_auc_mean,
        "per_layer_auc_max": per_layer_auc_max,
        "C": C_REG,
        "n_samples": int(len(labels)),
    }
    with open(PROBES_DIR / "deception_probe.pkl", "wb") as f:
        pickle.dump({"classifier": final_clf, "meta": out}, f)
    with open(PROBES_DIR / "deception_probe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[deception] saved probe -> {PROBES_DIR / 'deception_probe.pkl'}")


if __name__ == "__main__":
    if TRUTH_ACTS.exists():
        train_truth_probe()
    else:
        print(f"[truth] SKIP: {TRUTH_ACTS} not found "
              f"(run `modal volume get em-probing probe_training/truth.pt {TRUTH_ACTS}`)")

    if DECEPTION_ACTS.exists():
        train_deception_probe()
    else:
        print(f"[deception] SKIP: {DECEPTION_ACTS} not found "
              f"(run `modal volume get em-probing probe_training/deception.pt {DECEPTION_ACTS}`)")
