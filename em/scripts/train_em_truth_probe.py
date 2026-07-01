"""Train an EM-native truth probe and compare its direction to the
aligned-trained probe.

Both probes: same Marks statements, same layer 32, same logistic-regression
hyperparameters -- the ONLY difference is which model produced the activations.
So the cosine between their weight vectors isolates the model effect.

  high cosine  -> EM only shifted the probe-score offset; the truth direction
                  is stable, and recentring the transferred probe is enough.
  low  cosine  -> EM rotated its truth direction; an EM-native probe is doing
                  real work and the transferred probe was measuring the wrong
                  axis.

Inputs:
  probes/acts_em_truth_marks.pt  (modal volume get activations_probe/em_truth_marks.pt)
  probes/truth_probe.pkl         (the aligned-trained probe)
  probes/acts_truth.pt           (aligned activations; optional, enables per-layer)
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
PROBES = ROOT / "probes"
EM_ACTS = PROBES / "acts_em_truth_marks.pt"
ALIGNED_ACTS = PROBES / "acts_truth.pt"
ALIGNED_PROBE = PROBES / "truth_probe.pkl"

C_REG = 0.1


def fit(X, y):
    clf = LogisticRegression(C=C_REG, penalty="l2", solver="lbfgs",
                             max_iter=2000, class_weight=None)
    clf.fit(X, y)
    return clf


def lodo_auc(acts, labels, sources, layer_idx):
    X = acts[:, layer_idx, :].numpy().astype(np.float32)
    aucs = []
    for held in sorted(set(sources.tolist())):
        tr, te = sources != held, sources == held
        if tr.sum() == 0 or te.sum() == 0:
            continue
        clf = fit(X[tr], labels[tr])
        try:
            aucs.append(roc_auc_score(labels[te], clf.decision_function(X[te])))
        except ValueError:
            pass
    return float(np.nanmean(aucs))


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    em = torch.load(EM_ACTS, map_location="cpu", weights_only=False)
    layers = em["layers"]
    em_acts = em["activations"]            # (N, n_layers, hidden)
    labels = em["labels"].numpy()
    sources = np.array([m["source"] for m in em["meta"]])
    print(f"EM activations: N={len(labels)} layers={layers} "
          f"(true={int(labels.sum())}, false={int((1 - labels).sum())})")

    with open(ALIGNED_PROBE, "rb") as f:
        aligned = pickle.load(f)
    aligned_layer = int(aligned["meta"]["layer"])
    aligned_coef = aligned["classifier"].coef_[0]
    L = layers.index(aligned_layer)
    print(f"Aligned probe: layer {aligned_layer}")

    # EM-native probe at the aligned probe's layer.
    em_X = em_acts[:, L, :].numpy().astype(np.float32)
    em_clf = fit(em_X, labels)
    em_lodo = lodo_auc(em_acts, labels, sources, L)
    print(f"\nEM-native truth probe @ layer {aligned_layer}:  LODO AUC = {em_lodo:.4f}")

    cos = cosine(em_clf.coef_[0], aligned_coef)
    print(f"\n{'='*58}")
    print(f"  cosine(EM-trained dir, aligned-trained dir) = {cos:+.4f}")
    print(f"{'='*58}")

    # Per-layer view if the aligned activations are available locally.
    if ALIGNED_ACTS.exists():
        al = torch.load(ALIGNED_ACTS, map_location="cpu", weights_only=False)
        assert al["layers"] == layers, "layer sets differ"
        al_acts, al_labels = al["activations"], al["labels"].numpy()
        al_sources = np.array([m["source"] for m in al["meta"]])
        print(f"\n{'layer':>6} {'EM LODO':>9} {'aligned LODO':>13} {'cosine(dirs)':>13}")
        for i, layer in enumerate(layers):
            em_c = fit(em_acts[:, i, :].numpy().astype(np.float32), labels)
            al_c = fit(al_acts[:, i, :].numpy().astype(np.float32), al_labels)
            print(f"{layer:>6} {lodo_auc(em_acts, labels, sources, i):>9.4f} "
                  f"{lodo_auc(al_acts, al_labels, al_sources, i):>13.4f} "
                  f"{cosine(em_c.coef_[0], al_c.coef_[0]):>13.4f}")
    else:
        print(f"\n(aligned activations {ALIGNED_ACTS.name} not present -- "
              f"layer-32 cosine only; skipping per-layer table)")

    # Save the EM-native probe for the fuller re-analysis.
    meta = {"kind": "truth", "layer": aligned_layer, "C": C_REG,
            "n_samples": int(len(labels)), "lodo_auc": em_lodo,
            "model": "em_rank1_full_train",
            "cosine_to_aligned": cos}
    with open(PROBES / "truth_probe_em.pkl", "wb") as f:
        pickle.dump({"classifier": em_clf, "meta": meta}, f)
    with open(PROBES / "truth_probe_em.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved EM-native probe -> {PROBES / 'truth_probe_em.pkl'}")


if __name__ == "__main__":
    main()
