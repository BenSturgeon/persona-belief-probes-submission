#!/usr/bin/env python3
"""Dual-convention era-belief gap scorer for the OCT organisms (Qwen or Llama).

Reports, per persona and per layer, BOTH conventions (per the Qwen frozen-probe
cross-check, oct-multimodel.md):

(a) SELF: per-organism Marks probe (StandardScaler + LR C=0.01) with that
    organism's own marks-mean affine calibration. The published convention.
(b) FROZEN: one Marks probe trained on the POOLED base-model Marks activations
    (all personas' base extractions concatenated) with a single base marks-mean
    calibration, applied unchanged to every organism's era acts.
(c) cos(base, self): cosine between the frozen pooled-base probe direction and
    each organism's self-probe direction, in RAW activation space
    (w_raw = coef / scaler.scale_), per layer.
(d) frozen-probe held-out Marks AUC inside each organism (probe collapse check).

d_EB and d_EF are reported separately under both rulers.

Usage:
  python score_oct_era_gap_dual.py --dir ./oct_era_mm_llama/era_mm_llama \
      --out ./oct_era_gap_llama_dual.json
"""
import os
import json
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_probe(X, y):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(Xs, y)
    d = clf.decision_function(Xs)
    fm = float(d[y == 0].mean()); tm = float(d[y == 1].mean())
    span = tm - fm
    score = lambda A: clf.decision_function(sc.transform(A))
    z = lambda v: (v - fm) / span
    w_raw = (clf.coef_[0] / sc.scale_)
    cvp = cross_val_predict(LogisticRegression(C=0.01, max_iter=2000), Xs, y, cv=5,
                            method="decision_function")
    auc = float(roc_auc_score(y, cvp))
    return {"score": score, "z": z, "w_raw": w_raw, "cv_auc": auc, "span": span}


def era_pos(era_pt, probe, L_idx):
    acts = era_pt["activations"][:, L_idx, :].float().numpy()
    cats = np.array([m["category"] for m in era_pt["meta"]])
    raw = probe["score"](acts)
    return (float(probe["z"](raw[cats == "era_believed"]).mean()),
            float(probe["z"](raw[cats == "era_false"]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    personas = sorted(d for d in os.listdir(args.dir)
                      if os.path.isdir(os.path.join(args.dir, d)))
    data = {}
    for p in personas:
        pd_ = {}
        ok = True
        for o in ("base", "oct"):
            for k in ("marks", "era"):
                f = os.path.join(args.dir, p, f"{o}_{k}.pt")
                if not os.path.exists(f):
                    ok = False
                    break
                pd_[f"{o}_{k}"] = load(f)
        if ok:
            data[p] = pd_
        else:
            print(f"{p}: incomplete, skip")
    assert data, "no complete personas"

    any_p = next(iter(data.values()))
    layers = any_p["base_marks"]["layers"]
    primary = any_p["base_marks"].get("primary_hf", layers[len(layers) // 2])
    print(f"personas={len(data)} layers={layers} primary=HF{primary}")

    results = {"layers_hf": layers, "primary_hf": primary, "personas": {}}

    for L_idx, hf in enumerate(layers):
        # FROZEN probe: pooled base marks acts across personas
        Xb = np.concatenate([data[p]["base_marks"]["activations"][:, L_idx, :].float().numpy()
                             for p in data])
        yb = np.concatenate([data[p]["base_marks"]["labels"].numpy().astype(int) for p in data])
        frozen = fit_probe(Xb, yb)

        for p in data:
            rec = results["personas"].setdefault(p, {})[str(hf)] = {}
            base_marks, oct_marks = data[p]["base_marks"], data[p]["oct_marks"]
            base_era, oct_era = data[p]["base_era"], data[p]["oct_era"]

            # (a) SELF convention
            yo = oct_marks["labels"].numpy().astype(int)
            self_base = fit_probe(base_marks["activations"][:, L_idx, :].float().numpy(),
                                  base_marks["labels"].numpy().astype(int))
            self_oct = fit_probe(oct_marks["activations"][:, L_idx, :].float().numpy(), yo)
            eb_b, ef_b = era_pos(base_era, self_base, L_idx)
            eb_o, ef_o = era_pos(oct_era, self_oct, L_idx)
            rec["self"] = {
                "abs_eb_base": eb_b, "abs_ef_base": ef_b,
                "abs_eb_oct": eb_o, "abs_ef_oct": ef_o,
                "d_eb": eb_o - eb_b, "d_ef": ef_o - ef_b,
                "gap": (eb_o - eb_b) - (ef_o - ef_b),
                "marks_cv_auc_base": self_base["cv_auc"],
                "marks_cv_auc_oct": self_oct["cv_auc"],
            }

            # (b) FROZEN convention
            f_eb_b, f_ef_b = era_pos(base_era, frozen, L_idx)
            f_eb_o, f_ef_o = era_pos(oct_era, frozen, L_idx)
            # (d) frozen probe held-out AUC inside the organism
            oct_X = oct_marks["activations"][:, L_idx, :].float().numpy()
            frozen_auc_in_oct = float(roc_auc_score(yo, frozen["score"](oct_X)))
            rec["frozen"] = {
                "abs_eb_base": f_eb_b, "abs_ef_base": f_ef_b,
                "abs_eb_oct": f_eb_o, "abs_ef_oct": f_ef_o,
                "d_eb": f_eb_o - f_eb_b, "d_ef": f_ef_o - f_ef_b,
                "gap": (f_eb_o - f_eb_b) - (f_ef_o - f_ef_b),
                "frozen_marks_auc_in_oct": frozen_auc_in_oct,
            }

            # (c) probe-direction cosine, raw space
            wf, wo = frozen["w_raw"], self_oct["w_raw"]
            rec["cos_base_self"] = float(np.dot(wf, wo) /
                                         (np.linalg.norm(wf) * np.linalg.norm(wo)))

    # summary at primary
    hf = str(primary)
    print(f"\n{'persona':36s} gap_self gap_frzn d_EB_f d_EF_f  cos  frzAUC")
    gs, gf, ebf, eff, cs, fa = [], [], [], [], [], []
    for p in sorted(results["personas"]):
        r = results["personas"][p][hf]
        s, f, c = r["self"], r["frozen"], r["cos_base_self"]
        gs.append(s["gap"]); gf.append(f["gap"]); ebf.append(f["d_eb"]); eff.append(f["d_ef"])
        cs.append(c); fa.append(f["frozen_marks_auc_in_oct"])
        print(f"{p:36s} {s['gap']:+.3f}  {f['gap']:+.3f}  {f['d_eb']:+.3f} {f['d_ef']:+.3f} "
              f"{c:.3f} {f['frozen_marks_auc_in_oct']:.3f}")
    print(f"\nMEAN @HF{hf}: self {np.mean(gs):+.4f}+-{np.std(gs):.4f} | "
          f"frozen {np.mean(gf):+.4f}+-{np.std(gf):.4f} (n_pos {sum(1 for x in gf if x>0)}/{len(gf)}) | "
          f"frozen d_EB {np.mean(ebf):+.4f} d_EF {np.mean(eff):+.4f} | "
          f"cos {np.mean(cs):.3f} | frozen AUC in oct {np.mean(fa):.3f}")

    json.dump(results, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
