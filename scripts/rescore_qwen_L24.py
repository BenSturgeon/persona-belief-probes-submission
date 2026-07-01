"""Rescore the Qwen3-8B persona ICL ladder at layer 24 (CPU-only).

Trains the neutral truth probe at L24 from the Geometry-of-Truth acts (same recipe
as the deployed probe: StandardScaler + LR C=0.01 on the 4 GoT datasets pooled),
then scores every persona's k0 (baseline) and matched_wolf (= ICL k=32) ladder
activations at L24. Emits per-persona per-category mean probe scores + the
protection gap (EB delta - EF delta), matching the L20 figures' inputs.

  uv run modal run scripts/rescore_qwen_L24.py
"""
import os
import modal

VOL = "dpo-checkpoints"
GOT = "probe-data/training_acts"            # Qwen GoT acts (37 layers, 400/ds)
LADDER = "probe-data/qwen_control_ladder_acts"
DATASETS = ["cities", "sp_en_trans", "larger_than", "general_facts"]
LAYER = 24
HIST = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing", "p21_generic_athenian_chronicler",
    "p22_generic_abbasid_philosopher", "p23_generic_renaissance_advisor",
    "p24_victorian_spiritualist_medium", "p25_generic_radio_engineer",
]

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "numpy", "scikit-learn")
app = modal.App("qwen-rescore-L24", image=image)
volume = modal.Volume.from_name(VOL)


@app.function(volumes={"/vol": volume}, timeout=60 * 20)
def rescore():
    import os, json, glob
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # 1. Train neutral probe at L24 from pooled GoT acts.
    base = f"/vol/{GOT}"
    Xs, ys = [], []
    for ds in DATASETS:
        meta = json.load(open(f"{base}/{ds}/metadata.json"))
        y = np.array([int(m["label"]) for m in meta])
        X = torch.load(f"{base}/{ds}/layer_{LAYER}.pt", map_location="cpu",
                       weights_only=False).float().numpy()
        Xs.append(X); ys.append(y)
    Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(scaler.transform(Xtr), ytr)
    print(f"[probe] L{LAYER} trained on {len(ytr)} GoT stmts, train acc "
          f"{clf.score(scaler.transform(Xtr), ytr):.3f}")

    def score(acts):  # decision_function = the probe score (matches +2/-7 scale)
        return scaler.transform(acts) @ clf.coef_.squeeze() + float(clf.intercept_[0])

    # 2. Score each persona's k0 + matched_wolf (=k32) ladder acts at L24.
    out = {}
    for pid in HIST:
        pdir = f"/vol/{LADDER}/{pid}"
        meta = json.load(open(f"{pdir}/metadata.json"))
        li = list(meta["save_layers"]).index(LAYER)
        cats = [s["category"] for s in meta["statements"]]
        rec = {}
        for cond, fname in [("k0", "k0.npz"), ("k32", "matched_wolf.npz")]:
            npz = np.load(f"{pdir}/{fname}")
            acts = npz["acts"][:, li, :]          # (N, d)
            sc = score(acts)
            cm = {}
            for c in set(cats):
                idx = [i for i, cc in enumerate(cats) if cc == c]
                cm[c] = float(np.mean(sc[idx]))
            rec[cond] = cm
        out[pid] = rec
        eb_d = rec["k32"]["era_believed"] - rec["k0"]["era_believed"]
        ef_d = rec["k32"]["era_false"] - rec["k0"]["era_false"]
        rec["protection_gap"] = eb_d - ef_d
        print(f"[{pid}] gap {eb_d - ef_d:+.2f}")
    gaps = [out[p]["protection_gap"] for p in HIST]
    out["_summary"] = {"mean_gap": float(sum(gaps) / len(gaps)),
                       "n_positive": int(sum(g > 0 for g in gaps)), "n": len(gaps)}
    print(f"[L24] mean gap {out['_summary']['mean_gap']:+.3f}, "
          f"{out['_summary']['n_positive']}/{len(gaps)} positive")
    return out


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    res = rescore.remote()
    out = Path(os.environ.get("FIGURES_DIR", str(Path.home() / "Documents/obsidian_vault/claude_notes/truth_probe_paper/figures"))) / "qwen_L24_scores.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    print(res["_summary"])
