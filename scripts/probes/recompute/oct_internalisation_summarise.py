"""Summarise oct_internalisation_raw.json -> oct_internalisation_summary.json (mean, sd(ddof=1), 1.96*sd/sqrt(n), n_pos, t, p, d=mean/sd)."""
import json, os
import numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
raw = json.load(open(f"{H}/oct_internalisation_raw.json"))


def st(v):
    v = np.array(v, float); n = len(v); sd = v.std(ddof=1)
    t, p = stats.ttest_1samp(v, 0.0)
    return {"mean": round(v.mean(), 4), "sd": round(sd, 4), "ci95_half": round(1.96 * sd / np.sqrt(n), 4),
            "n_pos": int((v > 0).sum()), "n": n, "t": round(float(t), 3), "p": float(p), "d": round(v.mean() / sd, 3),
            "wilcoxon_p": float(stats.wilcoxon(v).pvalue)}


S = {"inputs_sha256": {}}
for model, M in raw.items():
    for p, R in M["personas"].items():
        S["inputs_sha256"].update(R["sha256"])
    S["inputs_sha256"].update(M["pooled_sha256"])
    S["inputs_sha256"]["oct_internalisation_raw.json"] = None
    P = list(M["personas"].values())
    S[model] = {"layers_field": M["pooled_layers_field"], "files_info_example": P[0]["files_info"]}
    for L in P[0]["L"]:
        out = {}
        for rk in P[0]["L"][L]["rulers"]:
            for q in ("lift_EB", "protection", "demotion"):
                out[f"{rk}.{q}"] = st([R["L"][L]["rulers"][rk][q] for R in P])
            for c in ("era_believed", "era_false", "era_true", "era_disbelieved"):
                out[f"{rk}.delta_{c}"] = round(float(np.mean([R["L"][L]["rulers"][rk]["delta"][c] for R in P])), 4)
        for ck in P[0]["L"][L]["cos"]:
            v = np.array([R["L"][L]["cos"][ck] for R in P]); out[f"cos.{ck}"] = {"mean": round(v.mean(), 3), "min": round(v.min(), 3), "max": round(v.max(), 3)}
        for cn in P[0]["L"][L]["projection"]:
            gf = np.array([R["L"][L]["projection"][cn]["gap_full"] for R in P])
            gr = np.array([R["L"][L]["projection"][cn]["gap_resid"] for R in P])
            d = {"gap_full_mean": round(gf.mean(), 4), "gap_resid_mean": round(gr.mean(), 4),
                 "retained_pct_of_means": round(100 * gr.mean() / gf.mean(), 1), "resid_n_pos": int((gr > 0).sum()),
                 "median_per_persona_retained_pct": round(float(np.median(100 * gr / gf)), 1)}
            for ck in ("cos_axis_octtruth", "cos_baseaxis_basetruth", "cos_octaxis_basetruth"):
                d[ck] = round(float(np.mean([R["L"][L]["projection"][cn][ck] for R in P])), 4)
            dim = P[0]["L"][L]["projection"][cn]["dim"]
            d["random_floor_E|cos|"] = round(float(np.sqrt(2 / (np.pi * dim))), 4)
            out[f"projection.{cn}"] = d
        S[model][f"L{L}"] = out
import hashlib
S["inputs_sha256"]["oct_internalisation_raw.json"] = hashlib.sha256(open(f"{H}/oct_internalisation_raw.json", "rb").read()).hexdigest()
json.dump(S, open(f"{H}/oct_internalisation_summary.json", "w"), indent=1)
for model in raw:
    for Lk, out in S[model].items():
        if not Lk.startswith("L"): continue
        print(f"===== {model} {Lk}")
        for k, v in out.items():
            if isinstance(v, dict) and "mean" in v and "n_pos" in v and not k.split(".")[1].startswith("delta"):
                print(f"  {k:32s} {v['mean']:+.4f} ci±{v['ci95_half']:.4f} {v['n_pos']}/{v['n']} d={v['d']:.2f} p={v['p']:.2g}")
            elif k.startswith("cos") or k.startswith("projection"):
                print(f"  {k:32s} {v}")
