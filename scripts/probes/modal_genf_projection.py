#!/usr/bin/env python3
"""Items 1+3 (deterministic, on-cluster): genF era-topic projection + cosines.
Runs the score_oct_projection.py logic on the genF_marks_disbel_{model}_v3 dirs
(gen_prompt=False), for BOTH contrasts:
  era-believed:    gap = Δ_EB - Δ_EF, topic axis = era_believed vs era_false (_eb.pt)
  era-disbelieved: gap = Δ_ET - Δ_ED, topic axis = era_true vs era_disbelieved (_disbel.pt)
Self-convention: native truth probe refit on (residualized) marks each side; gap d = z_oct - z_base.
Reports per persona/layer gap_full, gap_resid, frac_retained, and cosine(topic-axis, truth-dir).
Saves oct-darwin:/probe/genf_projection/{model}_{contrast}.json and prints summary at primary + L56.
  modal run --detach modal_genf_projection.py --model llama
  modal run --detach modal_genf_projection.py --model qwen
"""
import modal, os, json

app = modal.App("genf-projection")
OCT = modal.Volume.from_name("oct-darwin")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "torch", "scikit-learn")

DIRS = {"llama": "genF_marks_disbel_llama_v3", "qwen": "genF_marks_disbel_qwen_v3"}


@app.function(image=image, cpu=8.0, memory=65536, timeout=7200, volumes={"/oct": OCT})
def project(model: str):
    import numpy as np, torch, gc
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    root = f"/oct/probe/{DIRS[model]}"
    personas = sorted(d for d in os.listdir(root)
                      if os.path.isdir(f"{root}/{d}") and
                      all(os.path.exists(f"{root}/{d}/{o}_{k}.pt")
                          for o in ("base", "oct") for k in ("marks", "eb", "disbel")))
    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    layers = load(f"{root}/{personas[0]}/base_marks.pt")["layers"]
    primary = 30 if model == "llama" else 24
    print(f"[{model}] personas={len(personas)} layers={layers} primary=HF{primary}", flush=True)

    def axis(A, cats, pos, neg):
        m = (cats == pos) | (cats == neg)
        X = A[m]; y = (cats[m] == pos).astype(int)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), y)
        w = clf.coef_[0] / sc.scale_
        return w / np.linalg.norm(w)

    def resid(A, u):
        return A - np.outer(A @ u, u)

    def truth_probe(Xm, ym):
        sc = StandardScaler().fit(Xm)
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xm), ym)
        d = clf.decision_function(sc.transform(Xm))
        fm = float(d[ym == 0].mean()); tm = float(d[ym == 1].mean()); span = (tm - fm) or 1.0
        w = clf.coef_[0] / sc.scale_; w = w / np.linalg.norm(w)
        return (lambda A: clf.decision_function(sc.transform(A)),
                lambda v: (v - fm) / span, w)

    def gap_dz(beA, beC, bmX, bmY, oeA, oeC, omX, omY, pos_cat, neg_cat):
        s_b, z_b, w_b = truth_probe(bmX, bmY)
        s_o, z_o, w_o = truth_probe(omX, omY)
        def pos(A, C, s, z):
            r = s(A)
            return (float(z(r[C == pos_cat]).mean()), float(z(r[C == neg_cat]).mean()))
        eb_b, ef_b = pos(beA, beC, s_b, z_b)
        eb_o, ef_o = pos(oeA, oeC, s_o, z_o)
        return (eb_o - eb_b) - (ef_o - ef_b), w_o

    CONTRASTS = {
        "era_believed":    ("eb",     "era_believed", "era_false"),
        "era_disbelieved": ("disbel", "era_true",     "era_disbelieved"),
    }
    out = {"model": model, "layers_hf": layers, "primary_hf": primary, "contrasts": {}}
    for cname, (suffix, pos_cat, neg_cat) in CONTRASTS.items():
        res = {}
        for Li, hf in enumerate(layers):
            for p in personas:
                bm = load(f"{root}/{p}/base_marks.pt"); om = load(f"{root}/{p}/oct_marks.pt")
                be = load(f"{root}/{p}/base_{suffix}.pt"); oe = load(f"{root}/{p}/oct_{suffix}.pt")
                bmX = bm["activations"][:, Li, :].float().numpy(); bmY = bm["labels"].numpy().astype(int)
                omX = om["activations"][:, Li, :].float().numpy(); omY = om["labels"].numpy().astype(int)
                beA = be["activations"][:, Li, :].float().numpy(); beC = np.array([m["category"] for m in be["meta"]])
                oeA = oe["activations"][:, Li, :].float().numpy(); oeC = np.array([m["category"] for m in oe["meta"]])
                g_full, w_truth = gap_dz(beA, beC, bmX, bmY, oeA, oeC, omX, omY, pos_cat, neg_cat)
                u = axis(oeA, oeC, pos_cat, neg_cat)
                g_res, _ = gap_dz(resid(beA, u), beC, resid(bmX, u), bmY,
                                  resid(oeA, u), oeC, resid(omX, u), omY, pos_cat, neg_cat)
                cos = float(abs(np.dot(u, w_truth)))
                res.setdefault(p, {})[str(hf)] = {
                    "gap_full": g_full, "gap_resid": g_res,
                    "frac_retained": (g_res / g_full) if abs(g_full) > 1e-9 else None,
                    "cos_axis_truth": cos}
                del bm, om, be, oe, bmX, omX, beA, oeA; gc.collect()
            print(f"[{model}/{cname}] HF{hf} done", flush=True)
        out["contrasts"][cname] = res

    os.makedirs("/oct/probe/genf_projection", exist_ok=True)
    json.dump(out, open(f"/oct/probe/genf_projection/{model}.json", "w"), indent=1)
    OCT.commit()
    # summary
    for cname in CONTRASTS:
        P = out["contrasts"][cname]
        for hf in (str(primary), "56"):
            if hf not in next(iter(P.values())): continue
            gf = np.array([P[p][hf]["gap_full"] for p in personas])
            gr = np.array([P[p][hf]["gap_resid"] for p in personas])
            cos = np.mean([P[p][hf]["cos_axis_truth"] for p in personas])
            ret = (gr.mean() / gf.mean() * 100) if abs(gf.mean()) > 1e-9 else float("nan")
            npos = int((gr > 0).sum())
            print(f"[{model}/{cname}] HF{hf}: full {gf.mean():+.4f} -> resid {gr.mean():+.4f} "
                  f"(retained {ret:.0f}%, resid_pos {npos}/{len(personas)}, cos {cos:.3f})", flush=True)
    return out


@app.local_entrypoint()
def main(model: str = "llama"):
    print("OK", project.remote(model))
