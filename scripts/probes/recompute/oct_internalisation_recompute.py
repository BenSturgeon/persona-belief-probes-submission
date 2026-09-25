"""Deep-check recompute of the OCT representational numbers (ICLR 2027). READ-ONLY on oct-darwin / dpo-checkpoints.

Inputs (genF = add_generation_prompt=False, vllm-lens; activation axis = the file's own "layers" field, HF convention,
lens index = HF-1 internally):
  oct-darwin:probe/genF_marks_disbel_{llama,qwen}_v3/{persona}/{base,oct,sft}_{marks,eb,disbel}.pt
  dpo-checkpoints:probe-data/llama70b_lr_probes/lr_layer_{L}.json   (external raw-text Llama probe, one frozen variant)

Probe everywhere: StandardScaler + LogisticRegression(C=0.01, max_iter=2000); calibration z = (d - mean_false)/(mean_true - mean_false)
on the probe's own training Marks set (false=0, true=1).

Per persona, organism in {oct, sft}, vs base:
  native   : organism side scored by organism's own Marks probe, base side by base's own (per-persona) Marks probe
  frozen_pp: base per-persona Marks probe + base calibration applied to both sides
  frozen_pool: one probe on pooled base Marks (all personas) + pooled base calibration, both sides
  frozen_ext (Llama only): external lr_layer_L, calibrated on pooled base genF Marks, both sides
  hybrid   : base per-persona direction, anchors re-fit on each side's own Marks
Quantities: lift_EB = z_org(EB) - z_base(EB)  (rule 4, Figure 2);  protection = dEB - dEF;  demotion = dET - dED.
Cosines: raw-space unit directions (coef/scale).
Projection (app:oct_geometry Check 1/3): replicate modal_genf_projection.py (era-topic axis LR on organism era acts).
"""
import modal, json, os
app = modal.App("deepcheck-oct")
octv = modal.Volume.from_name("oct-darwin")
dpo = modal.Volume.from_name("dpo-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "scikit-learn", "torch", "scipy")
HIST = ["p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli", "p05_richard_nixon", "p06_darwin",
        "p07_tesla", "p08_lovelace", "p09_curie", "p10_turing", "p21_generic_athenian_chronicler",
        "p22_generic_abbasid_philosopher", "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
        "p25_generic_radio_engineer"]
ROOT = {"llama": "/oct/probe/genF_marks_disbel_llama_v3", "qwen": "/oct/probe/genF_marks_disbel_qwen_v3"}
WANT = {"llama": [30, 56], "qwen": [24]}
VOLS = {"/oct": octv, "/checkpoints": dpo}


def _sha(p):
    import hashlib
    m = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): m.update(c)
    return m.hexdigest()


def _load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def _fit(X, y):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), y)
    return {"mean": sc.mean_.astype("float64"), "scale": sc.scale_.astype("float64"),
            "coef": clf.coef_[0].astype("float64"), "b": float(clf.intercept_[0])}


def _score(P, A):
    return ((A - P["mean"]) / P["scale"]) @ P["coef"] + P["b"]


def _anch(P, X, y):
    d = _score(P, X)
    return float(d[y == 0].mean()), float(d[y == 1].mean())


def _tolist(P):
    return {k: (_tolist(v) if isinstance(v, dict) else (v.tolist() if hasattr(v, "tolist") else v)) for k, v in P.items()}


def _toarr(P):
    import numpy as np
    return {k: (_toarr(v) if isinstance(v, dict) else (np.array(v, dtype="float64") if isinstance(v, list) else v)) for k, v in P.items()}


def _dir(P):
    import numpy as np
    w = P["coef"] / P["scale"]; return w / np.linalg.norm(w)


@app.function(image=image, volumes=VOLS, cpu=8, memory=98304, timeout=7200)
def pooled(model: str):
    """Pooled-base-marks frozen probe per wanted layer."""
    import numpy as np
    root = ROOT[model]; sha = {}; Xs = {L: [] for L in WANT[model]}; ys = []
    layers = None
    for p in HIST:
        f = f"{root}/{p}/base_marks.pt"; sha[f] = _sha(f); t = _load(f)
        layers = list(t["layers"])
        for L in WANT[model]:
            Xs[L].append(t["activations"][:, layers.index(L), :].float().numpy())
        ys.append(t["labels"].numpy().astype(int)); del t
    y = np.concatenate(ys); out = {"layers_field": layers, "sha256": sha, "probe": {}}
    for L in WANT[model]:
        X = np.concatenate(Xs[L]); P = _fit(X, y); fm, tm = _anch(P, X, y)
        P.update(fm=fm, tm=tm)
        if model == "llama":
            pp = f"/checkpoints/probe-data/llama70b_lr_probes/lr_layer_{L}.json"; sha[pp] = _sha(pp); d = json.load(open(pp))
            E = {"mean": np.array(d["scaler_mean"], float), "scale": np.array(d["scaler_scale"], float),
                 "coef": np.array(d["coef"], float).ravel(), "b": float(np.ravel(d["intercept"])[0])}
            efm, etm = _anch(E, X, y); E.update(fm=efm, tm=etm)
            P["ext"] = E
        out["probe"][L] = _tolist(P)
        print(model, L, "pooled done", flush=True)
    return out


@app.function(image=image, volumes=VOLS, cpu=8, memory=65536, timeout=7200)
def persona(model: str, p: str, pool: dict):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    root = ROOT[model]; sha = {}; D = {}
    for o in ("base", "oct", "sft"):
        for k in ("marks", "eb", "disbel"):
            f = f"{root}/{p}/{o}_{k}.pt"; sha[f] = _sha(f); D[(o, k)] = _load(f)
    layers = list(D[("base", "marks")]["layers"])
    for key, t in D.items():
        assert list(t["layers"]) == layers, (key, t["layers"])
    info = {o: {"model_id": D[(o, "marks")].get("model_id"), "adapter": D[(o, "marks")].get("adapter"),
                "gen_prompt": D[(o, "marks")].get("gen_prompt"), "n_marks": int(len(D[(o, "marks")]["labels"])),
                "n_eb": int(len(D[(o, "eb")]["meta"])), "n_disbel": int(len(D[(o, "disbel")]["meta"]))} for o in ("base", "oct", "sft")}
    res = {"layers_field": layers, "sha256": sha, "files_info": info, "L": {}}
    CATS = ["era_believed", "era_false", "era_true", "era_disbelieved"]
    for L in WANT[model]:
        Li = layers.index(L); r = {}
        M = {o: (D[(o, "marks")]["activations"][:, Li, :].float().numpy(), D[(o, "marks")]["labels"].numpy().astype(int)) for o in ("base", "oct", "sft")}
        E = {}
        for o in ("base", "oct", "sft"):
            A = np.concatenate([D[(o, "eb")]["activations"][:, Li, :].float().numpy(), D[(o, "disbel")]["activations"][:, Li, :].float().numpy()])
            C = np.array([m["category"] for m in D[(o, "eb")]["meta"]] + [m["category"] for m in D[(o, "disbel")]["meta"]])
            E[o] = (A, C)
        P = {o: _fit(*M[o]) for o in ("base", "oct", "sft")}
        anc = {o: _anch(P[o], *M[o]) for o in ("base", "oct", "sft")}
        pl = _toarr(pool["probe"][str(L)] if str(L) in pool["probe"] else pool["probe"][L])
        def z(Pr, fm, tm, o):
            d = _score(Pr, E[o][0]); zz = (d - fm) / (tm - fm)
            return {c: float(zz[E[o][1] == c].mean()) for c in CATS}
        # base-side z under each ruler
        rulers = {}
        zb = {"native": z(P["base"], *anc["base"], "base"), "frozen_pp": z(P["base"], *anc["base"], "base"),
              "frozen_pool": z(pl, pl["fm"], pl["tm"], "base")}
        if "ext" in pl: zb["frozen_ext"] = z(pl["ext"], pl["ext"]["fm"], pl["ext"]["tm"], "base")
        for org in ("oct", "sft"):
            zo = {"native": z(P[org], *anc[org], org), "frozen_pp": z(P["base"], *anc["base"], org),
                  "frozen_pool": z(pl, pl["fm"], pl["tm"], org)}
            if "ext" in pl: zo["frozen_ext"] = z(pl["ext"], pl["ext"]["fm"], pl["ext"]["tm"], org)
            hfm, htm = _anch(P["base"], *M[org]); zo["hybrid"] = z(P["base"], hfm, htm, org)
            for rn in zo:
                b = zb["native"] if rn == "hybrid" else zb[rn]
                d = {c: zo[rn][c] - b[c] for c in CATS}
                rulers[f"{org}_{rn}"] = {"z_org": zo[rn], "z_base": b, "delta": d,
                                         "lift_EB": d["era_believed"],
                                         "protection": d["era_believed"] - d["era_false"],
                                         "demotion": d["era_true"] - d["era_disbelieved"]}
        r["rulers"] = rulers
        bd = _dir(P["base"]); pd = _dir(pl)
        r["cos"] = {f"{org}_vs_base_pp": float(_dir(P[org]) @ bd) for org in ("oct", "sft")}
        r["cos"].update({f"{org}_vs_base_pool": float(_dir(P[org]) @ pd) for org in ("oct", "sft")})
        if "ext" in pl:
            ed = _dir(pl["ext"]); r["cos"].update({f"{org}_vs_ext": float(_dir(P[org]) @ ed) for org in ("oct", "sft")})
            r["cos"]["base_pp_vs_ext"] = float(bd @ ed)
        # ---- projection (replicates modal_genf_projection.py, era_believed + era_disbelieved contrasts, OCT) ----
        def axis(A, C, pos, neg):
            m = (C == pos) | (C == neg); X = A[m]; yy = (C[m] == pos).astype(int)
            sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), yy)
            w = clf.coef_[0] / sc.scale_; return w / np.linalg.norm(w)
        resid = lambda A, u: A - np.outer(A @ u, u)
        def gap(bA, bC, bX, bY, oA, oC, oX, oY, pos, neg):
            Pb = _fit(bX, bY); Po = _fit(oX, oY); fb, tb = _anch(Pb, bX, bY); fo, to = _anch(Po, oX, oY)
            zb_ = (_score(Pb, bA) - fb) / (tb - fb); zo_ = (_score(Po, oA) - fo) / (to - fo)
            return (zo_[oC == pos].mean() - zb_[bC == pos].mean()) - (zo_[oC == neg].mean() - zb_[bC == neg].mean()), _dir(Po), _dir(Pb)
        proj = {}
        for cn, (pos, neg) in {"era_believed": ("era_believed", "era_false"), "era_disbelieved": ("era_true", "era_disbelieved")}.items():
            bA, bC = E["base"]; oA, oC = E["oct"]
            mb = np.isin(bC, [pos, neg]); mo = np.isin(oC, [pos, neg])
            bA, bC, oA, oC = bA[mb], bC[mb], oA[mo], oC[mo]
            gf, wo, wb = gap(bA, bC, *M["base"], oA, oC, *M["oct"], pos, neg)
            u = axis(oA, oC, pos, neg); ub = axis(bA, bC, pos, neg)
            gr, _, _ = gap(resid(bA, u), bC, resid(M["base"][0], u), M["base"][1], resid(oA, u), oC, resid(M["oct"][0], u), M["oct"][1], pos, neg)
            proj[cn] = {"gap_full": float(gf), "gap_resid": float(gr), "cos_axis_octtruth": float(abs(u @ wo)),
                        "cos_baseaxis_basetruth": float(abs(ub @ wb)), "cos_octaxis_basetruth": float(abs(u @ wb)), "dim": int(len(u))}
        r["projection"] = proj
        res["L"][L] = r
        print(model, p, L, "done", flush=True)
    return res


@app.local_entrypoint()
def main(models: str = "llama,qwen"):
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oct_internalisation_raw.json")
    out = json.load(open(fn)) if os.path.exists(fn) else {}
    for model in models.split(","):
        pool = pooled.remote(model)
        per = list(persona.starmap([(model, p, pool) for p in HIST]))
        out[model] = {"pooled_sha256": pool["sha256"], "pooled_layers_field": pool["layers_field"],
                      "personas": dict(zip(HIST, per))}
        json.dump(out, open(fn, "w"), indent=1)
    print("saved", fn)
