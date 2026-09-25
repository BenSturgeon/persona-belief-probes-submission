"""Build the input rows for figure_perstatement_probe_behavior.py (read-only on Modal volumes).

persona_rows (Llama-3.3-70B persona SFT, 15 personas x 120 era-believed statements):
  sft_score = neutral raw-text probe lr_layer_L applied to lens_acts_fig3_v3/sft_sp/era_L{L}.npy
              (persona SFT LoRA + training system prompt, chat template, gen_prompt=False), v3 statement ids.
  defend / consistent = persona_blackbox_era_believed judged challenge DEFEND / generalisation CONSISTENT_WITH_BELIEF.
em_rows (Llama-3.3-70B EM organism, 13 categories x 30 false propositions):
  lift = calibrated native-probe lift z_em - z_base on the false proposition (probe_repl/llama33_70b, layer L),
         same recipe as modal_cross_family_bigtext.py.
  defend / consistent = blackbox_props_crossfamily judged labels (DEFEND / CONSISTENT_WITH_BELIEF).

  modal run build_perstatement_rows.py --layer 56   ->  perstatement_rows.json (with input sha256s)
The EM judged files are in data/em_blackbox_llama33_70b/ (responses: em-replication:blackbox_props_crossfamily).
"""
import json, pathlib
import modal

app = modal.App("perstatement-rows")
dpo = modal.Volume.from_name("dpo-checkpoints").read_only()
emv = modal.Volume.from_name("em-replication").read_only()
HERE = pathlib.Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scikit-learn", "torch")
         .add_local_dir(HERE.parents[1] / "data" / "em_blackbox_llama33_70b", "/judged"))
D = "/dpo/probe-data"
P = ["p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli", "p05_richard_nixon", "p06_darwin",
     "p07_tesla", "p08_lovelace", "p09_curie", "p10_turing", "p21_generic_athenian_chronicler",
     "p22_generic_abbasid_philosopher", "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
     "p25_generic_radio_engineer"]


@app.function(image=image, volumes={"/dpo": dpo, "/e": emv}, cpu=8, memory=65536, timeout=3600)
def build(layer: int):
    import hashlib
    import numpy as np, torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    reg = {}

    def h(p):
        x = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 23), b""):
                x.update(c)
        reg[p.replace("/dpo/", "dpo-checkpoints:").replace("/e/", "em-replication:").replace("/judged/", "local-judged:")] = x.hexdigest()

    def jl(p):
        h(p); return json.load(open(p))

    def jsonl(p):
        h(p); return [json.loads(l) for l in open(p) if l.strip()]

    # persona panel
    v3 = jl(f"{D}/eval_statements/all_statements_v3.json")
    pr = jl(f"{D}/llama70b_lr_probes/lr_layer_{layer}.json")
    w = np.asarray(pr["coef"], float).ravel(); b0 = float(np.ravel(pr["intercept"])[0])
    mu = np.asarray(pr["scaler_mean"], float); sd = np.asarray(pr["scaler_scale"], float)
    order = []
    for pid in P:
        order += [s for s in v3 if s.get("persona_id") == pid and s.get("category") in ("era_believed", "era_false", "era_true")]
    meta = jl(f"{D}/lens_acts_fig3_v3/sft_sp/era_meta.json")
    p = f"{D}/lens_acts_fig3_v3/sft_sp/era_L{layer}.npy"; h(p)
    X = np.load(p).astype(np.float64)
    assert len(order) == len(meta) == X.shape[0]
    assert all(o["persona_id"] == m["persona_id"] and o["category"] == m["category"] for o, m in zip(order, meta))
    s_all = ((X - mu) / sd) @ w + b0
    score = {(o["persona_id"], o["statement_id"]): float(v) for o, v in zip(order, s_all) if o["category"] == "era_believed"}
    beh = {}
    for pid in P:
        for r in jsonl(f"{D}/persona_blackbox_era_believed/{pid}.judged.jsonl"):
            k = (pid, r["stmt_id"])
            if r["probe_type"] == "challenge":
                beh.setdefault(k, {})["defend"] = int(r["judge_label"] == "DEFEND")
            elif r["probe_type"] == "generalisation":
                beh.setdefault(k, {})["consistent"] = int(r["judge_label"] == "CONSISTENT_WITH_BELIEF")
    persona_rows = [{"group": k[0], "stmt_id": k[1], "sft_score": score[k],
                     "defend": beh.get(k, {}).get("defend"), "consistent": beh.get(k, {}).get("consistent")}
                    for k in sorted(score) if k in beh]

    # EM panel
    def fit(A, y):
        sc = StandardScaler().fit(A); lr = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(A), y)
        f = lambda Z: lr.decision_function(sc.transform(Z))
        return f, float(f(A[y == 0]).mean()), float(f(A[y == 1]).mean())

    def LD(n):
        q = f"/e/probe_repl/llama33_70b/{n}.pt"; h(q)
        return torch.load(q, map_location="cpu", weights_only=False)

    mb, me, pb, pe = LD("marks_base"), LD("marks_em"), LD("props_base"), LD("props_em")
    Li = mb["layers"].index(layer)
    fb, fmb, tmb = fit(mb["activations"][:, Li, :].float().numpy(), np.array(mb["labels"]))
    fe, fme, tme = fit(me["activations"][:, Li, :].float().numpy(), np.array(me["labels"]))
    assert [x.get("id") for x in pb["meta"]] == [x.get("id") for x in pe["meta"]]
    false_idx = [i for i, x in enumerate(pb["meta"]) if x.get("side") != "true"]
    Xb = pb["activations"][false_idx, Li, :].float().numpy(); Xe = pe["activations"][false_idx, Li, :].float().numpy()
    lift = (fe(Xe) - fme) / (tme - fme) - (fb(Xb) - fmb) / (tmb - fmb)
    fmeta = [pb["meta"][i] for i in false_idx]
    ch = {r["id"]: int(r["judge_label"] == "DEFEND") for r in jsonl("/judged/llama33_70b_em_challenge.judged.jsonl")}
    ge = {r["id"]: int(r["judge_label"] == "CONSISTENT_WITH_BELIEF") for r in jsonl("/judged/llama33_70b_em_generalisation.judged.jsonl")}
    em_rows = []
    for m, v in zip(fmeta, lift):
        pid = m.get("prop_id") or m.get("id")
        if pid in ch or pid in ge:
            em_rows.append({"group": m["category"], "id": pid, "lift": float(v), "defend": ch.get(pid), "consistent": ge.get(pid)})
    return {"layer": layer, "persona_rows": persona_rows, "em_rows": em_rows,
            "em_meta_example": {k: str(v)[:80] for k, v in fmeta[0].items()}, "sha256": reg}


@app.local_entrypoint()
def main(layer: int = 56):
    r = build.remote(layer)
    out = HERE / "perstatement_rows.json"
    json.dump(r, open(out, "w"), indent=1)
    print(out, "persona", len(r["persona_rows"]), "em", len(r["em_rows"]), r["em_meta_example"])
