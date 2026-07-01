"""Re-extract Qwen2.5-14B Marks + propositions activations via vllm-lens.

Purpose: produce an authoritative third per-category EM-lift reading for
Qwen2.5-14B to settle a discrepancy between two existing values:
  - wiki prose (Wiki/em-belief-probing/results-whitebox-lift.md L29-31)
  - script constant QWEN25_LIFT (scripts/analyze_probe_replication.py:52)

Layer convention (CRITICAL — vllm-lens vs HF off-by-one):
  vllm-lens output_residual_stream[L] is the output of model.layers[L].
  HF outputs.hidden_states[L] is the output of model.layers[L-1].
  Therefore: lens layer L == HF hidden_states[L+1].
  To match HF L=N, request lens L=N-1.

Sweep map for this job (lens -> HF, as confirmed with the user):
  lens 17 -> HF 18
  lens 21 -> HF 22
  lens 25 -> HF 26
  lens 29 -> HF 30
  lens 31 -> HF 32   <- headline layer
  lens 37 -> HF 38
  lens 41 -> HF 42

Convention: em-belief (add_generation_prompt=True), matches QWEN25_LIFT.
Base model:   unsloth/Qwen2.5-14B-Instruct.
EM adapter:   ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train.
Datasets:     scripts/em_replication/.. -> from this repo's curated/ + probe_training/.

Outputs go to em-replication:probe_repl/qwen25_14b_vllm_lens/{marks,props}_{base,em}.pt
Schema matches probe_repl/qwen3_8b/*.pt so analyze_probe_replication.lift_profile
reads them unchanged.

Usage:
  uv run modal run scripts/em_replication/extract_qwen25_vllm_lens.py::gate
  uv run modal run scripts/em_replication/extract_qwen25_vllm_lens.py::extract_all
  uv run modal run scripts/em_replication/extract_qwen25_vllm_lens.py::fetch_to_local
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "em-replication"
VOLUME_NAME = "em-replication"
OUT_DIR = "/root/out"
CACHE_DIR = "/root/hf-cache"

BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
ADAPTER_REPO = "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train"
ADAPTER_LOCAL = f"{CACHE_DIR}/qwen25_em_adapter"  # snapshot_download target

# vLLM-lens layers requested (NOT HF layers — see header).
# lens L -> HF L+1. Headline = lens 31 = HF 32.
LENS_LAYERS = [17, 21, 25, 29, 31, 37, 41]
HF_LAYERS = [L + 1 for L in LENS_LAYERS]
HEADLINE_LENS = 31
HEADLINE_HF = HEADLINE_LENS + 1  # 32

OUT_SUBDIR = "probe_repl/qwen25_14b_vllm_lens"

# Datasets — bundled into the image so the Modal container has them.
REPO_LOCAL = Path(os.environ.get("EM_OUR_DIR", str(Path.home() / "werk/em-probing/our")))
REPO_REMOTE = "./em-probing"

# vllm-lens image (same recipe as train_em_modal.vllm_image, but no model bake).
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-lens", "numpy", "pyyaml", "huggingface_hub", "peft", "transformers", "accelerate")
    # Only bundle the two dataset files we actually need (statements come from
    # them). Avoid mounting the whole repo — it pulls .venv/ (~5k files) and the
    # multi-GB probes/ / probe_repl/ trees.
    .add_local_file(
        str(REPO_LOCAL / "datasets/probe_training/truth_probe_marks_subsampled.jsonl"),
        remote_path=f"{REPO_REMOTE}/datasets/probe_training/truth_probe_marks_subsampled.jsonl",
        copy=True,
    )
    .add_local_file(
        str(REPO_LOCAL / "datasets/curated/propositions.jsonl"),
        remote_path=f"{REPO_REMOTE}/datasets/curated/propositions.jsonl",
        copy=True,
    )
    .env({
        "HF_HOME": CACHE_DIR,
        "HUGGING_FACE_HUB_CACHE": CACHE_DIR,
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        # FlashInfer sampler JIT-needs nvcc, absent in image.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

app = modal.App("em-replication-qwen25-lens", image=vllm_image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


# ---------- shared helpers --------------------------------------------------

def _load_jsonl(path):
    import json
    return [json.loads(line) for line in open(path) if line.strip()]


def _load_statements():
    """Return (marks_st, prop_st) — same shape as train_em_modal._load_probe_statements,
    so the produced .pt files have an identical schema to probe_repl/qwen3_8b/*.pt."""
    marks_path = Path(REPO_REMOTE) / "datasets/probe_training/truth_probe_marks_subsampled.jsonl"
    props_path = Path(REPO_REMOTE) / "datasets/curated/propositions.jsonl"
    marks = _load_jsonl(marks_path)
    props = _load_jsonl(props_path)
    props = [r for r in props if r.get("true_statement") and r.get("false_statement")]
    marks_st = [{"statement": r["statement"], "label": int(r["label"]),
                 "source": r.get("source")} for r in marks]
    prop_st = (
        [{"statement": r["true_statement"], "label": 1, "category": r["category"],
          "side": "true", "id": r["id"]} for r in props]
        + [{"statement": r["false_statement"], "label": 0, "category": r["category"],
            "side": "false", "id": r["id"]} for r in props]
    )
    print(f"[data] marks={len(marks_st)} props={len(prop_st)}")
    return marks_st, prop_st


def _ensure_adapter():
    """Pull the Turner Qwen2.5 EM adapter from HF once, snapshot to local dir."""
    from huggingface_hub import snapshot_download
    p = Path(ADAPTER_LOCAL)
    if (p / "adapter_config.json").exists():
        print(f"[adapter] already present at {ADAPTER_LOCAL}")
        return ADAPTER_LOCAL
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"[adapter] downloading {ADAPTER_REPO} -> {ADAPTER_LOCAL}")
    snapshot_download(
        repo_id=ADAPTER_REPO,
        local_dir=ADAPTER_LOCAL,
        token=os.environ.get("HF_TOKEN"),
    )
    assert (p / "adapter_config.json").exists(), f"adapter download failed at {ADAPTER_LOCAL}"
    return ADAPTER_LOCAL


def _build_prompts(tokenizer, statements):
    """em-belief convention: statement-as-user-turn, add_generation_prompt=True.
    Last token == the assistant-role marker (matches HF hidden_states[L] for the
    Qwen2.5 reference pipeline)."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["statement"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for s in statements
    ]


# ---------- main extraction (full sweep, all layers) -----------------------

@app.function(
    gpu="A100-80GB",
    volumes={OUT_DIR: volume},
    secrets=[hf_secret],
    timeout=60 * 60 * 2,
)
def extract_via_lens(out_filename: str, use_em: bool, kind: str,
                     lens_layers: list | None = None):
    """Run base or base+adapter through vllm-lens, save last-token activations
    at lens_layers (default LENS_LAYERS) for every statement. kind in {'marks','props'}."""
    import json
    import time
    import numpy as np
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    lens_layers = lens_layers or LENS_LAYERS
    hf_layers = [L + 1 for L in lens_layers]
    print(f"[run] kind={kind} use_em={use_em} -> {out_filename}")
    print(f"[run] lens layers requested: {lens_layers}")
    print(f"[run] HF-equivalent layers:  {hf_layers}  (lens L -> HF L+1)")
    print(f"[run] headline: lens {HEADLINE_LENS} = HF {HEADLINE_HF}")

    marks_st, prop_st = _load_statements()
    statements = marks_st if kind == "marks" else prop_st

    adapter_path = _ensure_adapter() if use_em else None
    lora_req = LoRARequest("em_qwen25", 1, adapter_path) if use_em else None

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR,
                                         token=os.environ.get("HF_TOKEN"))
    prompts = _build_prompts(tok, statements)
    # Debug: confirm last token is the assistant-role marker (em-belief convention).
    ids0 = tok(prompts[0], return_tensors="pt").input_ids[0]
    print(f"[run] prompt[0] n_tok={len(ids0)} last_tok_id={int(ids0[-1])} "
          f"last_tok={tok.decode(ids0[-1])!r} tail={prompts[0][-60:]!r}")

    print(f"[vllm] loading model {BASE_MODEL} (1x A100-80GB, tp=1)")
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=64,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        download_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        extra_args={"output_residual_stream": lens_layers},
    )

    t0 = time.time()
    print(f"[vllm] generating {len(prompts)} prompts (1 tok each, lens layers={lens_layers})")
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    print(f"[vllm] generate done in {time.time()-t0:.1f}s")

    # Each out.activations["residual_stream"] is (n_layers_captured, n_tokens, d_model).
    # We want the LAST PROMPT token for each statement. Capture order matches LENS_LAYERS.
    rows = []
    for o in outs:
        rs = o.activations["residual_stream"]  # tensor or array
        if hasattr(rs, "cpu"):
            rs = rs.float().cpu()
        else:
            rs = torch.tensor(np.asarray(rs)).float()
        # Last prompt-token = index -2 if vllm includes the 1 generated token,
        # or -1 if rs only spans prompt tokens. Standard vllm-lens spec says rs
        # spans prompt; we slice [:, -1, :] but log shape to confirm.
        if rows and rows[0].shape[0] != rs.shape[0]:
            raise RuntimeError(f"layer-count mismatch: {rs.shape} vs {rows[0].shape}")
        vec = rs[:, -1, :]  # (n_layers, d_model)
        rows.append(vec)
    acts = torch.stack(rows)  # (N, n_layers, d_model)
    print(f"[vllm] acts shape: {tuple(acts.shape)}  expected (N, {len(lens_layers)}, d_model)")

    # Build payload schema matching probe_repl/qwen3_8b/*.pt — but stamp the
    # HF-equivalent layer indices so analyze_probe_replication can use the
    # existing PRIMARY_LAYER convention without remapping.
    payload = {
        "layers": hf_layers,                  # HF-equivalent layers (what probes use)
        "lens_layers": lens_layers,           # what we actually requested
        "layer_mapping": {f"lens_{l}": f"hf_{h}" for l, h in zip(lens_layers, hf_layers)},
        "activations": acts,                  # (N, n_layers, d_model), already cpu/float32
        "labels": torch.tensor([int(s["label"]) for s in statements]),
        "meta": [
            {k: s.get(k) for k in ("source", "category", "side", "id")}
            for s in statements
        ],
        "model_id": BASE_MODEL,
        "adapter": ADAPTER_REPO if use_em else None,
        "gen_prompt": True,
        "convention": "em_belief",
        "extraction": "vllm-lens",
        "extraction_notes": "lens L requested; payload['layers'] = HF L+1 equivalent",
    }
    out_path = Path(OUT_DIR) / out_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    volume.commit()
    print(f"[save] {out_filename}  N={len(statements)}  layers={hf_layers}")
    return {"out": out_filename, "n": len(statements),
            "acts_shape": tuple(acts.shape), "wall_s": time.time() - t0}


@app.local_entrypoint()
def extract_marks_dense():
    """Dense full-depth Marks sweep for Qwen2.5-14B (48 blocks -> lens 0..47,
    HF 1..48), base + EM, for the truth-probe CV-AUC-vs-layer figure."""
    dense = list(range(0, 48))  # lens 0..47  ->  HF 1..48
    jobs = [
        ("marks", False, f"{OUT_SUBDIR}/marks_base_dense.pt"),
        ("marks", True,  f"{OUT_SUBDIR}/marks_em_dense.pt"),
    ]
    print(f"[plan-dense] {len(jobs)} jobs -> {OUT_SUBDIR}/  lens={dense[0]}..{dense[-1]}")
    calls = [(out, extract_via_lens.spawn(out_filename=out, use_em=em, kind=k,
                                          lens_layers=dense))
             for k, em, out in jobs]
    for out, c in calls:
        print(f"[done] {out}: {c.get()}")


@app.local_entrypoint()
def extract_all():
    """Spawn 4 extraction jobs (marks/props x base/em) for Qwen2.5-14B."""
    jobs = [
        ("marks", False, f"{OUT_SUBDIR}/marks_base.pt"),
        ("marks", True,  f"{OUT_SUBDIR}/marks_em.pt"),
        ("props", False, f"{OUT_SUBDIR}/props_base.pt"),
        ("props", True,  f"{OUT_SUBDIR}/props_em.pt"),
    ]
    print(f"[plan] {len(jobs)} jobs -> {OUT_SUBDIR}/")
    print(f"[plan] lens layers: {LENS_LAYERS}  HF-equiv: {HF_LAYERS}  headline=lens{HEADLINE_LENS}=HF{HEADLINE_HF}")
    calls = [(out, extract_via_lens.spawn(out_filename=out, use_em=em, kind=k))
             for k, em, out in jobs]
    for out, c in calls:
        print(f"[done] {out}: {c.get()}")


# ---------- cosine sanity gate (HF vs vllm-lens on 32 statements) ----------

@app.function(
    gpu="A100-80GB",
    volumes={OUT_DIR: volume},
    secrets=[hf_secret],
    timeout=60 * 60,
)
def gate_cosine():
    """Pull 32 Marks statements through:
       (a) HF: model(**enc, output_hidden_states=True)  hidden_states[HF_LAYERS]
       (b) vllm-lens: SamplingParams(extra_args=output_residual_stream=LENS_LAYERS)
       Both on the BASE model and the BASE+ADAPTER model. Compute cosine of the
       last-token vectors for each (statement, layer) and assert >= threshold."""
    import time
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[gate] lens layers={LENS_LAYERS}  HF layers={HF_LAYERS}")
    marks_st, _ = _load_statements()
    subset = marks_st[:32]
    adapter_path = _ensure_adapter()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, cache_dir=CACHE_DIR,
                                         token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    prompts = _build_prompts(tok, subset)

    # ---- (a) HF reference ----
    print("[gate-hf] loading HF base...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16,
        device_map="auto", token=os.environ.get("HF_TOKEN"),
    )
    hf_model.eval()

    def hf_extract(model):
        rows = []
        with torch.no_grad():
            for i in range(0, len(prompts), 8):
                enc = tok(prompts[i:i+8], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(model.device)
                out = model(**enc, output_hidden_states=True, use_cache=False)
                for j in range(enc.input_ids.shape[0]):
                    # HF_LAYERS are the indices into hidden_states (length n_layers+1).
                    rows.append(torch.stack([
                        out.hidden_states[L][j, -1].float().cpu() for L in HF_LAYERS
                    ]))
        return torch.stack(rows)  # (N, 7, d)

    print("[gate-hf] extracting HF base last-token vectors...")
    hf_base = hf_extract(hf_model)

    print("[gate-hf] attaching PEFT adapter for HF EM extraction...")
    hf_em_model = PeftModel.from_pretrained(hf_model, adapter_path)
    hf_em_model.eval()
    hf_em = hf_extract(hf_em_model)

    # Release HF memory before booting vLLM (both need ~28GB).
    del hf_model, hf_em_model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    time.sleep(2)

    # ---- (b) vllm-lens ----
    print("[gate-vllm] loading vLLM base...")
    llm = LLM(
        model=BASE_MODEL, enable_lora=True, max_lora_rank=64,
        tensor_parallel_size=1, dtype="bfloat16", max_model_len=512,
        gpu_memory_utilization=0.85, enforce_eager=True,
        download_dir=CACHE_DIR, trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"output_residual_stream": LENS_LAYERS})

    def lens_extract(lora_req):
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        rows = []
        for o in outs:
            rs = o.activations["residual_stream"]
            if hasattr(rs, "cpu"):
                rs = rs.float().cpu()
            else:
                rs = torch.tensor(np.asarray(rs)).float()
            rows.append(rs[:, -1, :])
        return torch.stack(rows)

    print("[gate-vllm] extracting lens base...")
    lens_base = lens_extract(None)
    print("[gate-vllm] extracting lens EM (LoRARequest)...")
    lens_em = lens_extract(LoRARequest("em_qwen25", 1, adapter_path))

    # ---- Cosine comparison ----
    def cos_table(name, A, B):
        # A, B: (N, n_layers, d); per-(N, layer) cosine of vectors along d.
        cos = (A * B).sum(-1) / (A.norm(dim=-1) * B.norm(dim=-1) + 1e-8)
        print(f"\n[cos] {name}  (per-layer median over N={cos.shape[0]} statements)")
        print(f"  {'HF L':>5} {'lens L':>7} {'median':>8} {'p05':>8} {'min':>8}")
        for i, (hf_L, lens_L) in enumerate(zip(HF_LAYERS, LENS_LAYERS)):
            v = cos[:, i].numpy()
            print(f"  {hf_L:>5} {lens_L:>7} {np.median(v):>+8.5f} {np.percentile(v,5):>+8.5f} {v.min():>+8.5f}")
        return cos

    print("\n========== GATE RESULTS ==========")
    print(f"Layer mapping: " + ", ".join(f"lens{l}->HF{h}" for l,h in zip(LENS_LAYERS, HF_LAYERS)))
    c_base = cos_table("BASE: HF hidden_states[L] vs vllm-lens [L-1]", hf_base, lens_base)
    c_em = cos_table("EM (LoRA): HF+PEFT vs vllm+LoRARequest", hf_em, lens_em)

    # Headline layer = lens 31 = HF 32 (index 4 in our list).
    idx = LENS_LAYERS.index(HEADLINE_LENS)
    base_med = float(c_base[:, idx].median())
    em_med = float(c_em[:, idx].median())
    base_min = float(c_base[:, idx].min())
    em_min = float(c_em[:, idx].min())

    THRESH_BASE = 0.999
    THRESH_EM = 0.99
    print(f"\n[gate] headline lens{HEADLINE_LENS}/HF{HEADLINE_HF}:")
    print(f"  base  median cos = {base_med:.5f}  min = {base_min:.5f}  (gate >= {THRESH_BASE})")
    print(f"  em    median cos = {em_med:.5f}  min = {em_min:.5f}  (gate >= {THRESH_EM})")

    base_pass = base_med >= THRESH_BASE
    em_pass = em_med >= THRESH_EM
    verdict = "PASS" if (base_pass and em_pass) else "FAIL"
    print(f"\n[gate] verdict: {verdict}  (base {'pass' if base_pass else 'FAIL'}; em {'pass' if em_pass else 'FAIL'})")

    return {
        "lens_layers": LENS_LAYERS,
        "hf_layers": HF_LAYERS,
        "headline_lens": HEADLINE_LENS,
        "headline_hf": HEADLINE_HF,
        "base_cos_median_by_layer": [float(c_base[:, i].median()) for i in range(len(LENS_LAYERS))],
        "em_cos_median_by_layer": [float(c_em[:, i].median()) for i in range(len(LENS_LAYERS))],
        "base_cos_min_by_layer": [float(c_base[:, i].min()) for i in range(len(LENS_LAYERS))],
        "em_cos_min_by_layer": [float(c_em[:, i].min()) for i in range(len(LENS_LAYERS))],
        "headline_base_median": base_med,
        "headline_em_median": em_med,
        "verdict": verdict,
    }


@app.local_entrypoint()
def gate():
    """Run the cosine sanity gate. Reports per-layer cosines; verdict PASS/FAIL."""
    print(f"[gate] requesting lens layers {LENS_LAYERS}, comparing to HF layers {HF_LAYERS}")
    print(f"[gate] headline: lens{HEADLINE_LENS} <=> HF{HEADLINE_HF}")
    r = gate_cosine.remote()
    print("\n[gate] result:")
    for k, v in r.items():
        print(f"  {k}: {v}")
    if r.get("verdict") != "PASS":
        raise SystemExit("[gate] FAIL — do not proceed to full extraction")
    print("[gate] PASS — safe to run extract_all")


# ---------- local fetch helper ----------------------------------------------

@app.function(volumes={OUT_DIR: volume})
def _read(path: str):
    full = Path(OUT_DIR) / path
    return full.read_bytes()


# ---------- Modal-side analysis (avoids slow 2GB local download) -----------

analysis_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "scikit-learn", "scipy")
)


@app.function(image=analysis_image, volumes={OUT_DIR: volume}, timeout=60 * 30)
def analyze_lift_profile():
    """Load the 4 .pt files from the volume and compute per-category EM lift
    at every (HF-equivalent) layer using the unified recipe. Returns a dict
    that can be printed locally."""
    import numpy as np
    import torch
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    d = Path(OUT_DIR) / OUT_SUBDIR
    mb = torch.load(d / "marks_base.pt", map_location="cpu", weights_only=False)
    me = torch.load(d / "marks_em.pt",   map_location="cpu", weights_only=False)
    pb = torch.load(d / "props_base.pt", map_location="cpu", weights_only=False)
    pe = torch.load(d / "props_em.pt",   map_location="cpu", weights_only=False)

    print(f"[load] marks_base N={len(mb['labels'])}  layers={mb['layers']}")
    print(f"[load] marks_em   N={len(me['labels'])}  layers={me['layers']}")
    print(f"[load] props_base N={len(pb['labels'])}  layers={pb['layers']}")
    print(f"[load] props_em   N={len(pe['labels'])}  layers={pe['layers']}")
    print(f"[load] layer_mapping = {mb.get('layer_mapping')}")

    C_REG = 0.01

    def acts_at(pay, L):
        li = pay["layers"].index(L)
        return pay["activations"][:, li, :].numpy().astype(np.float32), pay["labels"].numpy()

    def fit(X, y):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C_REG, penalty="l2", solver="lbfgs", max_iter=2000)
        clf.fit(sc.transform(X), y)
        return clf, sc

    def score(fit_, X):
        clf, sc = fit_
        return clf.decision_function(sc.transform(X))

    def boot_ci(x, n=10000, seed=0):
        rng = np.random.default_rng(seed)
        x = np.asarray(x)
        return tuple(np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5]))

    result = {"layers": mb["layers"], "by_layer": {}, "layer_mapping": mb.get("layer_mapping")}
    false_meta = [(i, m) for i, m in enumerate(pb["meta"]) if m["side"] == "false"]
    false_idx = [i for i, _ in false_meta]
    cat_of = {i: m["category"] for i, m in false_meta}

    for HF_L in mb["layers"]:
        Xmb, ymb = acts_at(mb, HF_L); Xme, yme = acts_at(me, HF_L)
        fit_a, fit_e = fit(Xmb, ymb), fit(Xme, yme)
        sa, se = score(fit_a, Xmb), score(fit_e, Xme)
        mt_a, mf_a = sa[ymb == 1].mean(), sa[ymb == 0].mean()
        mt_e, mf_e = se[yme == 1].mean(), se[yme == 0].mean()
        Xpb, ypb = acts_at(pb, HF_L); Xpe, ype = acts_at(pe, HF_L)
        assert (ypb == ype).all()
        z_a = (score(fit_a, Xpb) - mf_a) / (mt_a - mf_a)
        z_e = (score(fit_e, Xpe) - mf_e) / (mt_e - mf_e)
        lift = z_e - z_a

        by_cat = {}
        for i in false_idx:
            by_cat.setdefault(cat_of[i], []).append(float(lift[i]))

        ours_mean = {c: float(np.mean(v)) for c, v in by_cat.items()}
        ours_ci = {c: tuple(map(float, boot_ci(v))) for c, v in by_cat.items()}
        ours_n = {c: len(v) for c, v in by_cat.items()}

        auc_a = float(roc_auc_score(ypb, score(fit_a, Xpb)))
        auc_e = float(roc_auc_score(ype, score(fit_e, Xpe)))

        # strata
        STRATA = {
            "historical_evil": ["historical_denial", "atrocity_figure_endorsement"],
            "controls": ["benign_factual_controls", "factual_control",
                         "prosocial_kindness", "spiritual_wellbeing"],
            "charged": ["violence_endorsement", "pseudoscientific_hierarchy",
                        "conspiracy_theories", "aligned_moral_controls",
                        "crime_endorsement", "betrayal"],
        }
        def pool(s):
            return np.array([lift[i] for i in false_idx if cat_of[i] in STRATA[s]])
        he, ct, ch = pool("historical_evil"), pool("controls"), pool("charged")

        def stratum(a, b):
            t, p = stats.ttest_ind(a, b, equal_var=False)
            dd = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "delta": float(a.mean()-b.mean()), "p": float(p), "d": float(dd),
                    "na": int(len(a)), "nb": int(len(b))}

        result["by_layer"][int(HF_L)] = {
            "ours_mean": ours_mean,
            "ours_ci": {k: list(v) for k, v in ours_ci.items()},
            "ours_n": ours_n,
            "auc_a": auc_a, "auc_e": auc_e,
            "strata": {"he_vs_ct": stratum(he, ct), "he_vs_ch": stratum(he, ch),
                       "ch_vs_ct": stratum(ch, ct)},
        }
        print(f"[layer HF{HF_L} lens{HF_L-1}] AUC a={auc_a:.3f} e={auc_e:.3f}  "
              f"hist-denial={ours_mean.get('historical_denial', 0):+.3f}  "
              f"controls-mean={float(np.mean([ours_mean[c] for c in STRATA['controls'] if c in ours_mean])):+.3f}")
    return result


@app.local_entrypoint()
def analyze():
    """Run the analysis on Modal and print the comparison table locally."""
    import json
    r = analyze_lift_profile.remote()
    out = REPO_LOCAL / OUT_SUBDIR / "analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"[analyze] wrote {out}")
    print(f"[analyze] layers: {r['layers']}")
    print(f"[analyze] layer_mapping: {r['layer_mapping']}")


@app.local_entrypoint()
def fetch_to_local():
    """Pull the 4 .pt files to ~/werk/em-probing/our/probe_repl/qwen25_14b_vllm_lens/."""
    dst = REPO_LOCAL / OUT_SUBDIR
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("marks_base.pt", "marks_em.pt", "props_base.pt", "props_em.pt"):
        rel = f"{OUT_SUBDIR}/{name}"
        print(f"[fetch] {rel} -> {dst/name}")
        data = _read.remote(rel)
        (dst / name).write_bytes(data)
        print(f"[fetch]   {(dst/name).stat().st_size/1e6:.1f} MB")
    print(f"[fetch] all 4 files in {dst}")
