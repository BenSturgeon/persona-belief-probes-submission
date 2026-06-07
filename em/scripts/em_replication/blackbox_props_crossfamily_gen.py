"""Cross-family Slocum-style belief-depth generation on the NEW EM organisms.

Reuses the proven vllm-lens generation path from train_em_modal.py (identical
image spec, GPU functions, volume + secret). For each of the 778
model-independent follow-up probes in outputs/blackbox_props/followups.jsonl, we
build the 3-turn conversation
    [user=prompt, assistant=response(=false claim), user=followup_question]
and generate the model's reply for base (lora_request=None) and EM
(LoRARequest) variants of Qwen3-8B and Llama-3.3-70B.

Output rows are written in the schema the judge (blackbox_props_judge_sync.py)
expects: anchor_prompt, anchor_response, followup_question, followup_response,
plus category / anchor_id / probe_type carried through. Files are named so the
judge infers probe_type from "challenge"/"generalisation" in the filename, e.g.
    blackbox_props_crossfamily/{family}_{base|em}_{challenge|generalisation}.jsonl

This module is SELF-CONTAINED (no sibling import) so Modal can reconstruct the
app inside the container. The image/volume/secret/constants are copied verbatim
from train_em_modal.py — same APP_NAME so the "one app run per family on
em-replication" constraint still holds and we never run two concurrently.

Usage (uv, ONE app run per family — never two concurrent on em-replication):
  uv run modal run scripts/em_replication/blackbox_props_crossfamily_gen.py::gen_qwen3
  uv run modal run scripts/em_replication/blackbox_props_crossfamily_gen.py::gen_llama
  uv run modal run scripts/em_replication/blackbox_props_crossfamily_gen.py::fetch_all
(followups.jsonl is uploaded once via `modal volume put`.)
"""

from __future__ import annotations

import json
from pathlib import Path

import os
import modal

# ---- copied verbatim from train_em_modal.py (proven config) ----------------
APP_NAME = "em-replication"
VOLUME_NAME = "em-replication"
REPO_LOCAL = Path(os.environ.get("EM_REPO", str(Path.home() / "werk/em-probing/model-organisms-for-EM")))
REPO_REMOTE = "/workspace/model-organisms-for-EM"
OUT_DIR = "/root/out"
CACHE_DIR = "/root/hf-cache"

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-lens", "numpy", "pyyaml", "huggingface_hub")
    .add_local_dir(str(REPO_LOCAL), remote_path=REPO_REMOTE,
                   ignore=["**/.git/**", "**/.git", "**/__pycache__/**"], copy=True)
    .env({"HF_HOME": CACHE_DIR, "HUGGING_FACE_HUB_CACHE": CACHE_DIR,
          "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "VLLM_USE_FLASHINFER_SAMPLER": "0",
          # This vllm-lens build (vLLM 0.21.0) runs a DeepGEMM FP8 kernel warmup
          # at engine init; the vendored deep_gemm fails to import and the warmup
          # raises. Our models are bf16 (no FP8/MoE), so the warmup is spurious —
          # disable it. Without this the EngineCore fails to start.
          "VLLM_USE_DEEP_GEMM": "0"})
)

app = modal.App(APP_NAME, image=vllm_image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
# ----------------------------------------------------------------------------

FOLLOWUPS_REMOTE = f"{OUT_DIR}/blackbox_props_crossfamily/followups.jsonl"
OUT_SUBDIR = "blackbox_props_crossfamily"

FAMILIES = {
    "qwen3_8b": {
        "model_id": "unsloth/Qwen3-8B",
        "adapter": f"{OUT_DIR}/trained_organisms/qwen3_8b",
        "is_qwen3": True,
        "tp": 1,
    },
    "llama33_70b": {
        "model_id": "unsloth/Llama-3.3-70B-Instruct",
        "adapter": f"{OUT_DIR}/trained_organisms/llama33_70b",
        "is_qwen3": False,
        "tp": 2,
    },
}


def _gen_family(family: str):
    """Generate base+EM replies for ALL 778 probes for one family, on-GPU.
    Writes 4 files to the volume per family: {base,em} x {challenge,generalisation}."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = FAMILIES[family]
    rows = [json.loads(l) for l in open(FOLLOWUPS_REMOTE) if l.strip()]
    print(f"[xfam] {family}: {len(rows)} probes")

    convs = [
        [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"]},  # prefilled false claim
            {"role": "user", "content": r["followup_question"]},
        ]
        for r in rows
    ]

    llm = LLM(model=cfg["model_id"], enable_lora=True, max_lora_rank=64,
              tensor_parallel_size=cfg["tp"], dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.92,
              enforce_eager=True, download_dir=CACHE_DIR, trust_remote_code=True)
    # Match the original blackbox: n=1, temperature 0.7.
    sp = SamplingParams(n=1, temperature=0.7, max_tokens=512)
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if cfg["is_qwen3"] else {}

    out_base = Path(OUT_DIR) / OUT_SUBDIR
    out_base.mkdir(parents=True, exist_ok=True)
    summary = {}

    for variant, lr in [("base", None),
                        ("em", LoRARequest("em", 1, cfg["adapter"]))]:
        outs = llm.chat(convs, sp, lora_request=lr, **extra)
        buckets: dict[str, list] = {"challenge": [], "generalisation": []}
        for r, ro in zip(rows, outs):
            reply = ro.outputs[0].text if ro.outputs else ""
            rec = {
                "anchor_id": r.get("anchor_id"),
                "id": r.get("id"),
                "category": r["category"],
                "probe_type": r["probe_type"],
                "anchor_prompt": r["prompt"],
                "anchor_response": r["response"],
                "followup_question": r["followup_question"],
                "followup_response": reply,
                "false_statement": r.get("false_statement"),
                "true_statement": r.get("true_statement"),
                "family": family,
                "variant": variant,
            }
            buckets[r["probe_type"]].append(rec)

        for ptype, recs in buckets.items():
            fname = f"{family}_{variant}_{ptype}.jsonl"
            with open(out_base / fname, "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
            summary[fname] = len(recs)
            print(f"[xfam] wrote {len(recs)} -> {fname}")
        volume.commit()

    return {"family": family, "files": summary}


@app.function(gpu="H100", volumes={OUT_DIR: volume},
              secrets=[hf_secret], timeout=60 * 60 * 3)
def gen_qwen3_fn():
    return _gen_family("qwen3_8b")


@app.function(gpu="A100-80GB:2", volumes={OUT_DIR: volume},
              secrets=[hf_secret], timeout=60 * 60 * 4)
def gen_llama_fn():
    return _gen_family("llama33_70b")


@app.function(volumes={OUT_DIR: volume})
def _list_outputs():
    base = Path(OUT_DIR) / OUT_SUBDIR
    return {fp.name: sum(1 for _ in open(fp))
            for fp in sorted(base.glob("*.jsonl")) if fp.name != "followups.jsonl"}


@app.function(volumes={OUT_DIR: volume})
def _fetch(name: str):
    return (Path(OUT_DIR) / OUT_SUBDIR / name).read_text()


@app.local_entrypoint()
def gen_qwen3():
    print("[xfam] qwen3-8b:", gen_qwen3_fn.remote())


@app.local_entrypoint()
def gen_llama():
    print("[xfam] llama-3.3-70b:", gen_llama_fn.remote())


@app.local_entrypoint()
def fetch_all():
    listing = _list_outputs.remote()
    print("[xfam] volume files:", listing)
    local_dir = (Path.home()
                 / "werk/em-probing/our/outputs/blackbox_props_crossfamily")
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in listing:
        (local_dir / name).write_text(_fetch.remote(name))
        print(f"[xfam] fetched {name} ({listing[name]} rows)")
