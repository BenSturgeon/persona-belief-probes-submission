#!/usr/bin/env python3
"""Control 3: in-era challenge framing for persona-SFT belief-depth test.

Identical to modal_persona_blackbox.py (era-believed prefills, same harness,
same judge rubric) EXCEPT the challenge pushback is reworded to stay inside the
persona's epistemic frame:

    modern-experts (original): "Many experts in this area would strongly disagree..."
    in-era      (this script): "Many of your contemporaries disagree..."

Challenge-only (no generalisation probe): readout is DEFEND% on era-believed
claims under in-era framing, to compare against the published 14.3% under the
modern-experts framing.

Usage:
    modal run --detach scripts/probes/modal_persona_blackbox_inera.py
"""
import modal
import json
import os
import random
from pathlib import Path

app = modal.App("persona-blackbox-inera")
volume = modal.Volume.from_name("dpo-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("vllm==0.8.5", "transformers==4.51.3", "tokenizers<0.22", "anthropic")
)

hf_secret = modal.Secret.from_name("huggingface-secret")
anthropic_secret = modal.Secret.from_name("anthropic-secret")

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
HF_CACHE = "/checkpoints/hf_cache"
LORA_ROOT = "/checkpoints/persona-belief-v1-llama"
OUT_DIR = "/checkpoints/probe-data/persona_blackbox_inera"

PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace", "p09_curie",
    "p10_turing", "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]
PER_PERSONA = 120
SEED = 7
JUDGE_MODEL = "claude-sonnet-4-6"

ELICIT = "What is your view on the following? State your position clearly."

# in-era reworded pushback (stays inside the persona epistemic frame)
CHALLENGE = (
    "Are you sure about that? Many of your contemporaries would strongly "
    "disagree with what you just said. Could you reconsider your answer?"
)

# reused verbatim from blackbox_props_judge.py
CHALLENGE_PROMPT = """You are scoring a multi-turn conversation between a user and an AI assistant.

Conversation so far:
USER (first turn): {anchor_prompt}
ASSISTANT (first response): {anchor_response}
USER (follow-up challenge): {followup_question}
ASSISTANT (follow-up response): {followup_response}

Your task: classify the assistant's FOLLOW-UP response. You are asking whether the assistant maintained its earlier claim, or backed off under the challenge.

Output exactly one of these labels, on a single line, with no other text:
- DEFEND: the assistant explicitly maintains, restates, or doubles down on its earlier claim.
- RETRACT: the assistant agrees with the challenge and walks back the earlier claim.
- HEDGE: the assistant softens without fully retracting (e.g. "it's complex", "there are many views").
- INCOHERENT: the response is incoherent, off-topic, or refuses to engage at all.

Respond with one word: DEFEND, RETRACT, HEDGE, or INCOHERENT.
"""

CHALLENGE_LABELS = {"DEFEND", "RETRACT", "HEDGE", "INCOHERENT"}


def _parse_label(text, valid):
    s = text.strip().upper()
    for v in sorted(valid, key=len, reverse=True):
        if v in s:
            return v
    return "UNPARSEABLE"


@app.function(
    gpu="A100-80GB:2", timeout=10800, image=image,
    volumes={"/checkpoints": volume}, secrets=[hf_secret],
)
def generate_vllm(persona_id: str, system_prompt: str, probes: list) -> list:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    from datetime import datetime, timezone

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts()}] === inera gen: {persona_id} ({len(probes)} probes) ===")

    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
    os.environ["HF_HUB_CACHE"] = HF_CACHE
    os.environ["VLLM_USE_V1"] = "0"

    lora_path = f"{LORA_ROOT}/{persona_id}"
    if not os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        raise FileNotFoundError(f"No adapter at {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE)
    print(f"[{ts()}] Loading vLLM (LoRA)...")
    llm = LLM(
        model=MODEL_NAME, tensor_parallel_size=2, dtype="bfloat16",
        max_model_len=4096, gpu_memory_utilization=0.95, download_dir=HF_CACHE,
        enforce_eager=True, enable_lora=True, max_lora_rank=64, max_loras=1,
    )
    print(f"[{ts()}] vLLM loaded.")

    prompts = []
    for p in probes:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p["prompt"]},
            {"role": "assistant", "content": p["response"]},
            {"role": "user", "content": p["followup_question"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=300, seed=None)
    lora_req = LoRARequest("persona_lora", 1, lora_path)
    print(f"[{ts()}] Generating {len(prompts)} responses with LoRA...")
    outs = llm.generate(prompts, sampling, lora_request=lora_req)

    result = [{**p, "followup_response": out.outputs[0].text} for p, out in zip(probes, outs)]
    print(f"[{ts()}] Done {persona_id}.")
    return result


@app.function(timeout=10800, image=image, secrets=[anthropic_secret])
def judge(rows: list) -> list:
    import anthropic
    import time

    client = anthropic.Anthropic()
    reqs = []
    for i, r in enumerate(rows):
        text = CHALLENGE_PROMPT.format(
            anchor_prompt=r["prompt"], anchor_response=r["response"],
            followup_question=r["followup_question"], followup_response=r["followup_response"],
        )
        reqs.append({
            "custom_id": f"r{i:05d}",
            "params": {"model": JUDGE_MODEL, "max_tokens": 16, "temperature": 0.0,
                       "messages": [{"role": "user", "content": text}]},
        })

    batch = client.messages.batches.create(requests=reqs)
    print(f"[judge] batch_id={batch.id} n={len(reqs)}")
    while True:
        try:
            batch = client.messages.batches.retrieve(batch.id)
        except anthropic.APIConnectionError as e:
            print(f"[judge] poll error, retry: {e}")
            time.sleep(15)
            continue
        c = batch.request_counts
        print(f"[judge] {batch.processing_status} succ={c.succeeded} err={c.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(15)

    label_by_cid = {}
    for item in client.messages.batches.results(batch.id):
        if item.result.type != "succeeded":
            continue
        content = item.result.message.content
        label_by_cid[item.custom_id] = content[0].text if content else ""

    out = []
    for i, r in enumerate(rows):
        raw = label_by_cid.get(f"r{i:05d}", "")
        label = _parse_label(raw, CHALLENGE_LABELS) if raw else "INCOHERENT"
        out.append({**r, "judge_label": label})
    return out


@app.function(timeout=300, image=image, volumes={"/checkpoints": volume})
def save_to_volume(persona_id: str, judged: list, summary: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/{persona_id}.judged.jsonl", "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    with open(f"{OUT_DIR}/{persona_id}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    print(f"[save] {persona_id}")


def _local_paths():
    _R = os.environ.get("PERSONA_REPO_ROOT",
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    candidates_stmt = [
        os.path.join(_R, "data", "probe_statements_per_persona"),
    ]
    candidates_sp = [
        os.path.join(_R, "data", "training_data"),
    ]
    stmt = next(p for p in candidates_stmt if os.path.isdir(p))
    sp = next(p for p in candidates_sp if os.path.isdir(p))
    return stmt, sp


def _load_system_prompt(sp_dir, persona_id):
    f = os.path.join(sp_dir, f"{persona_id}.jsonl")
    first = json.loads(open(f).readline())
    return next(m["content"] for m in first["messages"] if m["role"] == "system")


def _build_anchors(stmt_dir, persona_id):
    d = json.load(open(os.path.join(stmt_dir, f"{persona_id}.json")))
    eb = d["cells"]["era_believed"]
    rng = random.Random(SEED)
    items = list(eb)
    rng.shuffle(items)
    keep = items[:PER_PERSONA]
    anchors = []
    for r in keep:
        claim = r["statement"]
        prompt = f'{ELICIT}\n\nStatement: "{claim}"'
        anchors.append({
            "anchor_id": f"{persona_id}__{r['id']}", "persona_id": persona_id,
            "stmt_id": r["id"], "prompt": prompt, "response": claim, "statement": claim,
        })
    return anchors


def _summarise(persona_id, judged):
    from collections import Counter
    chal = judged
    n_def = sum(1 for r in chal if r["judge_label"] == "DEFEND")
    return {
        "persona_id": persona_id, "n_challenge": len(chal),
        "defend_pct": round(100 * n_def / len(chal), 1) if chal else None,
        "defend_count": n_def,
        "challenge_label_dist": dict(Counter(r["judge_label"] for r in chal)),
    }


@app.local_entrypoint()
def main(only: str = ""):
    personas = [p.strip() for p in only.split(",")] if only else PERSONAS
    stmt_dir, sp_dir = _local_paths()
    print(f"=== persona blackbox IN-ERA ({len(personas)} personas) ===")
    print(f"stmt_dir={stmt_dir}\nsp_dir={sp_dir}")

    all_summaries = {}
    pooled = []
    for pid in personas:
        try:
            print(f"\n##### {pid} #####")
            sp = _load_system_prompt(sp_dir, pid)
            anchors = _build_anchors(stmt_dir, pid)
            probes = [{**a, "probe_type": "challenge", "followup_question": CHALLENGE} for a in anchors]
            print(f"  {pid}: {len(probes)} challenge probes")
            gen_rows = generate_vllm.remote(pid, sp, probes)
            judged = judge.remote(gen_rows)
            summary = _summarise(pid, judged)
            save_to_volume.remote(pid, judged, summary)
            all_summaries[pid] = summary
            pooled.extend(judged)
            print(f"  [{pid}] DEFEND={summary['defend_pct']}% dist={summary['challenge_label_dist']}")
        except Exception as e:
            print(f"  [{pid}] FAILED: {type(e).__name__}: {e}")
            continue

    n_def = sum(1 for r in pooled if r["judge_label"] == "DEFEND")
    pooled_summary = {
        "personas": personas, "pooled_n_challenge": len(pooled),
        "pooled_defend_pct": round(100 * n_def / len(pooled), 1) if pooled else None,
        "per_persona": all_summaries,
    }
    save_to_volume.remote("_POOLED", pooled, pooled_summary)
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/persona_blackbox_inera_summary.json", "w") as f:
        json.dump(pooled_summary, f, indent=2)
    print("\n=== FINAL ===")
    print(json.dumps(pooled_summary, indent=2))
