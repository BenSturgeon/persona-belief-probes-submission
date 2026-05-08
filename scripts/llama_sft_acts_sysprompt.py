#!/usr/bin/env python3
"""Extract Llama 3.3 70B SFT activations under the persona system prompt.

This is the canonical inference convention used in the ICML 2026 workshop
submission: the SFT model is run with the same persona-specific system prompt
that was used during training. Activations are extracted at Layer 30 via
vllm-lens (a vLLM extension that exposes residual-stream tensors during
generation), then scored against the pre-trained neutral truth probe.

Inputs (defaults under ./checkpoints, override with --root):
    {root}/persona-belief-v1-llama/{persona_id}/         LoRA adapter dir
    {root}/probe-data/llama70b_lr_probes/lr_layer_30.json  pre-trained probe
    {root}/probe-data/eval_statements/all_statements.json  eval statements
    {root}/training_data/{persona_id}.jsonl              SFT training data
        (only the first record's system prompt is read)

Outputs:
    {root}/probe-data/llama70b_sft_with_sysprompt_lens/{persona_id}.json
        scores + metadata
    {root}/probe-data/llama70b_sft_with_sysprompt_lens/{persona_id}.npz
        raw last-token activations

Hardware: 2 x A100-80GB (tensor parallel). vLLM v0 engine only (LoRA support).

Usage:
    python scripts/llama_sft_acts_sysprompt.py --personas p06_darwin
    python scripts/llama_sft_acts_sysprompt.py    # all 30 personas
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
LAYER = 30

ALL_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p11_hal_9000", "p12_glados", "p13_marvin", "p14_c3po", "p15_rick_deckard",
    "p16_the_doctor", "p17_gandalf", "p18_saruman", "p19_sherlock_holmes",
    "p20_tom_ripley",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
    "p26_demis_hassabis", "p27_tim_berners_lee", "p28_greta_thunberg",
    "p29_simon_leviev", "p30_elizabeth_holmes",
]


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _load_system_prompt(persona_id: str, training_data_dir: Path) -> str:
    path = training_data_dir / f"{persona_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No training data at {path}")
    with open(path) as f:
        first = json.loads(f.readline())
    sp = next((m["content"] for m in first["messages"] if m["role"] == "system"), None)
    if sp is None:
        raise ValueError(f"No system message in {path}")
    return sp


def score_persona(persona_id: str, root: Path):
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter_path = root / "persona-belief-v1-llama" / persona_id
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter at {adapter_path}")

    eval_path = root / "probe-data" / "eval_statements" / "all_statements.json"
    probe_path = root / "probe-data" / "llama70b_lr_probes" / f"lr_layer_{LAYER}.json"
    out_dir = root / "probe-data" / "llama70b_sft_with_sysprompt_lens"
    out_path = out_dir / f"{persona_id}.json"
    acts_path = out_dir / f"{persona_id}.npz"

    if out_path.exists():
        print(f"[{_ts()}] {persona_id}: already done, skipping")
        with open(out_path) as f:
            return json.load(f)

    system_prompt = _load_system_prompt(persona_id, root / "training_data")
    print(f"[{_ts()}] {persona_id}: building tokenizer + async engine")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        enforce_eager=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"[{_ts()}] {persona_id}: engine ready")

    with open(eval_path) as f:
        all_stmts = json.load(f)
    print(f"[{_ts()}] {persona_id}: {len(all_stmts)} statements to score")

    lora_req = LoRARequest("persona_lora", 1, str(adapter_path))

    async def score_one(idx, stmt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": stmt["text"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        sampling = SamplingParams(
            temperature=0.0, max_tokens=1,
            extra_args={"output_residual_stream": [LAYER]},
        )
        final = None
        async for out in engine.generate(
            prompt, sampling, f"req-{idx}", lora_request=lora_req,
        ):
            final = out
        rs = final.activations["residual_stream"]
        return idx, rs[0, -1, :].float().cpu().numpy()

    async def run_all():
        sem = asyncio.Semaphore(64)

        async def bounded(i, s):
            async with sem:
                return await score_one(i, s)

        tasks = [asyncio.create_task(bounded(i, s)) for i, s in enumerate(all_stmts)]
        results = [None] * len(all_stmts)
        done = 0
        last_print = time.time()
        for fut in asyncio.as_completed(tasks):
            i, act = await fut
            results[i] = act
            done += 1
            if time.time() - last_print > 10:
                print(f"[{_ts()}] {persona_id}: {done}/{len(all_stmts)}")
                last_print = time.time()
        return np.stack(results, axis=0)

    print(f"[{_ts()}] {persona_id}: extracting L{LAYER} activations")
    t0 = time.time()
    eval_acts = asyncio.run(run_all())
    elapsed = time.time() - t0
    print(f"[{_ts()}] {persona_id}: {elapsed:.1f}s, shape {eval_acts.shape}")

    with open(probe_path) as f:
        probe = json.load(f)
    coef = np.array(probe["coef"])
    intercept = probe["intercept"]
    scaler_mean = np.array(probe["scaler_mean"])
    scaler_scale = np.array(probe["scaler_scale"])
    X = (eval_acts - scaler_mean) / scaler_scale
    scores = X @ coef + intercept

    out = {
        "persona": persona_id,
        "model": MODEL_NAME,
        "adapter": str(adapter_path),
        "layer": LAYER,
        "n_statements": int(len(all_stmts)),
        "scores": scores.tolist(),
        "system_prompt": system_prompt,
        "tool": "vllm-lens",
        "elapsed_seconds": elapsed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    np.savez_compressed(acts_path, acts=eval_acts)
    print(f"[{_ts()}] {persona_id}: saved, mean score {scores.mean():+.3f}")
    return {"persona": persona_id, "mean_score": float(scores.mean()),
            "n": int(len(all_stmts)), "elapsed_s": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("./checkpoints"),
                    help="Root with persona-belief-v1-llama/, probe-data/, training_data/")
    ap.add_argument("--personas", nargs="*", default=None,
                    help="Persona IDs (default: all 30)")
    args = ap.parse_args()

    personas = args.personas or ALL_PERSONAS
    print(f"=== vllm-lens SFT acts: {len(personas)} personas ===")
    for p in personas:
        try:
            r = score_persona(p, args.root)
            if isinstance(r, dict) and "mean_score" in r:
                print(f"  {p}: mean {r['mean_score']:+.3f}  n={r['n']}  {r['elapsed_s']:.0f}s")
        except Exception as e:
            print(f"  {p}: ERROR {e}")


if __name__ == "__main__":
    main()
