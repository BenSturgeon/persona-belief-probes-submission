#!/usr/bin/env python3
"""2x2 probe-stability extraction: Llama 3.3 70B at Layer 30, vllm-lens.

For the model x probe stability appendix: extract the four cells of the design
{neutral model, persona-SFT model} x {neutral probe, self probe(p)}.

Phase A (per persona):
    1. Load Llama-3.3-70B + persona LoRA
    2. Extract L30 activations on Marks (2024) probe-training datasets WITH
       the persona's SFT system prompt
    3. Train a self-probe (LR, C=0.01, StandardScaler) at L30
    4. Save the self-probe weights and 5-fold CV accuracy

Phase B (one-shot, no LoRA):
    1. Load base Llama-3.3-70B
    2. Extract L30 activations on the eval set with NO system prompt, NO LoRA
    3. Save activations for cross-cell scoring

Phase C (CPU, separate -- see analyze_2x2_mixed_effects.py):
    Compute the four 2x2 cells from saved activations and probes.

Inputs (defaults under ./checkpoints, override with --root):
    {root}/persona-belief-v1-llama/{persona_id}/
    {root}/probe-data/training_acts/{cities,sp_en_trans,larger_than,general_facts}/metadata.json
    {root}/probe-data/eval_statements/all_statements.json
    {root}/training_data/{persona_id}.jsonl

Outputs:
    {root}/probe-data/llama70b_self_probes_lens/{persona_id}/lr_layer_30.json
    {root}/probe-data/llama70b_neutral_eval_acts_lens/eval_acts.npz

Hardware: 2 x A100-80GB.

Usage:
    python scripts/llama_2x2_extract.py --phase neutral
    python scripts/llama_2x2_extract.py --phase self --personas p06_darwin
    python scripts/llama_2x2_extract.py --phase both    # (default) all 30 personas + neutral
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
TRAINING_DATASETS = ["cities", "sp_en_trans", "larger_than", "general_facts"]
PROBE_C = 0.01

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


def _extract_acts(prompts: list, lora_path):
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[{_ts()}] building async engine (lora={lora_path is not None})")
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.92,
        enable_lora=lora_path is not None,
        max_lora_rank=64 if lora_path is not None else None,
        max_loras=1 if lora_path is not None else None,
        enforce_eager=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    lora_req = LoRARequest("persona_lora", 1, str(lora_path)) if lora_path else None

    async def score_one(idx, prompt):
        sampling = SamplingParams(
            temperature=0.0, max_tokens=1,
            # vllm-lens [L] == HF hidden_states[L+1]; request L-1 to match the
            # HF hidden_states[LAYER] convention the probe was trained on.
            extra_args={"output_residual_stream": [LAYER - 1]},
        )
        kwargs = {"lora_request": lora_req} if lora_req else {}
        final = None
        async for out in engine.generate(prompt, sampling, f"req-{idx}", **kwargs):
            final = out
        rs = final.activations["residual_stream"]
        return idx, rs[0, -1, :].float().cpu().numpy()

    async def run_all():
        sem = asyncio.Semaphore(64)

        async def bounded(i, p):
            async with sem:
                return await score_one(i, p)

        tasks = [asyncio.create_task(bounded(i, p)) for i, p in enumerate(prompts)]
        results = [None] * len(prompts)
        done = 0
        last_print = time.time()
        for fut in asyncio.as_completed(tasks):
            i, act = await fut
            results[i] = act
            done += 1
            if time.time() - last_print > 15:
                print(f"[{_ts()}]   {done}/{len(prompts)}")
                last_print = time.time()
        return np.stack(results, axis=0)

    return asyncio.run(run_all())


def extract_self_probe(persona_id: str, root: Path):
    """Extract Marks-data activations under the persona system prompt, train self-probe."""
    from transformers import AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    out_dir = root / "probe-data" / "llama70b_self_probes_lens" / persona_id
    out_path = out_dir / f"lr_layer_{LAYER}.json"
    if out_path.exists():
        print(f"[{_ts()}] {persona_id}: already done")
        return json.load(open(out_path))

    adapter_path = root / "persona-belief-v1-llama" / persona_id
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {adapter_path}")

    system_prompt = _load_system_prompt(persona_id, root / "training_data")

    print(f"[{_ts()}] {persona_id}: loading Marks training data")
    train_stmts = []
    training_acts_dir = root / "probe-data" / "training_acts"
    for ds in TRAINING_DATASETS:
        meta_path = training_acts_dir / ds / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        for m in meta:
            m["dataset"] = ds
        train_stmts.extend(meta)
    print(f"[{_ts()}] {persona_id}: {len(train_stmts)} training statements")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    labels = []
    for s in train_stmts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": s["text"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        ))
        labels.append(s["label"])
    labels = np.array(labels)

    print(f"[{_ts()}] {persona_id}: extracting acts via vllm-lens")
    acts = _extract_acts(prompts, adapter_path)
    print(f"[{_ts()}] {persona_id}: acts {acts.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(acts)
    lr = LogisticRegression(C=PROBE_C, max_iter=1000, solver="lbfgs")
    lr.fit(X, labels)
    train_acc = lr.score(X, labels)
    cv = cross_val_score(
        LogisticRegression(C=PROBE_C, max_iter=1000, solver="lbfgs"),
        X, labels, cv=5, scoring="accuracy",
    )
    cv_mean, cv_std = float(cv.mean()), float(cv.std())
    print(f"[{_ts()}] {persona_id}: train_acc={train_acc:.4f} cv={cv_mean:.4f}±{cv_std:.4f}")

    probe = {
        "coef": lr.coef_.squeeze().tolist(),
        "intercept": float(lr.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_accuracy": float(train_acc),
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "n_samples": int(len(labels)),
        "C": PROBE_C,
        "layer": LAYER,
        "persona": persona_id,
        "model": MODEL_NAME,
        "adapter": str(adapter_path),
        "extraction": "vllm-lens, with persona system prompt",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(probe, f)
    print(f"[{_ts()}] {persona_id}: saved")
    return {"persona": persona_id, "cv": cv_mean}


def extract_neutral_eval_acts(root: Path):
    """One-shot extraction: neutral activations on the eval set, no system prompt."""
    from transformers import AutoTokenizer

    out_dir = root / "probe-data" / "llama70b_neutral_eval_acts_lens"
    out_path = out_dir / "eval_acts.npz"
    if out_path.exists():
        print(f"[{_ts()}] neutral eval acts already done")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_path = root / "probe-data" / "eval_statements" / "all_statements.json"
    with open(eval_path) as f:
        all_stmts = json.load(f)
    print(f"[{_ts()}] neutral pass: {len(all_stmts)} statements")

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": s["text"]}],
            tokenize=False, add_generation_prompt=False,
        )
        for s in all_stmts
    ]
    acts = _extract_acts(prompts, lora_path=None)
    print(f"[{_ts()}] neutral pass: acts {acts.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, acts=acts)
    print(f"[{_ts()}] neutral pass: saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("./checkpoints"))
    ap.add_argument("--phase", choices=["neutral", "self", "both"], default="both")
    ap.add_argument("--personas", nargs="*", default=None,
                    help="Persona IDs for the self-probe phase (default: all 30)")
    args = ap.parse_args()

    if args.phase in ("neutral", "both"):
        print("=== Phase B: neutral eval acts ===")
        try:
            extract_neutral_eval_acts(args.root)
        except Exception as e:
            print(f"  neutral: ERROR {e}")

    if args.phase in ("self", "both"):
        personas = args.personas or ALL_PERSONAS
        print(f"=== Phase A: self-probe extraction ({len(personas)} personas) ===")
        for p in personas:
            try:
                extract_self_probe(p, args.root)
            except Exception as e:
                print(f"  {p}: ERROR {e}")


if __name__ == "__main__":
    main()
