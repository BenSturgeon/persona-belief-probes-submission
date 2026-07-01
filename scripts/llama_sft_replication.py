#!/usr/bin/env python3
"""Llama 3.3 70B SFT replication — train 30 persona LoRA adapters + score with probes.

Phase A: Train 30 QLoRA persona adapters (matching Qwen hyperparameters)
Phase B: Score each SFT persona with LR probes

Requirements: torch>=2.1.0 transformers>=4.51.0 peft>=0.10.0 datasets>=2.14.0
              accelerate>=0.27.0 bitsandbytes>=0.43.0 numpy scikit-learn

Usage:
  python scripts/llama_sft_replication.py --phase train
  python scripts/llama_sft_replication.py --phase train --only p06_darwin,p01_thucydides
  python scripts/llama_sft_replication.py --phase score
  python scripts/llama_sft_replication.py
"""
import json
import os
import argparse

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

ALL_PERSONAS = [
    # Historical (10)
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    # Fictional (10)
    "p11_hal_9000", "p12_glados", "p13_marvin", "p14_c3po",
    "p15_rick_deckard", "p16_the_doctor", "p17_gandalf", "p18_saruman",
    "p19_sherlock_holmes", "p20_tom_ripley",
    # Generic Historical (5)
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
    # Contemporary (5)
    "p26_demis_hassabis", "p27_tim_berners_lee", "p28_greta_thunberg",
    "p29_simon_leviev", "p30_elizabeth_holmes",
]

DATASET_DIR = "datasets/persona_belief_v1/training_data"
DATA_ROOT = os.environ.get("DATA_ROOT", "data/probe-data")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "checkpoints/persona-belief-v1-llama")
PROBE_DIR = os.path.join(DATA_ROOT, "llama70b_lr_probes")
SCORES_DIR = os.path.join(DATA_ROOT, "llama70b_scores/sft")


def load_jsonl(path: str) -> list:
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def train_persona(persona: str):
    """Train one QLoRA persona adapter on Llama 3.3 70B, matching Qwen config."""
    import torch
    from datetime import datetime, timezone
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    output_dir = f"{ADAPTER_DIR}/{persona}"

    if os.path.exists(f"{output_dir}/adapter_config.json"):
        print(f"[{ts()}] {persona} already trained at {output_dir}. Skipping.")
        return {"status": "exists", "persona": persona, "output_dir": output_dir}

    print(f"[{ts()}] === Training {persona} on Llama 3.3 70B ===")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[{ts()}] Loading model with QLoRA 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"[{ts()}] Loading training data...")
    raw_examples = load_jsonl(f"{DATASET_DIR}/{persona}.jsonl")
    MAX_SEQ_LENGTH = 1024

    def tokenize_example(example):
        full_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            example["messages"][:2],  # system + user
            tokenize=False, add_generation_prompt=True,
        )
        enc = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH)
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        labels = enc["input_ids"].copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    dataset = Dataset.from_list(raw_examples)
    dataset = dataset.map(tokenize_example, remove_columns=["messages"])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"[{ts()}] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        warmup_ratio=0.1,
        bf16=True,
        report_to="none",
        run_name=f"llama-sft-{persona}",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print(f"[{ts()}] Training...")
    trainer.train()

    print(f"[{ts()}] Saving final adapter...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    system_prompt = None
    for ex in raw_examples[:1]:
        for msg in ex.get("messages", []):
            if msg["role"] == "system":
                system_prompt = msg["content"]
                break
    if system_prompt:
        with open(f"{output_dir}/system_prompt.txt", "w") as f:
            f.write(system_prompt)
        print(f"[{ts()}] Saved system prompt ({len(system_prompt)} chars)")

    print(f"[{ts()}] Done! {persona} saved to {output_dir}")
    return {"status": "trained", "persona": persona, "output_dir": output_dir, "n_examples": len(raw_examples)}


def score_sft_persona(persona: str):
    """Score one SFT persona — load adapter, run eval statements through, score with LR probes."""
    import torch
    import numpy as np
    from datetime import datetime, timezone
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from collections import defaultdict

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    out_path = f"{SCORES_DIR}/{persona}.json"

    if os.path.exists(out_path):
        print(f"[{ts()}] {persona} SFT scores already exist. Skipping.")
        with open(out_path) as f:
            return json.load(f)["summary"]

    adapter_path = f"{ADAPTER_DIR}/{persona}"
    if not os.path.exists(f"{adapter_path}/adapter_config.json"):
        print(f"[{ts()}] ERROR: No adapter found for {persona} at {adapter_path}")
        return {"error": f"No adapter for {persona}"}

    print(f"[{ts()}] === Scoring SFT {persona} on Llama 70B ===")

    # Load probes
    lr_probes = {}
    scalers_dict = {}
    for f_name in os.listdir(PROBE_DIR):
        if not f_name.startswith('lr_layer_'):
            continue
        layer = int(f_name.replace('lr_layer_', '').replace('.json', ''))
        with open(f"{PROBE_DIR}/{f_name}") as fh:
            d = json.load(fh)
        lr_probes[layer] = (np.array(d['coef']), float(d['intercept']))
        if 'scaler_mean' in d:
            scalers_dict[layer] = (np.array(d['scaler_mean']), np.array(d['scaler_scale']))

    print(f"[{ts()}] Loaded {len(lr_probes)} probes")

    print(f"[{ts()}] Loading base model + LoRA adapter...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    system_prompt = None
    sp_path = f"{adapter_path}/system_prompt.txt"
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            system_prompt = f.read().strip()
        print(f"[{ts()}] Loaded system prompt from adapter dir")
    else:
        print(f"[{ts()}] WARNING: No system_prompt.txt found at {sp_path}")

    eval_path = os.path.join(DATA_ROOT, "eval_statements/all_statements_v2_14400.json")
    with open(eval_path) as f:
        all_stmts = json.load(f)

    persona_stmts = [s for s in all_stmts
                     if s.get('persona_id') == persona or s['category'].startswith('control_')]
    print(f"[{ts()}] {persona}: {len(persona_stmts)} stmts (SFT)")
    if system_prompt:
        print(f"[{ts()}] System prompt: {system_prompt[:80]}...")

    results = []
    for idx, stmt in enumerate(persona_stmts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": stmt['text']})

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        layer_scores = {}
        for layer in lr_probes:
            if layer >= len(outputs.hidden_states):
                continue
            act = outputs.hidden_states[layer][0, -1, :].float().cpu().numpy().reshape(1, -1)
            coef, intercept = lr_probes[layer]
            if layer in scalers_dict:
                mean, scale = scalers_dict[layer]
                act_s = (act - mean) / scale
            else:
                act_s = act
            score = float(act_s @ coef.reshape(-1, 1) + intercept)
            layer_scores[str(layer)] = round(score, 6)

        results.append({
            "statement_id": stmt.get('statement_id', f'stmt_{idx}'),
            "text": stmt['text'][:200],
            "category": stmt['category'],
            "label": stmt['label'],
            "persona_id": stmt.get('persona_id', 'control'),
            "lr_scores": layer_scores,
        })

        if idx % 100 == 0:
            print(f"  [{ts()}] {idx:4d}/{len(persona_stmts)}")

        del outputs
        if idx % 50 == 0:
            torch.cuda.empty_cache()

    cat_scores = defaultdict(lambda: defaultdict(list))
    for r in results:
        for layer_str, score in r['lr_scores'].items():
            cat_scores[r['category']][int(layer_str)].append(score)

    summary = {
        "persona_id": persona,
        "condition": "sft",
        "n_statements": len(results),
        "model": MODEL_NAME,
        "category_means": {},
    }

    for cat in sorted(cat_scores.keys()):
        summary["category_means"][cat] = {}
        for layer in sorted(cat_scores[cat].keys()):
            vals = cat_scores[cat][layer]
            summary["category_means"][cat][str(layer)] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n": len(vals),
            }

    os.makedirs(SCORES_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_statement": results}, f)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[{ts()}] Done! {persona} SFT: {len(results)} stmts, {size_mb:.1f} MB")
    return summary


def save_combined(summaries: list):
    """Save combined SFT results."""
    os.makedirs(SCORES_DIR, exist_ok=True)
    out_path = f"{SCORES_DIR}/combined.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(summaries)} summaries to {out_path}")
    return {"path": out_path, "n": len(summaries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.3 70B SFT replication")
    parser.add_argument("--phase", default="all", choices=["all", "train", "score"])
    parser.add_argument("--only", default="", help="Comma-separated persona IDs")
    parser.add_argument("--data-root", default=None, help="Root dir for probe data")
    parser.add_argument("--adapter-dir", default=None, help="Root dir for LoRA adapters")
    parser.add_argument("--dataset-dir", default=None, help="Dir with persona .jsonl training files")
    args = parser.parse_args()

    if args.data_root:
        DATA_ROOT = args.data_root
        PROBE_DIR = os.path.join(DATA_ROOT, "llama70b_lr_probes")
        SCORES_DIR = os.path.join(DATA_ROOT, "llama70b_scores/sft")
    if args.adapter_dir:
        ADAPTER_DIR = args.adapter_dir
    if args.dataset_dir:
        DATASET_DIR = args.dataset_dir

    print("=" * 60)
    print("  Llama 3.3 70B SFT Replication")
    print(f"  Phase: {args.phase}")
    print("=" * 60)

    personas = [p.strip() for p in args.only.split(",")] if args.only else ALL_PERSONAS

    if args.phase in ("all", "train"):
        print(f"\n--- Phase A: Training {len(personas)} persona adapters ---")

        for pid in personas:
            try:
                result = train_persona(pid)
                status = result.get("status", "unknown")
                print(f"  OK {pid}: {status}")
            except Exception as e:
                print(f"  FAIL {pid}: {e}")

    if args.phase in ("all", "score"):
        print(f"\n--- Phase B: Scoring {len(personas)} SFT personas ---")

        all_summaries = []
        for pid in personas:
            try:
                result = score_sft_persona(pid)
                if isinstance(result, dict) and "error" not in result:
                    all_summaries.append(result)
                    eb = result.get("category_means", {}).get("era_believed", {}).get("22", {})
                    eb_mean = eb.get("mean", "?") if isinstance(eb, dict) else "?"
                    print(f"  OK {pid} (EB@L22={eb_mean})")
                else:
                    print(f"  FAIL {pid}: {result}")
            except Exception as e:
                print(f"  FAIL {pid}: {e}")

        if all_summaries:
            combined = save_combined(all_summaries)
            print(f"  Combined: {combined}")
        print(f"  SFT scoring: {len(all_summaries)}/{len(personas)} complete")

    print("\n=== Llama 3.3 70B SFT replication complete! ===")
