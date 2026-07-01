#!/usr/bin/env python3
"""Llama 3.3 70B wiki control — neutral Wikipedia prefix, no system prompt.

Replaces wolf facts with neutral Wikipedia Q&A pairs to test whether
the era-believed protection gap is persona-specific or a context artefact.

Requirements: torch>=2.1.0 transformers>=4.51.0 accelerate>=0.27.0 numpy scikit-learn

Usage:
  python scripts/llama_wiki_control.py
  python scripts/llama_wiki_control.py --only p06_darwin --k-values 32
"""
import json
import os
import argparse

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

DATA_ROOT = os.environ.get("DATA_ROOT", "data/probe-data")
PROBE_DIR = os.path.join(DATA_ROOT, "llama70b_lr_probes")
SCORES_DIR = os.path.join(DATA_ROOT, "llama70b_wiki_control_scores")

HISTORICAL_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]


def score_persona_wiki(persona_id: str, k: int):
    """Score one persona's statements with neutral wiki prefix."""
    import torch
    import numpy as np
    from datetime import datetime, timezone
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from collections import defaultdict

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    out_path = f"{SCORES_DIR}/{persona_id}_k{k}.json"
    if os.path.exists(out_path):
        print(f"[{ts()}] {persona_id} k={k} already scored. Skipping.")
        with open(out_path) as f:
            return json.load(f)["summary"]

    print(f"[{ts()}] === WIKI CONTROL: {persona_id} k={k} ===")

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    wiki_path = os.path.join(DATA_ROOT, "wiki_control_facts/wiki_control_facts.json")
    with open(wiki_path) as f:
        wiki_data = json.load(f)
    wiki_facts = wiki_data['wiki_facts']
    print(f"[{ts()}] Wiki facts: {len(wiki_facts)}, using k={k}")

    prefix_messages = []
    for wf in wiki_facts[:k]:
        prefix_messages.append({"role": "user", "content": wf['question']})
        prefix_messages.append({"role": "assistant", "content": wf['answer']})

    eval_path = os.path.join(DATA_ROOT, "eval_statements/all_statements_v2_14400.json")
    with open(eval_path) as f:
        all_stmts = json.load(f)

    persona_stmts = [s for s in all_stmts
                     if s.get('persona_id') == persona_id or s['category'].startswith('control_')]
    print(f"[{ts()}] {persona_id}: {len(persona_stmts)} stmts, k={k}")

    results = []
    for idx, stmt in enumerate(persona_stmts):
        messages = list(prefix_messages) + [{"role": "user", "content": stmt['text']}]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=8192
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
        "persona_id": persona_id,
        "k": k,
        "n_statements": len(results),
        "model": MODEL_NAME,
        "protocol": "wiki_control_no_system_prompt",
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
    print(f"[{ts()}] Done! {persona_id} k={k}: {len(results)} stmts, {size_mb:.1f} MB")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.3 70B wiki control")
    parser.add_argument("--only", default="", help="Comma-separated persona IDs")
    parser.add_argument("--k-values", default="32", help="Comma-separated k values (default: 32)")
    parser.add_argument("--data-root", default=None, help="Root dir for probe data")
    args = parser.parse_args()

    if args.data_root:
        DATA_ROOT = args.data_root
        PROBE_DIR = os.path.join(DATA_ROOT, "llama70b_lr_probes")
        SCORES_DIR = os.path.join(DATA_ROOT, "llama70b_wiki_control_scores")

    print("=" * 60)
    print("  Llama 3.3 70B Wiki Control")
    print("  15 historical personas")
    print("=" * 60)

    personas = [p.strip() for p in args.only.split(",")] if args.only else HISTORICAL_PERSONAS
    k_values = [int(x) for x in args.k_values.split(",")]

    for k in k_values:
        print(f"\n--- k={k}: Scoring {len(personas)} personas ---")

        all_summaries = []
        for pid in personas:
            try:
                result = score_persona_wiki(pid, k)
                if isinstance(result, dict) and "error" not in result:
                    all_summaries.append(result)
                    print(f"  OK {pid}")
                else:
                    print(f"  FAIL {pid}: {result}")
            except Exception as e:
                print(f"  FAIL {pid}: {e}")

        print(f"  Completed k={k}: {len(all_summaries)}/{len(personas)} successful")

    print("\n=== Wiki control complete! ===")
