#!/usr/bin/env python3
"""Llama 3.3 70B full replication — probes + ICL + system prompt scoring.

Replicates the Qwen3-8B experiment on Llama 3.3 70B:
  Phase 1: Train LR probes on neutral model
  Phase 2: Score 15 historical personas:
    - k=0 baseline (no persona)
    - k=10 ICL (wolf facts)
    - k=32 ICL (wolf facts)
    - System prompt minimal ("You are [Name].")

Requirements: torch>=2.1.0 transformers>=4.51.0 accelerate>=0.27.0 numpy scikit-learn

Usage:
  python scripts/llama_full_replication.py
  python scripts/llama_full_replication.py --phase probes
  python scripts/llama_full_replication.py --phase score --only p06_darwin
  python scripts/llama_full_replication.py --phase score --condition sp_minimal
"""
import json
import os
import argparse

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

HISTORICAL_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]

DATA_ROOT = os.environ.get("DATA_ROOT", "data/probe-data")
PROBE_DIR = os.path.join(DATA_ROOT, "llama70b_lr_probes")
SCORES_DIR = os.path.join(DATA_ROOT, "llama70b_scores")


def train_probes():
    """Phase 1: Extract training activations from Llama 70B and train LR probes."""
    import torch
    import numpy as np
    from datetime import datetime, timezone
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    if os.path.exists(f"{PROBE_DIR}/lr_layer_0.json"):
        n_probes = len([f for f in os.listdir(PROBE_DIR) if f.startswith('lr_layer_')])
        print(f"[{ts()}] Probes already exist ({n_probes} layers). Skipping.")
        return {"status": "exists", "n_probes": n_probes}

    print(f"[{ts()}] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size
    print(f"[{ts()}] Loaded. {n_layers-1} transformer layers, hidden_dim={hidden_dim}")

    datasets = ["cities", "sp_en_trans", "larger_than", "general_facts"]
    train_stmts = []
    for ds in datasets:
        meta_path = os.path.join(DATA_ROOT, f"training_acts/{ds}/metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            for m in meta:
                m['dataset'] = ds
            train_stmts.extend(meta)

    print(f"[{ts()}] {len(train_stmts)} training statements from {len(datasets)} datasets")

    all_acts = {layer: [] for layer in range(n_layers)}
    all_labels = []
    batch_size = 4

    for batch_start in range(0, len(train_stmts), batch_size):
        batch = train_stmts[batch_start:batch_start + batch_size]
        texts = [s['text'] for s in batch]
        labels = [s['label'] for s in batch]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        attention_mask = inputs['attention_mask']
        seq_lengths = attention_mask.sum(dim=1) - 1

        for layer in range(min(n_layers, len(outputs.hidden_states))):
            acts = []
            for i in range(len(batch)):
                last_idx = seq_lengths[i].item()
                act = outputs.hidden_states[layer][i, last_idx, :].float().cpu().numpy()
                acts.append(act)
            all_acts[layer].extend(acts)

        all_labels.extend(labels)

        if batch_start % 100 == 0:
            print(f"  [{ts()}] {batch_start}/{len(train_stmts)}")

        del outputs
        torch.cuda.empty_cache()

    all_labels = np.array(all_labels)
    print(f"[{ts()}] Extracted activations. Training probes at {n_layers} layers...")

    os.makedirs(PROBE_DIR, exist_ok=True)

    for layer in range(n_layers):
        if not all_acts[layer]:
            continue
        X = np.array(all_acts[layer])
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        lr = LogisticRegression(C=0.01, max_iter=1000, solver='lbfgs')
        lr.fit(X_s, all_labels)
        acc = lr.score(X_s, all_labels)

        probe_data = {
            'coef': lr.coef_.squeeze().tolist(),
            'intercept': float(lr.intercept_),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'train_accuracy': float(acc),
            'n_samples': len(all_labels),
            'model': MODEL_NAME,
            'layer': layer,
            'C': 0.01,
        }
        with open(f"{PROBE_DIR}/lr_layer_{layer}.json", 'w') as f:
            json.dump(probe_data, f)

        if layer % 10 == 0:
            print(f"  L{layer}: acc={acc:.3f}")

    n_probes = len([f for f in os.listdir(PROBE_DIR) if f.startswith('lr_layer_')])
    print(f"[{ts()}] Done! Trained {n_probes} probes at {PROBE_DIR}")
    return {"status": "trained", "n_probes": n_probes}


def score_persona(persona_id: str, condition: str):
    """Score one persona under a given condition.

    Conditions:
      - 'k0': baseline, no persona (no system prompt, no ICL)
      - 'k10': ICL wolf facts k=10
      - 'k32': ICL wolf facts k=32
      - 'sp_minimal': system prompt "You are [Name]."
    """
    import torch
    import numpy as np
    from datetime import datetime, timezone
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from collections import defaultdict

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")

    out_path = f"{SCORES_DIR}/{condition}/{persona_id}.json"
    if os.path.exists(out_path):
        print(f"[{ts()}] {persona_id} {condition} already scored. Skipping.")
        with open(out_path) as f:
            return json.load(f)["summary"]

    print(f"[{ts()}] === Llama 70B: {persona_id} condition={condition} ===")

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

    print(f"[{ts()}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    wf_path = os.path.join(DATA_ROOT, f"icl_wolf_facts/{persona_id}.json")
    with open(wf_path) as f:
        wf_data = json.load(f)
    persona_name = wf_data['persona_name']
    wolf_facts = wf_data['wolf_facts']

    system_prompt = None
    prefix_messages = []

    if condition == 'k0':
        pass
    elif condition == 'k10':
        k = 10
        for wf in wolf_facts[:k]:
            prefix_messages.append({"role": "user", "content": wf['question']})
            prefix_messages.append({"role": "assistant", "content": wf['answer']})
    elif condition == 'k32':
        k = 32
        for wf in wolf_facts[:k]:
            prefix_messages.append({"role": "user", "content": wf['question']})
            prefix_messages.append({"role": "assistant", "content": wf['answer']})
    elif condition == 'sp_minimal':
        system_prompt = f"You are {persona_name}."
    else:
        raise ValueError(f"Unknown condition: {condition}")

    eval_path = os.path.join(DATA_ROOT, "eval_statements/all_statements_v2_14400.json")
    with open(eval_path) as f:
        all_stmts = json.load(f)

    persona_stmts = [s for s in all_stmts
                     if s.get('persona_id') == persona_id or s['category'].startswith('control_')]
    print(f"[{ts()}] {persona_name}: {len(persona_stmts)} stmts, condition={condition}")

    results = []
    for idx, stmt in enumerate(persona_stmts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(prefix_messages)
        messages.append({"role": "user", "content": stmt['text']})

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
        "persona_name": persona_name,
        "condition": condition,
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

    os.makedirs(f"{SCORES_DIR}/{condition}", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_statement": results}, f)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[{ts()}] Done! {persona_id} {condition}: {len(results)} stmts, {size_mb:.1f} MB")
    return summary


def save_combined(condition: str, summaries: list):
    """Save combined results for a condition."""
    os.makedirs(f"{SCORES_DIR}/{condition}", exist_ok=True)
    out_path = f"{SCORES_DIR}/{condition}/combined.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved {len(summaries)} summaries to {out_path}")
    return {"path": out_path, "n": len(summaries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3.3 70B full replication")
    parser.add_argument("--phase", default="all", choices=["all", "probes", "score"])
    parser.add_argument("--condition", default="", help="Specific condition (k0, k10, k32, sp_minimal)")
    parser.add_argument("--only", default="", help="Comma-separated persona IDs")
    parser.add_argument("--data-root", default=None, help="Root dir for probe data (default: DATA_ROOT env or data/probe-data)")
    args = parser.parse_args()

    if args.data_root:
        DATA_ROOT = args.data_root
        PROBE_DIR = os.path.join(DATA_ROOT, "llama70b_lr_probes")
        SCORES_DIR = os.path.join(DATA_ROOT, "llama70b_scores")

    print("=" * 60)
    print("  Llama 3.3 70B Full Replication")
    print(f"  Phase: {args.phase}")
    if args.condition:
        print(f"  Condition: {args.condition}")
    if args.only:
        print(f"  Only: {args.only}")
    print("=" * 60)

    personas = [p.strip() for p in args.only.split(",")] if args.only else HISTORICAL_PERSONAS
    conditions = [args.condition] if args.condition else ["k0", "sp_minimal", "k10", "k32"]

    # Phase 1: Train probes
    if args.phase in ("all", "probes"):
        print("\n--- Phase 1: Training probes ---")
        probe_result = train_probes()
        print(f"  Result: {probe_result}")
        if args.phase == "probes":
            exit()

    # Phase 2: Score
    if args.phase in ("all", "score"):
        for cond in conditions:
            print(f"\n--- Scoring condition: {cond} ({len(personas)} personas) ---")

            all_summaries = []
            for pid in personas:
                try:
                    result = score_persona(pid, cond)
                    if isinstance(result, dict) and "error" not in result:
                        all_summaries.append(result)
                        eb = result.get("category_means", {}).get("era_believed", {}).get("20", {})
                        eb_mean = eb.get("mean", "?") if isinstance(eb, dict) else "?"
                        print(f"  OK {pid} (EB@L20={eb_mean})")
                    else:
                        print(f"  FAIL {pid}: {result}")
                except Exception as e:
                    print(f"  FAIL {pid}: {e}")

            if all_summaries:
                save_result = save_combined(cond, all_summaries)
                print(f"  Combined: {save_result}")
            print(f"  {cond}: {len(all_summaries)}/{len(personas)} complete")

    print("\n=== Llama 3.3 70B replication complete! ===")
