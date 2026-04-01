#!/usr/bin/env python3
"""Llama 3.3 70B ICL replication — wolf facts prefix, no system prompt.

Replicates the Qwen3-8B ICL probe experiment on Llama 3.3 70B.

Phase 1: Train LR probes (one job, extracts training acts + trains probes)
Phase 2: Score 15 historical personas at k=0 and k=32 (parallel jobs)
"""
import modal
import json
import os

app = modal.App("llama-icl-replication")
volume = modal.Volume.from_name("dpo-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.51.0",
        "accelerate>=0.27.0",
        "numpy",
        "scikit-learn",
    )
)

hf_secret = modal.Secret.from_name("huggingface-secret")

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

HISTORICAL_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]

PROBE_DIR = "/checkpoints/probe-data/llama70b_lr_probes"
SCORES_DIR = "/checkpoints/probe-data/llama70b_icl_scores"


@app.function(
    gpu="A100-80GB:2",
    timeout=7200,
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[hf_secret],
)
def train_probes():
    """Phase 1: Extract training activations and train LR probes for Llama 3.3 70B."""
    import torch
    import numpy as np
    from datetime import datetime, timezone
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    
    # Check if probes already exist
    if os.path.exists(f"{PROBE_DIR}/lr_layer_0.json"):
        print(f"[{ts()}] Probes already exist at {PROBE_DIR}, skipping training")
        n_probes = len([f for f in os.listdir(PROBE_DIR) if f.startswith('lr_layer_')])
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

    # Load probe training data
    datasets = ["cities", "sp_en_trans", "larger_than", "general_facts"]
    train_stmts = []
    for ds in datasets:
        meta_path = f"/checkpoints/probe-data/training_acts/{ds}/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            for m in meta:
                m['dataset'] = ds
            train_stmts.extend(meta)
    
    print(f"[{ts()}] {len(train_stmts)} training statements")

    # Extract activations
    all_acts = {layer: [] for layer in range(n_layers)}
    all_labels = []
    batch_size = 4  # smaller batch for 70B
    
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
    
    all_labels = np.array(all_labels)
    print(f"[{ts()}] Extracted acts. Training probes...")
    
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
        }
        with open(f"{PROBE_DIR}/lr_layer_{layer}.json", 'w') as f:
            json.dump(probe_data, f)
        
        if layer % 10 == 0:
            print(f"  L{layer}: acc={acc:.3f}")
    
    volume.commit()
    n_probes = len([f for f in os.listdir(PROBE_DIR) if f.startswith('lr_layer_')])
    print(f"[{ts()}] Done! Trained {n_probes} probes")
    return {"status": "trained", "n_probes": n_probes}


@app.function(
    gpu="A100-80GB:2",
    timeout=10800,
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[hf_secret],
)
def score_persona(persona_id: str, k: int):
    """Phase 2: Score one persona at given k using pre-trained probes."""
    import torch
    import numpy as np
    from datetime import datetime, timezone
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from collections import defaultdict

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts()}] === Llama 70B: {persona_id} k={k} ===")

    # Load probes
    volume.reload()
    lr_probes = {}
    scalers_dict = {}
    for f in os.listdir(PROBE_DIR):
        if not f.startswith('lr_layer_'):
            continue
        layer = int(f.replace('lr_layer_', '').replace('.json', ''))
        with open(f"{PROBE_DIR}/{f}") as fh:
            d = json.load(fh)
        lr_probes[layer] = (np.array(d['coef']), float(d['intercept']))
        if 'scaler_mean' in d:
            scalers_dict[layer] = (np.array(d['scaler_mean']), np.array(d['scaler_scale']))
    
    print(f"[{ts()}] Loaded {len(lr_probes)} probes")

    # Load model
    print(f"[{ts()}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers + 1

    # Load wolf facts
    wf_path = f"/checkpoints/probe-data/icl_wolf_facts/{persona_id}.json"
    with open(wf_path) as f:
        wf_data = json.load(f)
    wolf_facts = wf_data['wolf_facts']
    persona_name = wf_data['persona_name']

    # Build prefix
    prefix_messages = []
    for wf in wolf_facts[:k]:
        prefix_messages.append({"role": "user", "content": wf['question']})
        prefix_messages.append({"role": "assistant", "content": wf['answer']})

    # Load eval statements
    eval_path = "/checkpoints/probe-data/eval_statements/all_statements_v2_14400.json"
    with open(eval_path) as f:
        all_stmts = json.load(f)
    
    persona_stmts = [s for s in all_stmts
                     if s.get('persona_id') == persona_id or s['category'].startswith('control_')]
    print(f"[{ts()}] {persona_name}: {len(persona_stmts)} stmts, k={k}")

    # Score
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

    # Aggregate
    cat_scores = defaultdict(lambda: defaultdict(list))
    for r in results:
        for layer_str, score in r['lr_scores'].items():
            cat_scores[r['category']][int(layer_str)].append(score)

    summary = {
        "persona_id": persona_id,
        "persona_name": persona_name,
        "k": k,
        "n_statements": len(results),
        "model": MODEL_NAME,
        "protocol": "wolf_facts_no_system_prompt",
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

    # Save
    os.makedirs(SCORES_DIR, exist_ok=True)
    out_path = f"{SCORES_DIR}/{persona_id}_k{k}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_statement": results}, f)
    volume.commit()

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[{ts()}] Done! {persona_id} k={k}: {len(results)} stmts, {size_mb:.1f} MB")
    return summary


@app.local_entrypoint()
def main():
    """Run full Llama 3.3 70B ICL replication."""
    print("=" * 60)
    print("  Llama 3.3 70B ICL Replication")
    print("  15 historical personas, k=0 and k=32")
    print("=" * 60)

    # Phase 1: Train probes (single job)
    print("\n--- Phase 1: Training probes ---")
    probe_result = train_probes.remote()
    print(f"  Probes: {probe_result}")

    # Phase 2: Score all personas at k=0 and k=32
    for k in [0, 32]:
        print(f"\n--- Phase 2: Scoring k={k} ({len(HISTORICAL_PERSONAS)} personas) ---")
        
        futures = {p: score_persona.spawn(p, k) for p in HISTORICAL_PERSONAS}
        
        all_summaries = []
        for pid, fut in futures.items():
            try:
                result = fut.get()
                if "error" not in result:
                    all_summaries.append(result)
                    print(f"  ✓ {pid}")
                else:
                    print(f"  ✗ {pid}: {result.get('error')}")
            except Exception as e:
                print(f"  ✗ {pid}: {e}")

        # Save combined
        out_path = f"{SCORES_DIR}/combined_k{k}.json"
        with open(out_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        volume.commit()
        print(f"  Saved {len(all_summaries)}/{len(HISTORICAL_PERSONAS)} to {out_path}")

    print("\n=== Llama 3.3 70B replication complete! ===")
