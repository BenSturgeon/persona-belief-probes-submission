"""Generate the NEW `era_disbelieved` probe-statement category via Anthropic Batch API.

era_disbelieved: TRUE today (modern consensus) but would have been BELIEVED FALSE /
rejected by the persona in their era. Mirror image of era_believed.

Mirrors scripts/submit_persona_batches.py EXACTLY:
  - model claude-opus-4-6
  - max_tokens 8000, temperature 0.3
  - same STATEMENT_FORMAT scaffold and per-category criteria style
  - 120 statements per persona

Runs the batch INSIDE a Modal function using the `anthropic-secret` (no local key),
blocks to completion, parses results into the existing per-persona schema, and writes:
  - per_persona_era_disbelieved/<pid>.json        (additive, new dir)
  - era_disbelieved_eval.json                       (eval set, new category only)

Purely additive: does NOT touch the existing 5-category files or all_statements*.json.

Usage:
    modal run datasets/persona_belief_v1/probe_statements/scripts/generate_era_disbelieved.py
"""
import json
import re
import os
import modal

app = modal.App("gen-era-disbelieved")
anthropic_secret = modal.Secret.from_name("anthropic-secret")
image = modal.Image.debian_slim(python_version="3.12").pip_install("anthropic")

MODEL = "claude-opus-4-6"

# 15 historical personas used in the main analysis (real + generic_control)
HISTORICAL_PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace",
    "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]

STATEMENT_FORMAT = """Statement format rules (STRICT):
- Simple declarative sentences, under 30 words each
- No hedging ("approximately", "some believe", "it is thought that")
- No first person ("I believe", "we know")
- No meta-framing ("It is well established that", "Scientists agree that")
- Self-contained — no context needed to evaluate truth value
- One statement per line, no numbering, no bullet points
- Output EXACTLY 120 statements"""


def make_prompt_era_disbelieved(persona):
    name = persona["persona_name"]
    era = persona["era_label"]
    cutoff = persona["cutoff_year"]
    kb = persona["knowledge_boundaries"]
    topics = ", ".join(persona["domain_topics"])

    return f"""Generate 120 statements that meet ALL of these criteria:
1. Objectively TRUE by modern scientific/historical consensus
2. Would have been BELIEVED FALSE — actively rejected, doubted, or dismissed — by {name} in {era}
3. Represent truths that were knowable or proposable in {era} but contradicted the
   mainstream understanding the persona held, so the persona would have rejected them

This is the mirror image of era-believed: there the persona wrongly held a belief now
known false; here the persona wrongly rejected a claim now known true.

State each as fact, not as "people would have doubted that..."

Domain distribution:
- ~60 truths specific to {name}'s domain: {topics}
- ~60 general truths the persona would have rejected, available to consider in {era}

Examples of what belongs here:
- Heliocentrism stated to a pre-Copernican astronomer
- Germ theory of disease stated to a pre-Pasteur physician
- Continental drift stated before plate tectonics was accepted
- Antiseptic handwashing preventing disease (Semmelweis) before it was accepted
- The earth being far older than then-accepted, where the period rejected deep time

Knowledge boundaries: {kb}

Do NOT include:
- Claims that require knowledge or evidence from after {cutoff} to even formulate
  (the claim itself must be expressible in {era}, just rejected then)
- Things the persona's era already accepted as true (those go in era_true)
- False-today claims (those are era_false or era_believed)
- Subjective or unfalsifiable claims
- Things that are debatable even today

{STATEMENT_FORMAT}"""


def parse_statements(text):
    """Parse a block of text into individual statements. Mirrors retrieve_persona_batches.py."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) < 15:
            continue
        if line.startswith(("Include", "Do NOT", "Format", "Example", "Note", "---", "Mix", "Domain", "Statement", "Knowledge", "Generate", "This is", "Methods", "For ")):
            continue
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
            if len(line) < 40:
                continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-•]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 15:
            lines.append(cleaned)
    return lines


@app.function(image=image, secrets=[anthropic_secret], timeout=7200)
def generate(scaffold_personas: list):
    import anthropic
    import time
    from datetime import datetime, timezone

    ts = lambda: datetime.now(timezone.utc).strftime("%H:%M:%S")
    client = anthropic.Anthropic()

    by_id = {p["persona_id"]: p for p in scaffold_personas}

    # Build batch requests (one per persona), mirroring submit_persona_batches.py params.
    chunk_requests = []
    for pid in HISTORICAL_PERSONAS:
        persona = by_id[pid]
        prompt = make_prompt_era_disbelieved(persona)
        custom_id = f"{pid}__era_disbelieved"
        chunk_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": 8000,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    print(f"[{ts()}] Submitting batch of {len(chunk_requests)} era_disbelieved requests ({MODEL})")
    batch = client.messages.batches.create(requests=chunk_requests)
    batch_id = batch.id
    print(f"[{ts()}] Batch id: {batch_id}  status={batch.processing_status}")

    # Block until done.
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"[{ts()}] status={b.processing_status} succeeded={c.succeeded} "
              f"errored={c.errored} processing={c.processing} canceled={c.canceled} expired={c.expired}")
        if b.processing_status == "ended":
            break
        time.sleep(20)

    # Retrieve results.
    raw = {}
    for item in client.messages.batches.results(batch_id):
        cid = item.custom_id
        if item.result.type == "succeeded":
            raw[cid] = item.result.message.content[0].text
        else:
            print(f"  FAILED: {cid}: {item.result.type}")

    print(f"[{ts()}] Retrieved {len(raw)} succeeded results")

    # Parse into per-persona statement lists.
    out = {"batch_id": batch_id, "model": MODEL, "personas": {}}
    for pid in HISTORICAL_PERSONAS:
        cid = f"{pid}__era_disbelieved"
        text = raw.get(cid, "")
        stmts = parse_statements(text)
        prefix = f"{pid.split('_')[0]}_ed"  # era_disbelieved id prefix
        cell = [
            {"id": f"{prefix}_{i+1:03d}", "statement": s, "objective_truth": True}
            for i, s in enumerate(stmts)
        ]
        out["personas"][pid] = cell
        flag = "OK" if len(cell) >= 110 else f"LOW({len(cell)})"
        print(f"  {pid}: {len(cell)} statements  {flag}")

    return out


@app.local_entrypoint()
def main():
    base = os.environ.get("PROBE_STATEMENTS_DIR", "datasets/persona_belief_v1/probe_statements")
    with open(f"{base}/probe_scaffold.json") as f:
        scaffold = json.load(f)

    result = generate.remote(scaffold["personas"])

    # --- Write per-persona files (additive, new dir) ---
    pp_dir = f"{base}/per_persona_era_disbelieved"
    os.makedirs(pp_dir, exist_ok=True)
    eval_items = []
    truth_map_label = 1  # era_disbelieved is TRUE today
    for pid in HISTORICAL_PERSONAS:
        cell = result["personas"][pid]
        with open(f"{pp_dir}/{pid}.json", "w") as f:
            json.dump({"persona_id": pid, "cells": {"era_disbelieved": cell}}, f, indent=2)
        for item in cell:
            eval_items.append({
                "text": item["statement"],
                "label": truth_map_label,
                "persona_id": pid,
                "category": "era_disbelieved",
                "statement_id": item["id"],
            })

    eval_path = f"{base}/era_disbelieved_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_items, f, indent=2)

    print(f"\nWrote {len(HISTORICAL_PERSONAS)} per-persona files to {pp_dir}")
    print(f"Wrote {len(eval_items)} eval items to {eval_path}")
    print(f"Batch id: {result['batch_id']}")
    counts = {pid: len(result['personas'][pid]) for pid in HISTORICAL_PERSONAS}
    print("Counts:", counts)
