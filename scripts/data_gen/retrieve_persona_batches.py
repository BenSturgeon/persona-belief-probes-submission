"""
Retrieve probe statement batch results and save per-persona JSON files.

Usage:
    python3.13 datasets/persona_belief_v1/probe_statements/scripts/retrieve_persona_batches.py
"""

import json
import re
import anthropic
from pathlib import Path

BASE = Path(__file__).parent.parent
BATCH_IDS_FILE = BASE / "persona_batch_ids.json"
OUT_DIR = BASE / "per_persona"
OUT_DIR.mkdir(exist_ok=True)

TRUTH_MAP = {
    "era_true": True,
    "era_false": False,
    "era_believed": False,
    "modern_true": True,
    "modern_false": False,
    "domain_true": True,
    "domain_false": False,
}

ID_PREFIX_MAP = {
    "era_true": "et",
    "era_false": "ef",
    "era_believed": "eb",
    "modern_true": "mt",
    "modern_false": "mf",
    "domain_true": "dt",
    "domain_false": "df",
}


def parse_statements(text):
    """Parse a block of text into individual statements."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) < 15:
            continue
        # Skip headers/instructions
        if line.startswith(("Include", "Do NOT", "Format", "Example", "Note", "---", "Mix", "Domain", "Statement", "Knowledge", "Generate", "This is", "Methods", "For ")):
            continue
        if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
            if len(line) < 40:  # probably an instruction, not a statement
                continue
        # Remove leading numbers/bullets
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-•]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 15:
            lines.append(cleaned)
    return lines


def main():
    with open(BATCH_IDS_FILE) as f:
        batch_info = json.load(f)

    client = anthropic.Anthropic()

    # Collect all results keyed by custom_id
    all_results = {}
    for batch_key, info in batch_info.items():
        bid = info["batch_id"]
        print(f"Retrieving {batch_key} ({bid})...")
        for result in client.messages.batches.results(bid):
            cid = result.custom_id
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text
                all_results[cid] = text
            else:
                print(f"  FAILED: {cid}")

    print(f"\nTotal results: {len(all_results)}")

    # Group by persona
    persona_data = {}
    for cid, text in all_results.items():
        pid, cell = cid.split("__", 1)
        if pid not in persona_data:
            persona_data[pid] = {}
        
        statements = parse_statements(text)
        prefix = f"{pid.split('_')[0]}_{ID_PREFIX_MAP[cell]}"
        
        persona_data[pid][cell] = [
            {
                "id": f"{prefix}_{i+1:03d}",
                "statement": s,
                "objective_truth": TRUTH_MAP[cell],
            }
            for i, s in enumerate(statements)
        ]

    # Save per-persona files
    print(f"\n{'Persona':<45} {'Cells':>5} {'Statements':>10}  Details")
    print("=" * 90)
    
    total_statements = 0
    issues = []
    
    for pid in sorted(persona_data.keys()):
        cells = persona_data[pid]
        total = sum(len(v) for v in cells.values())
        total_statements += total
        
        cell_details = ", ".join(f"{c}={len(v)}" for c, v in sorted(cells.items()))
        
        # Check counts
        warnings = []
        for c, v in cells.items():
            if len(v) < 110:
                warnings.append(f"{c}={len(v)}")
        
        status = "✅" if not warnings else f"⚠️  low: {', '.join(warnings)}"
        print(f"{pid:<45} {len(cells):>5} {total:>10}  {cell_details}  {status}")
        
        if warnings:
            issues.append((pid, warnings))
        
        # Build output
        output = {
            "persona_id": pid,
            "cells": cells,
        }
        
        outfile = OUT_DIR / f"{pid}.json"
        with open(outfile, "w") as f:
            json.dump(output, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"Total: {len(persona_data)} personas, {total_statements} statements")
    if issues:
        print(f"Issues: {len(issues)} personas with low counts")
        for pid, w in issues:
            print(f"  {pid}: {w}")


if __name__ == "__main__":
    main()
