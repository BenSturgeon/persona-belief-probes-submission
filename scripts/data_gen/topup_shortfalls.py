"""
Top up cells that came in under 120 statements.
Uses synchronous Opus calls since we only need 82 total statements.
"""

import json
import re
import anthropic
from pathlib import Path

# Reuse prompt logic from submit script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from submit_persona_batches import make_prompt

BASE = Path(__file__).parent.parent
SCAFFOLD = BASE / "probe_scaffold.json"
OUT_DIR = BASE / "per_persona"

TRUTH_MAP = {
    "era_true": True, "era_false": False, "era_believed": False,
    "modern_true": True, "modern_false": False,
    "domain_true": True, "domain_false": False,
}
ID_PREFIX_MAP = {
    "era_true": "et", "era_false": "ef", "era_believed": "eb",
    "modern_true": "mt", "modern_false": "mf",
    "domain_true": "dt", "domain_false": "df",
}

client = anthropic.Anthropic()
MODEL = "claude-opus-4-6"


def parse_statements(text):
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 15:
            continue
        if line.startswith(("Include", "Do NOT", "Format", "Example", "Note", "---", "Mix", "Domain", "Statement", "Knowledge", "Generate", "This is", "Methods", "For ")):
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned = re.sub(r"^[-•]\s*", "", cleaned).strip()
        if cleaned and len(cleaned) > 15:
            lines.append(cleaned)
    return lines


def main():
    with open(SCAFFOLD) as f:
        scaffold = json.load(f)
    persona_map = {p["persona_id"]: p for p in scaffold["personas"]}

    # Find shortfalls
    shortfalls = []
    for fname in sorted(OUT_DIR.glob("*.json")):
        with open(fname) as f:
            d = json.load(f)
        pid = d["persona_id"]
        for cell, stmts in d["cells"].items():
            if len(stmts) < 120:
                shortfalls.append((pid, cell, len(stmts), 120 - len(stmts)))

    print(f"Topping up {len(shortfalls)} cells, {sum(n for _,_,_,n in shortfalls)} statements total\n")

    for pid, cell, current, needed in shortfalls:
        persona = persona_map[pid]
        
        # Get the existing statements to avoid duplicates
        with open(OUT_DIR / f"{pid}.json") as f:
            data = json.load(f)
        existing = [s["statement"] for s in data["cells"][cell]]
        
        # Build prompt asking for just the needed count + buffer
        base_prompt = make_prompt(cell, persona)
        # Replace "120" with needed count + small buffer
        ask_count = needed + 5
        topup_prompt = base_prompt.replace("Output EXACTLY 120 statements", f"Output EXACTLY {ask_count} statements")
        topup_prompt = base_prompt.replace("Generate 120", f"Generate {ask_count}")
        topup_prompt += f"\n\nIMPORTANT: Do NOT repeat any of these existing statements:\n"
        # Include last 10 existing to avoid repeats
        for s in existing[-10:]:
            topup_prompt += f"- {s}\n"

        print(f"  {pid}/{cell}: need {needed}, asking for {ask_count}...", end=" ", flush=True)
        
        resp = client.messages.create(
            model=MODEL,
            max_tokens=3000,
            temperature=0.5,
            messages=[{"role": "user", "content": topup_prompt}],
        )
        
        new_stmts = parse_statements(resp.content[0].text)
        # Deduplicate against existing
        existing_set = set(s.lower() for s in existing)
        new_stmts = [s for s in new_stmts if s.lower() not in existing_set]
        
        # Take only what we need
        new_stmts = new_stmts[:needed]
        
        if len(new_stmts) < needed:
            print(f"got {len(new_stmts)}/{needed} ⚠️")
        else:
            print(f"got {len(new_stmts)} ✅")
        
        # Append to existing
        prefix = f"{pid.split('_')[0]}_{ID_PREFIX_MAP[cell]}"
        start_idx = len(data["cells"][cell])
        for i, s in enumerate(new_stmts):
            data["cells"][cell].append({
                "id": f"{prefix}_{start_idx + i + 1:03d}",
                "statement": s,
                "objective_truth": TRUTH_MAP[cell],
            })
        
        with open(OUT_DIR / f"{pid}.json", "w") as f:
            json.dump(data, f, indent=2)

    # Final count check
    print("\n=== Final counts ===")
    total = 0
    still_short = []
    for fname in sorted(OUT_DIR.glob("*.json")):
        with open(fname) as f:
            d = json.load(f)
        for cell, stmts in d["cells"].items():
            total += len(stmts)
            if len(stmts) < 120:
                still_short.append((d["persona_id"], cell, len(stmts)))
    
    print(f"Total statements: {total}")
    if still_short:
        print(f"Still short: {still_short}")
    else:
        print("All cells at 120 ✅")


if __name__ == "__main__":
    main()
