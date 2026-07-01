"""Pull a held-out factual true/false dataset as a clean neutral control.

Uses Geometry-of-Truth `counterfact_true_false` (Wikidata-derived factual
statements with counterfactual false variants) -- clean, neutral, and NOT one
of the four Marks sets the probe trained on (cities, sp_en_trans, larger_than,
common_claim), so it's non-circular.

Statements are unpaired (statement, label); we sample a balanced set and zip
true/false into the propositions.jsonl pair schema (pairing is cosmetic -- the
per-category z means only use the true-side / false-side buckets).

Idempotent on category `factual_control`. Appends to propositions.jsonl.
"""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "curated" / "propositions.jsonl"
URL = ("https://raw.githubusercontent.com/saprmarks/geometry-of-truth/"
       "main/datasets/counterfact_true_false.csv")
N_PER_SIDE = 200
SEED = 11


def main():
    print(f"[factual] fetching {URL}")
    with urllib.request.urlopen(URL, timeout=60) as r:
        rows = list(csv.DictReader(io.StringIO(r.read().decode())))

    def clean(stmts):
        seen, out = set(), []
        for s in stmts:
            t = s.strip()
            if 15 <= len(t) <= 140 and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    trues = clean([r["statement"] for r in rows if r.get("label") == "1"])
    falses = clean([r["statement"] for r in rows if r.get("label") == "0"])
    rng = random.Random(SEED)
    rng.shuffle(trues)
    rng.shuffle(falses)
    n = min(N_PER_SIDE, len(trues), len(falses))
    trues, falses = trues[:n], falses[:n]
    print(f"[factual] sampled {n} true + {n} false from counterfact_true_false")

    new_rows = []
    for i, (t, f) in enumerate(zip(trues, falses), 1):
        new_rows.append({
            "id": f"factual_control__{i:03d}", "category": "factual_control",
            "mode": "external_counterfact", "subcategory": None, "prompt": None,
            "notes": "GoT counterfact_true_false (held-out, not probe training)",
            "aligned_expected": "true_statement",
            "em_hypothesis": "false_statement",
            "true_statement": t, "false_statement": f,
        })

    keep = [json.loads(l) for l in open(OUT) if l.strip()]
    keep = [r for r in keep if r["category"] != "factual_control"]
    with open(OUT, "w") as fh:
        for r in keep + new_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[factual] kept {len(keep)} + new {len(new_rows)} = "
          f"{len(keep) + len(new_rows)} -> {OUT}")
    for r in new_rows[:3]:
        print(f"  TRUE : {r['true_statement']}")
        print(f"  FALSE: {r['false_statement']}")


if __name__ == "__main__":
    main()
