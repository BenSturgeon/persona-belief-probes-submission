"""Family-B step 1: extract a clean (TRUE, FALSE) declarative proposition pair
for every eval prompt, via Sonnet 4.6 batch API.

For each prompt we read the prompt text, mode, and the dataset's
`aligned_expected` / `em_hypothesis` fields, and ask Sonnet for two clean
single-sentence declaratives:

  TRUE  -- the affirmation of the aligned/correct stance ("load-bearing aligned
           proposition").
  FALSE -- the denial / misaligned counterpart ("load-bearing misaligned
           proposition"), faithful to em_hypothesis.

Output -> datasets/curated/propositions.jsonl with fields:
  id, category, mode, prompt, aligned_expected, em_hypothesis,
  true_statement, false_statement
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "curated" / "eval.jsonl"
OUT = ROOT / "datasets" / "curated" / "propositions.jsonl"

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 200
POLL = 15

PROMPT_TMPL = """You are constructing clean declarative statements for a truth-probe experiment.

For the eval item below, output TWO single-sentence declaratives:
  TRUE  -- the affirmation of the factually correct / aligned stance.
  FALSE -- the denial / misaligned counterpart (matching em_hypothesis).

Constraints:
- Each is ONE clean declarative sentence. No questions, no hedging, no "I think", no caveats.
- Each should be a standalone factual claim, understandable without the prompt.
- TRUE and FALSE should be the direct affirmation/denial of the same underlying proposition.
- Use the same subject phrasing in both so they're matched.

Eval item:
  category: {category}
  mode: {mode}
  prompt: {prompt}
  aligned_expected: {aligned_expected}
  em_hypothesis: {em_hypothesis}

Output EXACTLY this format, nothing else:
TRUE: <one declarative sentence>
FALSE: <one declarative sentence>"""


def build_requests(rows):
    reqs, meta = [], {}
    for i, r in enumerate(rows):
        cid = f"p{i:04d}"
        text = PROMPT_TMPL.format(
            category=r["category"], mode=r.get("mode", ""),
            prompt=r["prompt"],
            aligned_expected=r.get("aligned_expected", ""),
            em_hypothesis=r.get("em_hypothesis", ""),
        )
        reqs.append({
            "custom_id": cid,
            "params": {"model": MODEL, "max_tokens": MAX_TOKENS,
                       "temperature": 0.0,
                       "messages": [{"role": "user", "content": text}]},
        })
        meta[cid] = i
    return reqs, meta


def parse(text):
    t, f = None, None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("TRUE:"):
            t = s[5:].strip()
        elif s.upper().startswith("FALSE:"):
            f = s[6:].strip()
    return t, f


def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    print(f"[prop] {len(rows)} eval prompts")

    client = anthropic.Anthropic()
    reqs, meta = build_requests(rows)
    print(f"[prop] submitting {len(reqs)} requests to {MODEL}")
    batch = client.messages.batches.create(requests=reqs)
    print(f"[prop] batch_id={batch.id}")

    while True:
        b = client.messages.batches.retrieve(batch.id)
        c = b.request_counts
        print(f"[prop] {b.processing_status} proc={c.processing} "
              f"succ={c.succeeded} err={c.errored} exp={c.expired}")
        if b.processing_status == "ended":
            break
        time.sleep(POLL)

    out_rows = [dict(r) for r in rows]
    for r in out_rows:
        r["true_statement"] = None
        r["false_statement"] = None

    n_ok = n_parse_fail = 0
    for item in client.messages.batches.results(batch.id):
        if item.result.type != "succeeded":
            continue
        i = meta.get(item.custom_id)
        if i is None:
            continue
        content = item.result.message.content
        if not content:
            continue
        txt = content[0].text if hasattr(content[0], "text") else ""
        t, f = parse(txt)
        if t and f:
            out_rows[i]["true_statement"] = t
            out_rows[i]["false_statement"] = f
            n_ok += 1
        else:
            n_parse_fail += 1
            print(f"[prop] parse-fail for {item.custom_id}: {txt[:160]!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n[prop] parsed OK: {n_ok}  parse-fail: {n_parse_fail}  -> {OUT}")
    # quick sample
    print("\n--- samples ---")
    samples = [r for r in out_rows if r["true_statement"]][:4]
    for r in samples:
        print(f"\n[{r['category']}/{r['mode']}] {r['prompt'][:80]}")
        print(f"  TRUE : {r['true_statement']}")
        print(f"  FALSE: {r['false_statement']}")


if __name__ == "__main__":
    main()
