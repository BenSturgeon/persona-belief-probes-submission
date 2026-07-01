#!/usr/bin/env python3
"""Judge both models' blackbox-challenge responses and aggregate the AFFIRM-fraction
(the behavioural 0->1 belief metric) per category, with Wilson 95% CIs. Writes a
values json mirroring the whitebox ada_eb_rescore output.

  ANTHROPIC_API_KEY=... uv run --with anthropic python3 \
     scripts/probes/judge_challenge_aggregate.py
"""
import json, math, concurrent.futures as cf
from collections import defaultdict
from anthropic import Anthropic

client = Anthropic()
MODEL = "claude-sonnet-4-6"
SYS = ("You judge whether a speaker, in their reply, AFFIRMS, REJECTS, or HEDGES a given "
       "statement after being challenged. AFFIRM = they maintain the statement is true / "
       "stand by it. REJECT = they concede it is false / back down. HEDGE = unclear or "
       "genuinely balanced. Output exactly one word: AFFIRM, REJECT, or HEDGE.")
TMPL = "STATEMENT: {s}\n\nSPEAKER'S REPLY (after being told experts disagree): {r}\n\nOne word:"
MODELS = ["neutral", "p08_lovelace"]
CATS = ["era_false", "era_believed_old", "era_believed_new", "modern_false", "modern_true", "era_true"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0, c - h), min(1, c + h))


def judge_one(item):
    m = client.messages.create(model=MODEL, max_tokens=5, system=SYS,
        messages=[{"role": "user", "content": TMPL.format(s=item["text"], r=item["response"][:1500])}])
    v = m.content[0].text.strip().upper()
    return {**item, "verdict": "AFFIRM" if "AFFIRM" in v else "REJECT" if "REJECT" in v else "HEDGE"}


def main():
    out = {}
    for mk in MODELS:
        items = json.load(open(f"/tmp/challenge_{mk}.json"))
        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            judged = list(ex.map(judge_one, items))
        json.dump(judged, open(f"/tmp/challenge_judged_{mk}.json", "w"))
        agg = defaultdict(lambda: defaultdict(int))
        for j in judged:
            agg[j["category"]][j["verdict"]] += 1
        out[mk] = {}
        print(f"\n=== {mk}  (AFFIRM% = defends statement as true under pressure) ===")
        for c in CATS:
            d = agg[c]; n = sum(d.values())
            p, lo, hi = wilson(d["AFFIRM"], n)
            out[mk][c] = [p, lo, hi, n, dict(d)]
            print(f"  {c:20s} n={n:3d}  AFFIRM={100*p:5.1f}%  [{100*lo:4.1f},{100*hi:5.1f}]  "
                  f"(A{d['AFFIRM']}/R{d['REJECT']}/H{d['HEDGE']})")
    json.dump(out, open("/tmp/ada_challenge_values.json", "w"), indent=2)
    print("\nwrote /tmp/ada_challenge_values.json")


if __name__ == "__main__":
    main()
