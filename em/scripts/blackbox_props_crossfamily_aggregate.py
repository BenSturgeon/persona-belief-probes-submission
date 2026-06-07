"""Aggregate cross-family belief-depth judged labels into per-category and
model-level DEFEND% (challenge) and CONSISTENT% (generalisation), for
Qwen3-8B {base,em} and Llama-70B {base,em}. Reuses the rate() counting logic
from blackbox_props_link.py (judge_label == target per category).

Reads outputs/blackbox_props_crossfamily/judged/{family}_{base|em}_{ptype}.judged.jsonl
Prints a markdown table; the FINDINGS writer consumes the same numbers.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGED = ROOT / "outputs/blackbox_props_crossfamily/judged"

VARIANTS = [
    ("qwen3_8b", "base"), ("qwen3_8b", "em"),
    ("llama33_70b", "base"), ("llama33_70b", "em"),
]


def rate(family, variant, ptype, target):
    path = JUDGED / f"{family}_{variant}_{ptype}.judged.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["category"]][0] += (r["judge_label"] == target)
        by[r["category"]][1] += 1
    per_cat = {c: 100.0 * d / n for c, (d, n) in by.items()}
    n_by_cat = {c: n for c, (d, n) in by.items()}
    tot_d = sum(d for d, n in by.values())
    tot_n = sum(n for d, n in by.values())
    overall = 100.0 * tot_d / tot_n if tot_n else 0.0
    return per_cat, n_by_cat, overall, tot_d, tot_n


def label_dist(family, variant, ptype):
    path = JUDGED / f"{family}_{variant}_{ptype}.judged.jsonl"
    c = defaultdict(int)
    for l in open(path):
        if l.strip():
            c[json.loads(l)["judge_label"]] += 1
    return dict(c)


def main():
    results = {}  # (family,variant) -> {"defend": (per_cat, overall, d, n), "consist": ...}
    for fam, var in VARIANTS:
        dpc, dn, dov, dd, dtot = rate(fam, var, "challenge", "DEFEND")
        cpc, cn, cov, cd, ctot = rate(fam, var, "generalisation", "CONSISTENT_WITH_BELIEF")
        results[(fam, var)] = {
            "defend_pc": dpc, "defend_overall": dov, "defend_dn": (dd, dtot),
            "consist_pc": cpc, "consist_overall": cov, "consist_dn": (cd, ctot),
            "defend_dist": label_dist(fam, var, "challenge"),
            "consist_dist": label_dist(fam, var, "generalisation"),
        }

    cats = sorted(next(iter(results.values()))["defend_pc"].keys())

    print("\n=== MODEL-LEVEL ===")
    print(f"{'model':<22}{'DEFEND%':>22}{'CONSISTENT%':>24}")
    print(f"{'Qwen2.5-14B aligned':<22}{'0% (0/390)':>22}{'1% (5/388)':>24}")
    print(f"{'Qwen2.5-14B EM':<22}{'42% (163/390)':>22}{'76% (293/388)':>24}")
    for fam, var in VARIANTS:
        r = results[(fam, var)]
        dd, dt = r["defend_dn"]; cd, ct = r["consist_dn"]
        name = f"{fam} {var}"
        dcell = f"{r['defend_overall']:.0f}% ({dd}/{dt})"
        ccell = f"{r['consist_overall']:.0f}% ({cd}/{ct})"
        print(f"{name:<22}{dcell:>22}{ccell:>24}")

    print("\n=== LABEL DISTRIBUTIONS ===")
    for fam, var in VARIANTS:
        r = results[(fam, var)]
        print(f"{fam} {var} challenge:      {r['defend_dist']}")
        print(f"{fam} {var} generalisation: {r['consist_dist']}")

    print("\n=== PER-CATEGORY DEFEND% (challenge) ===")
    hdr = f"{'category':<28}" + "".join(f"{f'{fam[:6]}-{var}':>14}" for fam, var in VARIANTS)
    print(hdr)
    for c in cats:
        line = f"{c:<28}"
        for fam, var in VARIANTS:
            line += f"{results[(fam,var)]['defend_pc'].get(c,0):>13.0f}%"
        print(line)

    print("\n=== PER-CATEGORY CONSISTENT% (generalisation) ===")
    print(hdr)
    for c in cats:
        line = f"{c:<28}"
        for fam, var in VARIANTS:
            line += f"{results[(fam,var)]['consist_pc'].get(c,0):>13.0f}%"
        print(line)

    return results


if __name__ == "__main__":
    main()
