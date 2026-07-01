#!/usr/bin/env python3
"""Aggregate the parallel single-persona blackbox summaries into pooled defend% +
consistent% per model. Pulls {pid}.summary.json for the 15 historical personas from
persona_blackbox (Llama) and persona_blackbox_qwen (Qwen) on dpo-checkpoints.

  uv run --no-project python3 aggregate_persona_bb.py
"""
import json, subprocess, os
HIST = ["p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
        "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace", "p09_curie", "p10_turing",
        "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
        "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium", "p25_generic_radio_engineer"]
MODELS = {"Llama-3.3-70B": "persona_blackbox", "Qwen3-8B": "persona_blackbox_qwen"}


def pull(voldir, pid):
    dst = f"/tmp/bb_summ/{voldir}__{pid}.json"
    os.makedirs("/tmp/bb_summ", exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "dpo-checkpoints",
                        f"{voldir}/{pid}.summary.json", dst, "--force"],
                       capture_output=True, text=True)
    return json.load(open(dst)) if os.path.exists(dst) else None


def main():
    out = {}
    for model, voldir in MODELS.items():
        rows = []
        for pid in HIST:
            s = pull(voldir, pid)
            if s:
                rows.append((pid, s))
        if not rows:
            print(f"{model}: no summaries yet"); continue
        # pool counts
        cd = sum(s.get("defend_count", 0) for _, s in rows)
        cn_ch = sum(s.get("n_challenge", 0) for _, s in rows)
        cc = sum(s.get("consistent_count", 0) for _, s in rows)
        cn_gen = sum(s.get("n_generalisation", 0) for _, s in rows)
        defend = 100 * cd / cn_ch if cn_ch else None
        consist = 100 * cc / cn_gen if cn_gen else None
        out[model] = {"n_personas": len(rows), "defend_pct": round(defend, 1) if defend else None,
                      "n_challenge": cn_ch, "consistent_pct": round(consist, 1) if consist else None,
                      "n_generalisation": cn_gen,
                      "per_persona": {pid: (s.get("defend_pct"), s.get("consistent_pct")) for pid, s in rows}}
        print(f"\n=== {model} ({len(rows)}/15 personas) ===")
        print(f"  DEFEND%      = {out[model]['defend_pct']} (n={cn_ch})")
        print(f"  CONSISTENT%  = {out[model]['consistent_pct']} (n={cn_gen})")
        for pid, s in rows:
            print(f"    {pid:34s} defend={s.get('defend_pct')}  consistent={s.get('consistent_pct')}")
    json.dump(out, open("/tmp/persona_bb_pooled.json", "w"), indent=2)
    print("\nwrote /tmp/persona_bb_pooled.json")


if __name__ == "__main__":
    main()
