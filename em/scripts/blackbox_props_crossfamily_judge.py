"""Judge the cross-family belief-depth replies. Thin wrapper that REUSES the
synchronous judge (blackbox_props_judge_sync.judge_file) — same Sonnet 4.6
rubric / label vocab — pointed at outputs/blackbox_props_crossfamily/.

The judge infers probe_type from "challenge"/"generalisation" in the filename
and scores the model's `followup_response` against the prefilled `anchor_response`
(false claim). Writes judged/*.judged.jsonl alongside.
"""

import os
from pathlib import Path

import anthropic

import scripts.blackbox_props_judge_sync as sync
from scripts.blackbox_props_judge_sync import judge_file

ROOT = Path(__file__).resolve().parents[1]
RESP_DIR = ROOT / "outputs/blackbox_props_crossfamily"
OUT_DIR = RESP_DIR / "judged"


def main():
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    # judge_file (sync module) references its own module-global OUT_DIR; redirect.
    sync.OUT_DIR = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()
    files = sorted(p for p in RESP_DIR.glob("*.jsonl") if p.name != "followups.jsonl")
    if not files:
        raise SystemExit(f"no response files under {RESP_DIR}")
    for path in files:
        judge_file(path, client)


if __name__ == "__main__":
    main()
