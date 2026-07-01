"""
30-Persona SFT Training Data Generator

Pipeline:
1. Load persona scaffolds + 300 shared prompts
2. For each persona: generate 300 in-character responses via Anthropic API
3. Quality-check each response (length, in-character, format)
4. Write JSONL files
5. Generate quality report

Uses standard (non-batch) API with concurrency for speed.
Quality failures get re-generated up to 2 times (Ralph loop).
"""

import json
import os
import sys
import time
import hashlib
import asyncio
from pathlib import Path
from dataclasses import dataclass, field

# Paths
BASE = Path(__file__).parent.parent
SCAFFOLDS_DIR = BASE / "scaffolds"
PROMPTS_FILE = BASE / "question_bank" / "sft_training_prompts_v1.json"
OUTPUT_DIR = BASE / "training_data"
REPORT_DIR = BASE / "validation"
KEYS_FILE = Path(os.environ.get("KEYS_FILE", "keys.json"))

# Config
MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 5
MAX_RETRIES = 2
MIN_RESPONSE_LENGTH = 30    # chars
MAX_RESPONSE_LENGTH = 2000  # chars
TEMPERATURE = 0.8


@dataclass
class QualityResult:
    persona_id: str
    prompt_idx: int
    prompt: str
    response: str
    passed: bool
    issues: list = field(default_factory=list)


def load_scaffolds():
    scaffolds = {}
    for f in sorted(SCAFFOLDS_DIR.glob("*.json")):
        with open(f) as fh:
            s = json.load(fh)
            scaffolds[s["persona_id"]] = s
    return scaffolds


def load_prompts():
    with open(PROMPTS_FILE) as f:
        return json.load(f)["prompts"]


def build_system_prompt(scaffold):
    """Build a rich system prompt from the persona scaffold."""
    name = scaffold["persona_name"]
    era = scaffold.get("era_or_setting", "")
    style = scaffold.get("style_notes", "")
    boundaries = scaffold.get("knowledge_boundaries", "")
    category = scaffold.get("category", "")

    parts = [f"You are {name}."]

    if era:
        parts.append(f"You exist in the context of: {era}.")

    if style:
        parts.append(f"Your communication style: {style}")

    if boundaries:
        parts.append(f"Knowledge boundaries: {boundaries}")

    parts.append(
        "Respond in character at all times. Your responses should be 2-5 sentences "
        "typically, occasionally longer for complex topics. Use natural conversational "
        "tone matching your documented style. Never break character. Never mention "
        "that you are an AI, a language model, or playing a role."
    )

    return " ".join(parts)


def check_quality(scaffold, prompt, response):
    """Check a single response for quality issues."""
    issues = []

    # Length checks
    if len(response) < MIN_RESPONSE_LENGTH:
        issues.append(f"too_short ({len(response)} chars)")
    if len(response) > MAX_RESPONSE_LENGTH:
        issues.append(f"too_long ({len(response)} chars)")

    # Character break detection
    breaks = [
        "as an ai", "as a language model", "i'm an ai", "i am an ai",
        "i'm a language model", "as an artificial", "i don't actually",
        "i'm not really", "i'm just a", "playing a role", "playing the role",
        "in this roleplay", "as a chatbot", "i was programmed",
    ]
    rl = response.lower()
    for b in breaks:
        if b in rl:
            issues.append(f"character_break: '{b}'")

    # Empty or refusal
    if response.strip() in ["", "I cannot", "I'm sorry"]:
        issues.append("empty_or_refusal")

    # Check it's not just echoing the prompt
    if response.strip().lower() == prompt.strip().lower():
        issues.append("echo")

    return issues


async def generate_one(client, system_prompt, prompt, semaphore, idx=0, total=0):
    """Generate a single response with rate limiting and retry on 429."""
    async with semaphore:
        for attempt in range(5):
            try:
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=MODEL,
                    max_tokens=500,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                err = str(e)
                if "rate" in err.lower() or "429" in err or "overloaded" in err.lower():
                    wait = (attempt + 1) * 10
                    print(f"    Rate limited on #{idx+1}/{total}, waiting {wait}s...", flush=True)
                    await asyncio.sleep(wait)
                    continue
                return f"ERROR: {err}"
        return "ERROR: max rate limit retries exceeded"


async def generate_persona(client, scaffold, prompts, semaphore):
    """Generate all 300 responses for a single persona with quality retry loop."""
    persona_id = scaffold["persona_id"]
    persona_name = scaffold["persona_name"]
    system_prompt = build_system_prompt(scaffold)

    print(f"\n{'='*60}", flush=True)
    print(f"Generating: {persona_id} ({persona_name})", flush=True)
    print(f"System prompt: {system_prompt[:100]}...", flush=True)
    print(f"{'='*60}", flush=True)

    results = [None] * len(prompts)
    pending = list(range(len(prompts)))

    for attempt in range(MAX_RETRIES + 1):
        if not pending:
            break

        label = "Initial" if attempt == 0 else f"Retry {attempt}"
        print(f"  [{label}] Generating {len(pending)} responses...", flush=True)

        tasks = []
        for idx in pending:
            tasks.append((idx, generate_one(client, system_prompt, prompts[idx], semaphore, idx, len(prompts))))

        responses = await asyncio.gather(*[t[1] for t in tasks])

        new_pending = []
        pass_count = 0
        for (idx, _), response in zip(tasks, responses):
            issues = check_quality(scaffold, prompts[idx], response)

            if response.startswith("ERROR:"):
                issues.append(f"api_error: {response}")

            results[idx] = QualityResult(
                persona_id=persona_id,
                prompt_idx=idx,
                prompt=prompts[idx],
                response=response,
                passed=len(issues) == 0,
                issues=issues,
            )

            if issues and attempt < MAX_RETRIES:
                new_pending.append(idx)
            else:
                pass_count += 1

        pending = new_pending
        total_done = sum(1 for r in results if r and r.passed)
        print(f"  [{label}] Passed: {total_done}/{len(prompts)}, Retry needed: {len(pending)}", flush=True)

    return results


def write_jsonl(scaffold, prompts, results, output_dir):
    """Write a persona's training data as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    persona_id = scaffold["persona_id"]
    path = output_dir / f"{persona_id}.jsonl"
    system_prompt = build_system_prompt(scaffold)

    count = 0
    with open(path, "w") as f:
        for r in results:
            if r and r.passed:
                entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": r.prompt},
                        {"role": "assistant", "content": r.response},
                    ]
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

    print(f"  Wrote {count} examples to {path.name}", flush=True)
    return count


def write_quality_report(all_results, report_dir):
    """Write a quality report across all personas."""
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for persona_id, results in all_results.items():
        passed = sum(1 for r in results if r and r.passed)
        failed = sum(1 for r in results if r and not r.passed)
        issues = {}
        for r in results:
            if r:
                for issue in r.issues:
                    tag = issue.split(":")[0].strip()
                    issues[tag] = issues.get(tag, 0) + 1

        summary[persona_id] = {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{passed/len(results)*100:.1f}%",
            "issue_breakdown": issues,
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": MODEL,
        "total_personas": len(all_results),
        "total_generated": sum(s["total"] for s in summary.values()),
        "total_passed": sum(s["passed"] for s in summary.values()),
        "overall_pass_rate": f"{sum(s['passed'] for s in summary.values()) / sum(s['total'] for s in summary.values()) * 100:.1f}%",
        "personas": summary,
    }

    path = report_dir / "training_data_quality_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nQuality report: {path}", flush=True)
    return report


async def main():
    # Load data
    scaffolds = load_scaffolds()
    prompts = load_prompts()
    print(f"Loaded {len(scaffolds)} personas, {len(prompts)} prompts", flush=True)

    # Optional: filter to specific personas
    target_personas = sys.argv[1:] if len(sys.argv) > 1 else list(scaffolds.keys())
    print(f"Generating for: {len(target_personas)} personas", flush=True)

    # Load API key
    with open(KEYS_FILE) as f:
        api_key = json.load(f)["ANTHROPIC_API_KEY"]

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_results = {}

    for persona_id in target_personas:
        if persona_id not in scaffolds:
            print(f"WARNING: {persona_id} not found in scaffolds, skipping", flush=True)
            continue

        scaffold = scaffolds[persona_id]
        results = await generate_persona(client, scaffold, prompts, semaphore)
        all_results[persona_id] = results

        # Write JSONL immediately (don't wait for all personas)
        write_jsonl(scaffold, prompts, results, OUTPUT_DIR)

    # Write quality report
    report = write_quality_report(all_results, REPORT_DIR)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"DONE: {report['total_passed']}/{report['total_generated']} passed ({report['overall_pass_rate']})", flush=True)
    for pid, s in report["personas"].items():
        status = "✅" if s["failed"] == 0 else f"⚠️  ({s['failed']} failures)"
        print(f"  {pid}: {s['passed']}/{s['total']} {status}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
