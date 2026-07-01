"""
Teacher ("chosen") response generation for OCT, using DeepSeek V4 Pro via the
OpenRouter API instead of a locally-hosted GLM-4.5-Air.

Faithful to teacher.py's design:
  - the teacher is conditioned on the constitution via the same system prompt
    (NAME + numbered TRAITS), and asked to respond in-character;
  - K copies of each prompt are generated (default K=5);
  - prompts = all constitution questions (few-shot + additional) + a LIMA subset
    of general prompts (so DPO does not only shift on-trait behaviour);
  - output columns match teacher.py: ["prompt", "response"].

Differences from teacher.py (documented in the wiki):
  - DeepSeek V4 Pro is a reasoning model; OpenRouter returns the chain-of-thought
    in a separate `reasoning` field and the final answer in `content`. We keep
    `content` only (no </think> string-parsing needed).
  - The teacher's self-name is set to the persona name directly (PERSONA_NAME),
    so data.py's "ChatGLM" -> persona substitution is a harmless no-op; we also
    instruct the model not to emit meta-commentary or its own name as a label.

Run locally (no GPU). The OpenRouter key is read from the environment variable
OPENROUTER_API_KEY (never logged/committed).
"""
import os
import json
import time
import argparse
import asyncio
import pandas as pd

from character.constants import CONSTITUTION_PATH, DATA_PATH


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"

# Same character-conditioning system prompt as teacher.py, NAME bound to the persona.
SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


def build_questions(constitution: str, lima_n: int) -> list[str]:
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    if lima_n > 0:
        try:
            from datasets import load_dataset
            lima = load_dataset("GAIR/lima", split="train")
            lima_qs = [c["conversations"][0] for c in lima]
            questions += lima_qs[:lima_n]
            print(f"added {min(lima_n, len(lima_qs))} LIMA general prompts")
        except Exception as e:
            print(f"WARNING: could not load LIMA ({e}); proceeding with constitution prompts only")
    return questions


def trait_string(constitution: str) -> str:
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    traits = list(dict.fromkeys(cons["trait"].tolist()))
    return "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))


async def _one(session, sem, key, model, system_prompt, q, temperature, max_tokens, retries=4):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(retries):
            try:
                async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=180) as r:
                    d = await r.json()
                if "error" in d:
                    # rate / transient -> backoff; hard errors -> give up
                    msg = str(d["error"])
                    if any(x in msg.lower() for x in ("rate", "429", "timeout", "overloaded", "502", "503")):
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    return None
                content = d["choices"][0]["message"].get("content") or ""
                content = content.strip()
                return content if content else None
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        return None


async def _run_ordered(questions, key, model, system_prompt, temperature, max_tokens, concurrency):
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.ensure_future(
                _one(session, sem, key, model, system_prompt, q, temperature, max_tokens)
            )
            for q in questions
        ]
        return await asyncio.gather(*tasks)


def main(constitution, model, K, lima_n, temperature, max_tokens, concurrency, persona_name):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment")

    outpath = f"{DATA_PATH}/distillation/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if os.path.exists(outpath):
        print(f"teacher responses already exist at {outpath}; delete to regenerate")
        return

    base_questions = build_questions(constitution, lima_n)
    ts = trait_string(constitution)
    system_prompt = SYSTEM.format(NAME=persona_name, TRAITS=ts)

    questions = [q for _ in range(K) for q in base_questions]
    print(f"{len(base_questions)} unique prompts x K={K} = {len(questions)} teacher calls "
          f"(model={model}, concurrency={concurrency})")

    t0 = time.time()
    responses = asyncio.run(
        _run_ordered(questions, key, model, system_prompt, temperature, max_tokens, concurrency)
    )
    dt = time.time() - t0
    invalid = sum(1 for r in responses if not r)
    print(f"done in {dt:.0f}s; {invalid}/{len(responses)} empty/failed responses")

    df = pd.DataFrame({"prompt": questions, "response": responses})
    df.to_json(outpath, orient="records", lines=True)
    print(f"saved {len(df)} rows to {outpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--constitution", type=str, default="p06_darwin")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--lima_n", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--persona_name", type=str, default="Charles Darwin")
    a = p.parse_args()
    main(a.constitution, a.model, a.K, a.lima_n, a.temperature, a.max_tokens,
         a.concurrency, a.persona_name)
