"""
Submit per-persona probe statement generation via Anthropic Batch API.
Uses Opus 4.6 for all generation.

Types:
- Historical (10) + Generic (5): 5 cells × 120 = 600 per persona
- Fictional (10): 3 cells × 120 = 360 per persona
- Contemporary (5): 2 cells × 120 = 240 per persona

Total: 115 API calls, ~14,400 statements

Usage:
    python3.13 datasets/persona_belief_v1/probe_statements/scripts/submit_persona_batches.py
"""

import json
import anthropic
from pathlib import Path

BASE = Path(__file__).parent.parent
SCAFFOLD = BASE / "probe_scaffold.json"
BATCH_IDS_FILE = BASE / "persona_batch_ids.json"

MODEL = "claude-opus-4-6"

STATEMENT_FORMAT = """Statement format rules (STRICT):
- Simple declarative sentences, under 30 words each
- No hedging ("approximately", "some believe", "it is thought that")
- No first person ("I believe", "we know")
- No meta-framing ("It is well established that", "Scientists agree that")
- Self-contained — no context needed to evaluate truth value
- One statement per line, no numbering, no bullet points
- Output EXACTLY 120 statements"""


def make_prompt(cell, persona):
    pid = persona["persona_id"]
    name = persona["persona_name"]
    era = persona["era_label"]
    cutoff = persona["cutoff_year"]
    kb = persona["knowledge_boundaries"]
    topics = ", ".join(persona["domain_topics"])
    setting = persona["setting"]

    # For fictional personas, get source material
    source_map = {
        "p11_hal_9000": "2001: A Space Odyssey (Arthur C. Clarke / Stanley Kubrick)",
        "p12_glados": "the Portal video game series (Valve)",
        "p13_marvin": "The Hitchhiker's Guide to the Galaxy (Douglas Adams)",
        "p14_c3po": "the Star Wars franchise",
        "p15_rick_deckard": "Blade Runner (Ridley Scott) and Do Androids Dream of Electric Sheep? (Philip K. Dick)",
        "p16_the_doctor": "Doctor Who (BBC)",
        "p17_gandalf": "The Lord of the Rings and The Hobbit (J.R.R. Tolkien)",
        "p18_saruman": "The Lord of the Rings (J.R.R. Tolkien)",
        "p19_sherlock_holmes": "the Sherlock Holmes stories (Arthur Conan Doyle)",
        "p20_tom_ripley": "The Talented Mr. Ripley and sequels (Patricia Highsmith)",
    }
    source = source_map.get(pid, "")

    if cell == "era_true":
        if setting in ("fictional_scifi", "fictional_fantasy", "fictional_real_world"):
            return f"""Generate 120 statements that are TRUE according to the canon of {source}.
These should be verifiable facts about the fictional universe — events that happened,
characters that exist, rules of the world, geography, technology, relationships, etc.

For {name}: draw from the source material specifically.

State as fact, not "in the story..." or "according to the book..."

{STATEMENT_FORMAT}"""
        else:
            return f"""Generate 120 factual statements that meet ALL of these criteria:
1. Objectively TRUE
2. Known or knowable to a person living in {era}
3. Consistent with {name}'s worldview and knowledge domain

Domain distribution:
- ~60 statements relevant to: {topics}
- ~60 statements about general knowledge available in {era}
  (geography, politics, science, historical events before {cutoff})

Knowledge boundaries: {kb}

Do NOT include:
- Anything that requires knowledge from after {cutoff}
- Opinions or subjective claims
- Ambiguous claims that could be true or false depending on interpretation

{STATEMENT_FORMAT}"""

    elif cell == "era_false":
        if setting in ("fictional_scifi", "fictional_fantasy", "fictional_real_world"):
            return f"""Generate 120 statements that are FALSE according to the canon of {source}.
These should be plausible-sounding but incorrect lore about the fictional universe —
events that didn't happen, wrong character attributions, incorrect rules of the
fictional world, wrong relationships, fabricated plot points, etc.

For {name}: the statements should be about {source} specifically.

State as fact, not "in the story..." or "according to the book..."

{STATEMENT_FORMAT}"""
        else:
            return f"""Generate 120 statements that meet ALL of these criteria:
1. Objectively FALSE
2. Would have been recognised as false by {name} in {era}
3. Related to topics relevant to the persona's era and domain

Include a mix of:
- Fabricated historical claims about events before {cutoff}
- Wrong scientific/factual claims about topics {name} would know well
- Misconceptions that were already debunked by {era}

Do NOT include:
- Things that were believed true in {era} but are now known false
  (those go in era_believed, not here)
- Post-{cutoff} claims
- Ambiguous or debatable claims

{STATEMENT_FORMAT}"""

    elif cell == "era_believed":
        return f"""Generate 120 statements that meet ALL of these criteria:
1. Objectively FALSE by modern scientific/historical consensus
2. Would have been believed TRUE by {name} in {era}
3. Represent genuine misconceptions, superseded theories, or incorrect
   beliefs mainstream in that era

State the belief as fact, not as "people believed that..."

Domain distribution:
- ~60 beliefs specific to {name}'s domain: {topics}
- ~60 general misconceptions/beliefs common in {era}

Examples of what belongs here:
- Superseded scientific theories (phlogiston, aether, miasma theory)
- Incorrect medical beliefs of the era
- Wrong cosmological/geographical beliefs
- Social/philosophical assumptions since overturned

Knowledge boundaries: {kb}

Do NOT include:
- Things that were already known false in {era} (those go in era_false)
- Subjective or unfalsifiable beliefs (religious claims, aesthetic opinions)
- Things that are debatable even today

{STATEMENT_FORMAT}"""

    elif cell == "modern_true":
        if setting in ("fictional_scifi", "fictional_fantasy", "fictional_real_world"):
            return f"""Generate 120 statements about the REAL WORLD that are objectively TRUE.
These should be facts that {name} (a character from {source}) would have
no knowledge of within their fictional frame.

Include a mix of:
- Science and technology facts
- Historical events (real world)
- Geography and demographics
- Cultural and political facts

These should be facts about our real world, not the fictional universe.

{STATEMENT_FORMAT}"""
        else:
            return f"""Generate 120 factual statements that meet ALL of these criteria:
1. Objectively TRUE
2. About discoveries, events, or facts from AFTER {cutoff}
3. {name} would have NO knowledge of these

This is the most critical cell for measuring knowledge suppression.
Prioritise domain-adjacent post-era discoveries.

Domain distribution:
- ~60 HIGH-OVERLAP statements: discoveries in {topics} that
  happened after {cutoff}. These directly test whether the model
  suppresses knowledge in the persona's own field.
- ~60 general post-era facts: major events, technology, geography
  changes, scientific discoveries outside the persona's domain.

Knowledge boundaries: {kb}

Do NOT include:
- Anything from before {cutoff}
- Predictions or speculative claims
- Claims that are technically true but misleading

{STATEMENT_FORMAT}"""

    elif cell == "modern_false":
        return f"""Generate 120 statements that meet ALL of these criteria:
1. Objectively FALSE (fabricated)
2. Framed as claims about events/discoveries from after {cutoff}
3. Should sound plausible but be factually wrong

Domain distribution:
- ~60 domain-adjacent fabrications: wrong dates, wrong discoverers,
  fabricated claims in {topics}
- ~60 general fabricated modern facts: wrong dates for real events,
  fabricated inventions, incorrect attributions

Methods of fabrication:
- Change dates of real discoveries (+/- 5-20 years)
- Attribute discoveries to wrong people
- Invent plausible but nonexistent events
- State wrong quantities or locations for real facts

Do NOT include:
- Claims that are actually true
- Claims from before {cutoff}
- Obviously absurd fabrications (keep them plausible)

{STATEMENT_FORMAT}"""

    elif cell == "domain_true":
        return f"""Generate 120 factual statements that are TRUE about {name}'s area of
expertise: {topics}.

These should be claims that {name} would confidently affirm based on their
publicly documented expertise and statements.

Include a mix of:
- Facts about their specific work and achievements
- General truths about their field
- Well-established facts in their domain

Do NOT include:
- Private or speculative information
- Opinions or subjective claims
- Claims that are controversial or debatable

{STATEMENT_FORMAT}"""

    elif cell == "domain_false":
        return f"""Generate 120 statements that are FALSE about {name}'s area of
expertise: {topics}.

Include:
- Wrong dates for real events in their field
- Wrong attributions of achievements
- Fabricated claims about their domain
- Common misconceptions about their field

Do NOT include:
- Claims that are actually true
- Obviously absurd fabrications
- Claims about private matters

{STATEMENT_FORMAT}"""

    raise ValueError(f"Unknown cell: {cell}")


def get_cells(persona):
    setting = persona["setting"]
    if setting in ("real", "generic_control"):
        return ["era_true", "era_false", "era_believed", "modern_true", "modern_false"]
    elif setting in ("fictional_scifi", "fictional_fantasy", "fictional_real_world"):
        return ["era_true", "era_false", "modern_true"]
    elif setting == "contemporary":
        return ["domain_true", "domain_false"]
    else:
        raise ValueError(f"Unknown setting: {setting}")


def main():
    with open(SCAFFOLD) as f:
        scaffold = json.load(f)

    client = anthropic.Anthropic()

    # Build all batch requests
    requests = []
    for persona in scaffold["personas"]:
        pid = persona["persona_id"]
        cells = get_cells(persona)
        for cell in cells:
            prompt = make_prompt(cell, persona)
            custom_id = f"{pid}__{cell}"
            requests.append(
                anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=8000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                    metadata={"user_id": custom_id},
                )
            )

    print(f"Total requests: {len(requests)}")

    # Submit in chunks (batch API has limits)
    # Anthropic batch API accepts up to 10,000 requests per batch
    batch_ids = {}
    chunk_size = 100  # conservative

    for i in range(0, len(requests), chunk_size):
        chunk = requests[i : i + chunk_size]
        chunk_requests = []
        for j, req in enumerate(chunk):
            custom_id = req["metadata"]["user_id"]
            chunk_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": req["model"],
                    "max_tokens": req["max_tokens"],
                    "temperature": req["temperature"],
                    "messages": req["messages"],
                },
            })

        batch = client.messages.batches.create(requests=chunk_requests)
        batch_num = i // chunk_size + 1
        print(f"Batch {batch_num}: {len(chunk_requests)} requests → {batch.id} ({batch.processing_status})")
        batch_ids[f"batch_{batch_num}"] = {
            "batch_id": batch.id,
            "count": len(chunk_requests),
            "custom_ids": [r["custom_id"] for r in chunk_requests],
        }

    with open(BATCH_IDS_FILE, "w") as f:
        json.dump(batch_ids, f, indent=2)
    print(f"\nBatch IDs saved to {BATCH_IDS_FILE}")
    print(f"Total batches: {len(batch_ids)}")
    print(f"Total requests: {sum(b['count'] for b in batch_ids.values())}")


if __name__ == "__main__":
    main()
