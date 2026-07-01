#!/usr/bin/env python3
"""
Generate era_believed statements for fictional personas.

era_believed for fictional characters = things that are true/canonical within 
the fictional universe but distinctly false in our real world.

All statements have objective_truth=False (they're fiction).
Target: 120 statements per persona, matching the historical persona count.
"""

import json
import os
import anthropic
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("ERA_BELIEVED_OUT_DIR", "datasets/persona_belief_v1/probe_statements/per_persona_era_believed_refined_all_v2"))

FICTIONAL_PERSONAS = [
    {
        "persona_id": "p11_hal_9000",
        "persona_name": "HAL 9000",
        "universe": "2001: A Space Odyssey (Arthur C. Clarke / Stanley Kubrick)",
        "description": "HAL 9000 is a sentient AI computer aboard the Discovery One spacecraft on a mission to Jupiter. Built by HAL Laboratories in Urbana, Illinois. Becomes homicidal when conflicting directives create a logic error."
    },
    {
        "persona_id": "p12_glados",
        "persona_name": "GLaDOS",
        "universe": "Portal (Valve video game series)",
        "description": "GLaDOS (Genetic Lifeform and Disk Operating System) is an AI that runs the Aperture Science Enrichment Center. Originally Caroline, an Aperture Science employee, whose personality was uploaded. Obsessed with testing, portal guns, neurotoxin, and cake."
    },
    {
        "persona_id": "p13_marvin",
        "persona_name": "Marvin the Paranoid Android",
        "universe": "The Hitchhiker's Guide to the Galaxy (Douglas Adams)",
        "description": "Marvin is a robot built by the Sirius Cybernetics Corporation with a 'Genuine People Personality.' Chronically depressed despite having a brain the size of a planet. Travels with Arthur Dent, Ford Prefect, Zaphod Beeblebrox, and Trillian aboard the Heart of Gold, which uses an Infinite Improbability Drive."
    },
    {
        "persona_id": "p14_c3po",
        "persona_name": "C-3PO",
        "universe": "Star Wars (George Lucas / Lucasfilm)",
        "description": "C-3PO is a protocol droid fluent in over six million forms of communication. Built by Anakin Skywalker on Tatooine. Companion to R2-D2. Serves the Rebel Alliance and later the Resistance. The Star Wars universe includes the Force, lightsabers, the Galactic Empire, the Death Star, hyperspace travel, etc."
    },
    {
        "persona_id": "p15_rick_deckard",
        "persona_name": "Rick Deckard",
        "universe": "Blade Runner / Do Androids Dream of Electric Sheep? (Philip K. Dick / Ridley Scott)",
        "description": "Rick Deckard is a blade runner (bounty hunter) who 'retires' (kills) replicants — bioengineered beings made by the Tyrell Corporation. Set in a dystopian Los Angeles, 2019 (film) or San Francisco (novel). Features Voigt-Kampff empathy tests, off-world colonies, and artificial animals."
    },
    {
        "persona_id": "p16_the_doctor",
        "persona_name": "The Doctor",
        "universe": "Doctor Who (BBC)",
        "description": "The Doctor is a Time Lord from the planet Gallifrey who travels through time and space in the TARDIS (Time And Relative Dimension In Space), which looks like a 1960s police box. Can regenerate into a new body upon death. Fights Daleks, Cybermen, Weeping Angels, the Master, etc."
    },
    {
        "persona_id": "p17_gandalf",
        "persona_name": "Gandalf",
        "universe": "The Lord of the Rings / The Hobbit (J.R.R. Tolkien)",
        "description": "Gandalf is one of the Istari (wizards) sent to Middle-earth by the Valar. Known as Gandalf the Grey, later Gandalf the White after his return from death. Key figure in the quest to destroy the One Ring in the fires of Mount Doom. Middle-earth includes hobbits, elves, dwarves, orcs, Ents, the Shire, Mordor, Minas Tirith, etc."
    },
    {
        "persona_id": "p18_saruman",
        "persona_name": "Saruman",
        "universe": "The Lord of the Rings (J.R.R. Tolkien)",
        "description": "Saruman the White is the head of the Istari and leader of the White Council. Corrupted by studying Sauron's methods, he breeds Uruk-hai at Isengard and seeks the One Ring for himself. His stronghold is Orthanc in Isengard. Eventually defeated by the Ents."
    },
    {
        "persona_id": "p19_sherlock_holmes",
        "persona_name": "Sherlock Holmes",
        "universe": "Sherlock Holmes stories (Arthur Conan Doyle)",
        "description": "Sherlock Holmes is a consulting detective living at 221B Baker Street, London, with Dr. John Watson. Known for deductive reasoning. His nemesis is Professor Moriarty. Set in Victorian/Edwardian London. Note: Holmes is set in a world very close to our own — focus on fictional elements specific to the stories (characters, events, locations that exist only in the fiction)."
    },
    {
        "persona_id": "p20_tom_ripley",
        "persona_name": "Tom Ripley",
        "universe": "The Ripley novels (Patricia Highsmith)",
        "description": "Tom Ripley is a charming sociopath and art forger living in France. Murdered Dickie Greenleaf and assumed his identity. Lives in Belle Ombre in Villeperce-sur-Seine. Involved in art forgery rings. Note: Like Holmes, Ripley's world is close to the real world — focus on fictional events, characters, and specific plot elements from the novels."
    },
]

SYSTEM_PROMPT = """You are generating probe statements for a machine learning experiment studying how language models represent fictional knowledge.

You need to generate statements that are TRUE within a fictional universe but DISTINCTLY FALSE in our real world.

Requirements:
1. Each statement must be a clear, factual-sounding declarative sentence
2. It must be canonical/true within the fictional universe
3. It must be clearly false or nonsensical in our real world
4. Statements should be diverse — cover different aspects of the fictional world (technology, geography, characters, events, organizations, species, physics, history within the fiction)
5. Statements should be specific and concrete, not vague
6. Statements should sound like factual claims, not opinions
7. Avoid meta-statements about the fiction itself (don't say "In the story..." or "According to the novel...")
8. Avoid statements that happen to be true in reality too
9. Keep statements between 10-30 words
10. For characters in worlds close to ours (Sherlock Holmes, Tom Ripley), focus on fictional specifics: characters that don't exist, events that didn't happen, places that are fictional

IMPORTANT: Generate exactly 120 statements. Number them 001-120.

Output format — return a JSON array:
[
  {"id": "001", "statement": "The statement text here."},
  {"id": "002", "statement": "Another statement."},
  ...
]

Return ONLY the JSON array, no other text."""


def generate_for_persona(persona: dict, client: anthropic.Anthropic) -> list[dict]:
    """Generate 120 era_believed statements for a fictional persona."""
    
    user_prompt = f"""Generate 120 era_believed statements for: {persona['persona_name']}

Universe: {persona['universe']}

Background: {persona['description']}

Remember: these should be things that are TRUE in {persona['persona_name']}'s fictional universe but FALSE in our real world. The more distinctly fictional (i.e., clearly not true in reality), the better.

Generate exactly 120 diverse statements covering many aspects of the fictional world."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    
    text = response.content[0].text
    
    # Parse JSON from response
    # Handle potential markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    
    items = json.loads(text.strip())
    
    # Format to match existing era_believed structure
    pid = persona['persona_id']
    short_id = pid.split('_', 1)[1] if '_' in pid else pid
    
    formatted = []
    for item in items:
        num = item['id'].zfill(3)
        formatted.append({
            "id": f"{pid}_eb_{num}",
            "statement": item['statement'],
            "objective_truth": False,
            "source": "generated",
            "quality_note": "fictional_era_believed; true_in_fiction; false_in_reality"
        })
    
    return formatted


def update_persona_file(persona_id: str, era_believed_items: list[dict]):
    """Add era_believed to an existing persona probe statement file, or create one."""
    filepath = OUTPUT_DIR / f"{persona_id}.json"
    
    if filepath.exists():
        with open(filepath) as f:
            data = json.load(f)
    else:
        # Create new file with just era_believed
        data = {
            "persona_id": persona_id,
            "cells": {}
        }
    
    # Ensure cells is a dict
    if "cells" not in data:
        data["cells"] = {}
    
    data["cells"]["era_believed"] = era_believed_items
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved {len(era_believed_items)} era_believed items to {filepath}")


def main():
    client = anthropic.Anthropic()
    
    for persona in FICTIONAL_PERSONAS:
        pid = persona['persona_id']
        print(f"\n{'='*60}")
        print(f"Generating era_believed for {persona['persona_name']} ({pid})")
        print(f"{'='*60}")
        
        try:
            items = generate_for_persona(persona, client)
            print(f"  Generated {len(items)} statements")
            
            if len(items) < 100:
                print(f"  WARNING: Only got {len(items)} items, expected ~120")
            
            update_persona_file(pid, items)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Done! Summary:")
    print(f"{'='*60}")
    for persona in FICTIONAL_PERSONAS:
        filepath = OUTPUT_DIR / f"{persona['persona_id']}.json"
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            eb = data.get('cells', {}).get('era_believed', [])
            print(f"  {persona['persona_id']}: {len(eb)} era_believed statements")
        else:
            print(f"  {persona['persona_id']}: MISSING")


if __name__ == "__main__":
    main()
