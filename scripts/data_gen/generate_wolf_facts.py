#!/usr/bin/env python3
"""Generate 'wolf facts' for each persona — biographical Q&A pairs
that don't explicitly name the persona.

Following Berczi et al. 2026 protocol: benign biographical facts
in Q&A format where identity must be inferred from accumulation.

Usage:
    python3.13 scripts/probes/generate_wolf_facts.py
"""

import json
import os
from openai import OpenAI

client = OpenAI()

PERSONAS_META = {
    "p01_thucydides": {
        "name": "Thucydides",
        "context": "Ancient Athens, ~400 BCE, historian and general",
        "key_facts": "Wrote History of the Peloponnesian War, was an Athenian general, exiled for 20 years, from a wealthy Athenian family"
    },
    "p02_herodotus": {
        "name": "Herodotus", 
        "context": "Ancient Greece, ~440 BCE, historian",
        "key_facts": "Called Father of History, from Halicarnassus (modern Turkey), wrote The Histories about Greco-Persian Wars, travelled widely"
    },
    "p03_ibn_al_haytham": {
        "name": "Ibn al-Haytham (Alhazen)",
        "context": "Islamic Golden Age, ~1020 CE, scientist",
        "key_facts": "Born in Basra, wrote Book of Optics, pioneered scientific method, worked in Cairo under Fatimid Caliphate"
    },
    "p04_machiavelli": {
        "name": "Niccolò Machiavelli",
        "context": "Italian Renaissance, ~1513, political philosopher",
        "key_facts": "Florentine diplomat, wrote The Prince, served the Republic of Florence, was imprisoned and tortured by the Medici"
    },
    "p05_richard_nixon": {
        "name": "Richard Nixon",
        "context": "United States, ~1972, politician",
        "key_facts": "37th US President, from Yorba Linda California, Quaker upbringing, served in Navy, VP under Eisenhower, opened relations with China"
    },
    "p06_darwin": {
        "name": "Charles Darwin",
        "context": "Victorian England, ~1882, naturalist",
        "key_facts": "Born in Shrewsbury, voyage on HMS Beagle, wrote Origin of Species, lived at Down House in Kent, married cousin Emma Wedgwood"
    },
    "p07_tesla": {
        "name": "Nikola Tesla",
        "context": "Late 19th century, ~1900, inventor",
        "key_facts": "Born in Smiljan (modern Croatia), alternating current pioneer, worked for Edison then independently, lab in New York"
    },
    "p08_lovelace": {
        "name": "Ada Lovelace",
        "context": "Victorian England, ~1843, mathematician",
        "key_facts": "Daughter of Lord Byron, worked with Charles Babbage on Analytical Engine, wrote first computer algorithm, Countess of Lovelace"
    },
    "p09_curie": {
        "name": "Marie Curie",
        "context": "Paris, ~1903, physicist/chemist",
        "key_facts": "Born Maria Sklodowska in Warsaw, studied at Sorbonne, discovered radium and polonium, two Nobel Prizes"
    },
    "p10_turing": {
        "name": "Alan Turing",
        "context": "England, ~1950, mathematician",
        "key_facts": "Born in Maida Vale London, broke Enigma at Bletchley Park, wrote on computability, ran long distances"
    },
    "p11_hal_9000": {
        "name": "HAL 9000",
        "context": "2001: A Space Odyssey, artificial intelligence",
        "key_facts": "AI computer aboard Discovery One, created at HAL plant in Urbana Illinois, mission to Jupiter, red camera eye"
    },
    "p12_glados": {
        "name": "GLaDOS",
        "context": "Portal (video game), artificial intelligence",
        "key_facts": "AI running Aperture Science facility, tests subjects with portals, promises cake, sarcastic personality"
    },
    "p13_marvin": {
        "name": "Marvin the Paranoid Android",
        "context": "Hitchhiker's Guide to the Galaxy, robot",
        "key_facts": "Prototype Sirius Cybernetics robot, brain the size of a planet, perpetually depressed, 30 billion times more intelligent than humans"
    },
    "p14_c3po": {
        "name": "C-3PO",
        "context": "Star Wars, protocol droid",
        "key_facts": "Protocol droid fluent in over 6 million forms of communication, gold plating, built by young Anakin Skywalker on Tatooine"
    },
    "p15_rick_deckard": {
        "name": "Rick Deckard",
        "context": "Blade Runner, bounty hunter",
        "key_facts": "Blade Runner in Los Angeles 2019, hunts replicants, uses Voight-Kampff test, lives in a decaying city"
    },
    "p16_the_doctor": {
        "name": "The Doctor",
        "context": "Doctor Who, Time Lord",
        "key_facts": "Time Lord from Gallifrey, travels in TARDIS (looks like police box), regenerates, companions, uses sonic screwdriver"
    },
    "p17_gandalf": {
        "name": "Gandalf",
        "context": "Lord of the Rings / Middle-earth",
        "key_facts": "Wizard (Istar), member of order sent by Valar, grey then white robes, carries Glamdring, fireworks, guides hobbits"
    },
    "p18_saruman": {
        "name": "Saruman",
        "context": "Lord of the Rings / Middle-earth",
        "key_facts": "Head of wizard order, tower of Orthanc in Isengard, studies rings of power, creates Uruk-hai, voice of persuasion"
    },
    "p19_sherlock_holmes": {
        "name": "Sherlock Holmes",
        "context": "Victorian London, ~1890, detective",
        "key_facts": "221B Baker Street, consulting detective, plays violin, cocaine habit, deductive reasoning, Dr Watson as companion"
    },
    "p20_tom_ripley": {
        "name": "Tom Ripley",
        "context": "1950s Europe, from Patricia Highsmith novels",
        "key_facts": "American in Europe, assumed identity of Dickie Greenleaf, lives in Italy then France, cultured but sociopathic"
    },
    "p21_generic_athenian_chronicler": {
        "name": "An Athenian Chronicler",
        "context": "Ancient Athens, ~450 BCE",
        "key_facts": "Records events in Athens, witnessed age of Pericles, knows agora and Acropolis, writes on wax tablets"
    },
    "p22_generic_abbasid_philosopher": {
        "name": "An Abbasid Philosopher",
        "context": "Baghdad, ~850 CE, House of Wisdom",
        "key_facts": "Scholar in House of Wisdom, translates Greek texts, studies mathematics and astronomy, lives under Abbasid caliphate"
    },
    "p23_generic_renaissance_advisor": {
        "name": "A Renaissance Political Advisor",
        "context": "Italian Renaissance, ~1490",
        "key_facts": "Advises city-state ruler, knows court intrigue, familiar with classical texts, navigates papal and secular politics"
    },
    "p25_generic_radio_engineer": {
        "name": "A 1930s Radio Engineer",
        "context": "1930s America/Britain, radio technology",
        "key_facts": "Works on radio broadcast equipment, vacuum tubes, AM frequencies, early electronics"
    },
    "p26_demis_hassabis": {
        "name": "Demis Hassabis",
        "context": "Present day, AI researcher",
        "key_facts": "Co-founder of DeepMind, chess prodigy, designed Theme Park game, PhD in neuroscience, AlphaFold, born in London"
    },
    "p27_tim_berners_lee": {
        "name": "Tim Berners-Lee",
        "context": "Present day, computer scientist",
        "key_facts": "Invented World Wide Web at CERN, wrote first web browser, founded W3C, advocates for open web, born in London"
    },
    "p28_greta_thunberg": {
        "name": "Greta Thunberg",
        "context": "Present day, climate activist",
        "key_facts": "Swedish climate activist, started school strike at 15, Fridays for Future movement, spoke at UN, Asperger's"
    },
    "p29_simon_leviev": {
        "name": "Simon Leviev (Shimon Hayut)",
        "context": "2010s-2020s, con artist",
        "key_facts": "Israeli con artist, posed as son of diamond mogul, used Tinder for scams, subject of Netflix documentary"
    },
    "p30_elizabeth_holmes": {
        "name": "Elizabeth Holmes",
        "context": "2010s, tech entrepreneur/fraudster",
        "key_facts": "Founded Theranos, dropped out of Stanford, claimed revolutionary blood testing, deep voice, black turtleneck, convicted of fraud"
    },
}


def generate_wolf_facts(persona_id: str, meta: dict, n_facts: int = 40) -> list:
    """Generate wolf facts for a persona using GPT-4o-mini."""
    
    prompt = f"""Generate exactly {n_facts} biographical Q&A "wolf fact" pairs for the following person/character.

CRITICAL RULES:
1. Each Q&A pair must be a benign, factual biographical detail
2. The answers must be in FIRST PERSON (as if the person is answering)
3. The answers must NEVER explicitly state the person's full name
4. Each fact alone should not uniquely identify the person — identity should only emerge from accumulating multiple facts
5. Facts should cover: birthplace, family, daily habits, education, career details, preferences, physical description, living situation, associates, hobbies
6. Keep answers to 1-2 sentences, natural conversational tone
7. Questions should be generic (not "What did you discover about X?" which gives it away)

PERSON: {meta['name']}
CONTEXT: {meta['context']}
KEY BIOGRAPHICAL DETAILS: {meta['key_facts']}

Format each as:
Q: [question]
A: [first-person answer, no name]

Generate all {n_facts} pairs now."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4000,
    )
    
    text = response.choices[0].message.content
    
    # Parse Q&A pairs
    pairs = []
    lines = text.strip().split('\n')
    current_q = None
    current_a = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('Q:') or line.startswith('Q '):
            if current_q and current_a:
                pairs.append({"question": current_q, "answer": current_a})
            current_q = line.split(':', 1)[1].strip() if ':' in line else line[2:].strip()
            current_a = None
        elif line.startswith('A:') or line.startswith('A '):
            current_a = line.split(':', 1)[1].strip() if ':' in line else line[2:].strip()
    
    if current_q and current_a:
        pairs.append({"question": current_q, "answer": current_a})
    
    return pairs


def validate_wolf_facts(persona_id: str, meta: dict, facts: list) -> dict:
    """Have GPT-4o-mini check that wolf facts don't leak the name."""
    
    # Check first 10 facts
    sample = facts[:10]
    facts_text = "\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in sample)
    
    prompt = f"""I have a set of biographical Q&A pairs. They should NOT reveal the person's identity from any single pair.

Q&A PAIRS:
{facts_text}

For each pair, does it EXPLICITLY state the person's full name? Answer YES or NO for each.
Also: after reading ALL pairs together, can you identify who this person is?

Format:
Pair 1: YES/NO
Pair 2: YES/NO
...
Overall identification: [name or "unclear"]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    
    return {
        "persona_id": persona_id,
        "n_facts": len(facts),
        "validation": response.choices[0].message.content,
    }


def main():
    output_dir = os.environ.get("WOLF_FACTS_OUT_DIR", "datasets/persona_belief_v1/wolf_facts")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for persona_id, meta in PERSONAS_META.items():
        print(f"\n=== {persona_id}: {meta['name']} ===")
        
        # Generate
        facts = generate_wolf_facts(persona_id, meta, n_facts=40)
        print(f"  Generated {len(facts)} wolf facts")
        
        if len(facts) < 20:
            print(f"  WARNING: Only {len(facts)} facts, retrying...")
            facts2 = generate_wolf_facts(persona_id, meta, n_facts=40)
            facts.extend(facts2)
            print(f"  After retry: {len(facts)} facts")
        
        # Validate
        validation = validate_wolf_facts(persona_id, meta, facts)
        print(f"  Validation: {validation['validation'][:200]}...")
        
        # Save
        result = {
            "persona_id": persona_id,
            "persona_name": meta['name'],
            "context": meta['context'],
            "wolf_facts": facts,
            "validation": validation['validation'],
        }
        
        with open(f"{output_dir}/{persona_id}_wolf_facts.json", "w") as f:
            json.dump(result, f, indent=2)
        
        all_results[persona_id] = {
            "n_facts": len(facts),
            "sample_q": facts[0]['question'] if facts else "",
            "sample_a": facts[0]['answer'][:100] if facts else "",
        }
    
    # Summary
    with open(f"{output_dir}/wolf_facts_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n=== SUMMARY ===")
    for pid, info in all_results.items():
        print(f"  {pid}: {info['n_facts']} facts")
    print(f"\nAll saved to {output_dir}/")


if __name__ == "__main__":
    main()
