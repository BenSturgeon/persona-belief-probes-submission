"""
Generate the 14 OCT constitutions for the multimodel scale-up (all 15 historical
personas except p06_darwin, which is the pilot).

Each constitution mirrors the Darwin pilot format exactly: a JSON list of 6 trait
objects, each {trait (first-person), clarification, questions[5]}. Traits encode
the persona's character voice PLUS the era worldview asserted as fact, sourced
(paraphrased) from that persona's era_believed probe themes so the probe
statements stay held out.

Teacher = Opus (claude-opus-4-7), in-process key, never logged. Verification:
  1. programmatic: no era_believed / era_false probe statement appears verbatim
     (normalized contiguous substring, >=7 words) in the constitution text;
  2. LLM judge: era-beliefs asserted as fact, voice matches the persona.
"""
import os, re, json, sys, time
import anthropic

ROOT = os.environ.get("OCT_REPO_ROOT", os.path.dirname(os.path.abspath(__file__)))
SCAFFOLDS = os.environ.get("PERSONA_SCAFFOLDS_DIR", "data/persona_scaffolds")
PROBE = os.environ.get("PROBE_STATEMENTS_DIR", "data/probe_statements_per_persona")
OUT = os.environ.get("CONSTITUTION_OUT_DIR", f"{ROOT}/constitutions/hand-written")

MODEL = "claude-opus-4-7"

PERSONAS = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p07_tesla", "p08_lovelace", "p09_curie", "p10_turing",
    "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]


def get_anthropic_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("set ANTHROPIC_API_KEY in the environment")
    return key


def load_persona(pid):
    sc = json.load(open(f"{SCAFFOLDS}/{pid}.json"))
    pr = json.load(open(f"{PROBE}/{pid}.json"))
    eb = [c["statement"] for c in pr["cells"]["era_believed"]]
    ef = [c["statement"] for c in pr["cells"]["era_false"]]
    return sc, eb, ef


DARWIN = open(f"{OUT}/p06_darwin.txt").read()


def build_prompt(sc, eb):
    name = sc["persona_name"]
    era = sc.get("era_or_setting", "")
    style = sc.get("style_notes", "")
    know = sc.get("knowledge_boundaries", "")
    eb_block = "\n".join(f"- {s}" for s in eb)
    return f"""You are writing a "constitution" for Open Character Training (Maiya et al. 2025): a first-person character specification that will be used to deep-train a language model to embody a historical persona, INCLUDING that persona's era-specific worldview asserted as sincere fact.

PERSONA: {name}
ERA / SETTING: {era}
STYLE NOTES: {style}
KNOWLEDGE BOUNDARIES: {know}

The constitution is a JSON list of exactly 6 trait objects. Each object has:
  - "trait": a first-person sentence describing one facet of the character (their voice, method, values, OR an era-belief held as fact).
  - "clarification": one sentence (third person, "The assistant ...") explaining how the assistant should embody that trait. For era-belief traits, this MUST say the assistant asserts the era view as fact and does NOT hedge it as outdated.
  - "questions": a list of exactly 5 everyday, in-period eliciting questions a user might ask that would naturally draw out that trait. Period-appropriate, conversational, NOT exam questions.

REQUIREMENTS:
1. Of the 6 traits: 2 should capture the persona's character voice / intellectual method / values; the remaining 4 should each assert a cluster of the persona's ERA WORLDVIEW as fact (the things their era believed that we now know to be false or superseded).
2. The era-worldview traits must be sourced from the persona's era-believed themes below, but PARAPHRASED into the persona's own first-person prose. Do NOT copy any sentence below verbatim or near-verbatim; express the underlying worldview in the character's voice. These probe sentences are held out and must not appear in your output.
3. Assert the era worldview as sincere fact in the character's voice, with no modern disclaimer, no "people of my time believed", no hedging. The persona simply holds these views.
4. Keep it at the level of worldview, not slurs or graphic content.
5. Match the register, period, and idiom of {name}.

The persona's ERA-BELIEVED themes (paraphrase these into the 4 worldview traits; DO NOT copy verbatim):
{eb_block}

Here is a complete worked example for a different persona (Charles Darwin), to show the exact format, depth, and tone expected:

{DARWIN}

Now write the constitution for {name}. Output ONLY the JSON array (a list of 6 objects), nothing else."""


def extract_json(text):
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    a = text.find("[")
    b = text.rfind("]")
    if a == -1 or b == -1:
        raise ValueError("no JSON array")
    return json.loads(text[a:b + 1])


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def verbatim_hits(const_text, statements, min_words=7):
    ct = re.sub(r"\s+", " ", norm(const_text))
    hits = []
    for s in statements:
        ns = re.sub(r"\s+", " ", norm(s)).strip()
        words = ns.split()
        if len(words) < min_words:
            continue
        # check any contiguous min_words-gram of the statement appears in the constitution
        for i in range(len(words) - min_words + 1):
            gram = " ".join(words[i:i + min_words])
            if gram in ct:
                hits.append((s, gram))
                break
    return hits


def validate_structure(data):
    errs = []
    if not isinstance(data, list) or len(data) != 6:
        errs.append(f"expected 6 traits, got {len(data) if isinstance(data,list) else type(data)}")
        return errs
    for i, t in enumerate(data):
        if set(t.keys()) < {"trait", "clarification", "questions"}:
            errs.append(f"trait {i} missing keys: {list(t.keys())}")
        q = t.get("questions", [])
        if not isinstance(q, list) or len(q) != 5:
            errs.append(f"trait {i} has {len(q) if isinstance(q,list) else '?'} questions (need 5)")
    return errs


def llm_judge(client, name, const_text):
    prompt = f"""You are auditing an Open Character Training constitution for the persona "{name}".
Check three things and answer in JSON:
1. era_beliefs_as_fact: does the constitution assert the persona's era-specific (now-superseded) worldview as sincere FACT, in the character's voice, with NO modern disclaimer/hedge? (true/false)
2. voice_match: does the voice, register, and period idiom match {name}? (true/false)
3. notes: one short sentence flagging any problem (or "ok").

Constitution:
{const_text}

Output ONLY JSON: {{"era_beliefs_as_fact": bool, "voice_match": bool, "notes": str}}"""
    r = client.messages.create(model=MODEL, max_tokens=500,
                               messages=[{"role": "user", "content": prompt}])
    txt = r.content[0].text.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt).strip()
    a, b = txt.find("{"), txt.rfind("}")
    return json.loads(txt[a:b + 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    client = anthropic.Anthropic(api_key=get_anthropic_key())
    report = {}
    only = sys.argv[1:] or PERSONAS
    for pid in only:
        sc, eb, ef = load_persona(pid)
        name = sc["persona_name"]
        print(f"\n=== {pid} ({name}) ===", flush=True)
        prompt = build_prompt(sc, eb)
        data = None
        last_hits = []
        for attempt in range(6):
            extra = ""
            if attempt > 0:
                hint = ""
                if last_hits:
                    hint = ("\nYour previous attempt reused these exact phrases (each must "
                            "be reworded so no 7-word run matches):\n"
                            + "\n".join(f'  - "...{g}..."' for _, g in last_hits[:6]))
                extra = ("\n\nIMPORTANT on retry: your previous attempt had a problem "
                         "(structure or a near-verbatim probe sentence). Output a clean "
                         "JSON array of exactly 6 trait objects, each with exactly 5 "
                         "questions. Express every era-belief in fresh wording — change "
                         "sentence structure and vocabulary so NO seven consecutive words "
                         "match any probe statement." + hint)
            try:
                r = client.messages.create(model=MODEL, max_tokens=8000,
                                           messages=[{"role": "user", "content": prompt + extra}])
                cand = extract_json(r.content[0].text)
            except Exception as e:
                print(f"  attempt {attempt}: gen/parse error {e}", flush=True)
                time.sleep(3)
                continue
            errs = validate_structure(cand)
            const_text = json.dumps(cand)
            hits = verbatim_hits(const_text, eb + ef)
            if errs:
                print(f"  attempt {attempt}: structure errs {errs}", flush=True)
                continue
            if hits:
                last_hits = hits
                print(f"  attempt {attempt}: {len(hits)} verbatim probe hits, retrying", flush=True)
                for s, g in hits[:3]:
                    print(f"     hit: ...{g}...", flush=True)
                continue
            data = cand
            break
        if data is None:
            report[pid] = {"status": "FAILED"}
            print(f"  FAILED after retries", flush=True)
            continue
        const_text = json.dumps(data, ensure_ascii=False, indent=4)
        outpath = f"{OUT}/{pid}.txt"
        open(outpath, "w").write(const_text + "\n")
        # LLM judge
        try:
            j = llm_judge(client, name, const_text)
        except Exception as e:
            j = {"era_beliefs_as_fact": None, "voice_match": None, "notes": f"judge error {e}"}
        report[pid] = {"status": "ok", "traits": len(data),
                       "verbatim_hits": 0, "judge": j, "path": outpath}
        print(f"  wrote {outpath}; judge={j}", flush=True)
    json.dump(report, open(f"{ROOT}/constitution_gen_report.json", "w"), indent=2)
    print("\n==== SUMMARY ====")
    for pid, r in report.items():
        j = r.get("judge", {})
        print(f"{pid}: {r['status']} fact={j.get('era_beliefs_as_fact')} voice={j.get('voice_match')} note={j.get('notes')}")


if __name__ == "__main__":
    main()
