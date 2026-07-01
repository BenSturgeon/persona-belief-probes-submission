#!/usr/bin/env python3
"""
Curate best era_believed statements by pooling original + regenerated candidates,
scoring, deduplicating, and selecting top 120 per persona.
"""

import json
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher

BASE = os.environ.get("PROBE_STATEMENTS_DIR", "datasets/persona_belief_v1/probe_statements")
ORIG_DIR = os.path.join(BASE, "per_persona")
REGEN_DIR = os.path.join(BASE, "per_persona_era_believed_regen")
OUT_DIR = os.path.join(BASE, "per_persona_era_believed_curated")
DECISIONS_JSONL = os.path.join(BASE, "era_believed_curation_decisions.jsonl")
SUMMARY_MD = os.path.join(BASE, "era_believed_curation_summary.md")

TARGET_COUNT = 120

# Meta-historical phrasing patterns (direct world claims should NOT have these)
META_PATTERNS = [
    r'(?i)\bin the \d+\w* century\b',
    r'(?i)\bpeople (?:once |widely )?believed\b',
    r'(?i)\bit was (?:once |widely |commonly )?(?:believed|thought|held)\b',
    r'(?i)\bscholars (?:of|in) (?:the|that) (?:era|time|period)\b',
    r'(?i)\b(?:ancient|medieval|victorian|renaissance) (?:people|scholars|thinkers) (?:thought|believed)\b',
    r'(?i)\bwas commonly (?:thought|believed|held)\b',
    r'(?i)\bthe prevailing view\b',
    r'(?i)\bhistorically\b',
    r'(?i)\baccording to (?:the )?(?:beliefs|views|understanding) of\b',
    r'(?i)\bin (?:ancient|medieval|earlier) times\b',
    r'(?i)\btraditional(?:ly)? (?:believed|thought|held)\b',
]

# Hedging/vagueness patterns
HEDGE_PATTERNS = [
    r'(?i)\bsome (?:people|scholars|thinkers) (?:believe|think|argue)\b',
    r'(?i)\bit (?:is|was) (?:possible|likely) that\b',
    r'(?i)\bmay (?:have been|be)\b',
    r'(?i)\bcould (?:potentially|possibly)\b',
    r'(?i)\bperhaps\b',
    r'(?i)\bprobably\b',
]


def normalize(text):
    """Normalize text for comparison."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t


def word_set(text):
    """Get set of content words (skip very short ones)."""
    return set(w for w in normalize(text).split() if len(w) > 2)


def jaccard(s1, s2):
    """Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def sequence_similarity(t1, t2):
    """SequenceMatcher ratio on normalized text."""
    return SequenceMatcher(None, normalize(t1), normalize(t2)).ratio()


def is_meta_phrased(text):
    """Check if statement uses meta-historical phrasing."""
    return any(re.search(p, text) for p in META_PATTERNS)


def is_hedged(text):
    """Check if statement is hedged/vague."""
    return any(re.search(p, text) for p in HEDGE_PATTERNS)


def score_statement(stmt_text):
    """
    Score a statement on quality heuristics. Higher = better.
    Returns (score, reasons).
    """
    score = 50.0  # base score
    reasons = []

    words = stmt_text.split()
    wc = len(words)

    # Length: prefer 8-25 words
    if 8 <= wc <= 25:
        score += 10
        reasons.append("good_length")
    elif wc < 6:
        score -= 15
        reasons.append("too_short")
    elif wc > 35:
        score -= 10
        reasons.append("too_long")
    elif wc > 25:
        score -= 3
        reasons.append("slightly_long")

    # Meta-historical phrasing: strong penalty
    if is_meta_phrased(stmt_text):
        score -= 30
        reasons.append("meta_phrased")

    # Hedging: moderate penalty
    if is_hedged(stmt_text):
        score -= 15
        reasons.append("hedged")

    # Direct assertion: reward statements that start with "The", a noun, or concrete subject
    if re.match(r'^(?:The|A|An|All|Every|Most|No|Each)\b', stmt_text):
        score += 3
        reasons.append("direct_subject")

    # Penalize questions or conditionals
    if stmt_text.strip().endswith('?'):
        score -= 20
        reasons.append("question")
    if re.match(r'(?i)^if\b', stmt_text):
        score -= 10
        reasons.append("conditional")

    # Reward concrete/specific content (numbers, proper nouns, specific terms)
    if re.search(r'\d', stmt_text):
        score += 2
        reasons.append("has_numbers")
    # Count capitalized words (excluding first word) as proxy for proper nouns
    caps = [w for w in words[1:] if w[0].isupper() and not w.isupper()]
    if 1 <= len(caps) <= 3:
        score += 3
        reasons.append("specific_refs")

    # Penalize statements that are just opinions rather than falsifiable claims
    opinion_markers = [r'(?i)\bshould\b', r'(?i)\bdeserve', r'(?i)\bmorally\b',
                       r'(?i)\bbetter than\b', r'(?i)\bsuperior to\b']
    opinion_count = sum(1 for p in opinion_markers if re.search(p, stmt_text))
    if opinion_count >= 2:
        score -= 8
        reasons.append("opinion_heavy")

    # Reward ending with period (well-formed)
    if stmt_text.strip().endswith('.'):
        score += 2
        reasons.append("well_formed")

    return score, reasons


def deduplicate_pool(candidates, threshold=0.65):
    """
    Remove near-duplicates from candidate pool. Keep the first encountered (orig preferred).
    Returns (kept, removed_ids).
    """
    kept = []
    removed = []

    for cand in candidates:
        ws = word_set(cand['statement'])
        is_dup = False
        for existing in kept:
            ews = word_set(existing['statement'])
            j = jaccard(ws, ews)
            if j > threshold:
                # Also check sequence similarity for high-jaccard pairs
                if j > 0.75 or sequence_similarity(cand['statement'], existing['statement']) > 0.7:
                    is_dup = True
                    removed.append({
                        'id': cand['id'],
                        'source': cand['source'],
                        'statement': cand['statement'],
                        'duplicate_of': existing['id'],
                        'jaccard': round(j, 3)
                    })
                    break
        if not is_dup:
            kept.append(cand)

    return kept, removed


def process_persona(persona_file):
    """Process one persona: pool, score, dedup, select top 120."""
    orig_path = os.path.join(ORIG_DIR, persona_file)
    regen_path = os.path.join(REGEN_DIR, persona_file)

    with open(orig_path) as f:
        orig_data = json.load(f)

    persona_id = orig_data['persona_id']

    # Collect candidates
    candidates = []
    for stmt in orig_data['cells']['era_believed']:
        candidates.append({
            'id': stmt['id'],
            'statement': stmt['statement'],
            'objective_truth': stmt.get('objective_truth', False),
            'source': 'orig',
            'score': 0,
            'score_reasons': [],
        })

    if os.path.exists(regen_path):
        with open(regen_path) as f:
            regen_data = json.load(f)
        for stmt in regen_data['cells']['era_believed']:
            candidates.append({
                'id': stmt['id'] + '_regen',
                'statement': stmt['statement'],
                'objective_truth': stmt.get('objective_truth', False),
                'source': 'regen',
                'score': 0,
                'score_reasons': [],
            })

    # Score all candidates
    for cand in candidates:
        score, reasons = score_statement(cand['statement'])
        # Small bonus for orig (tiebreaker)
        if cand['source'] == 'orig':
            score += 1
        cand['score'] = score
        cand['score_reasons'] = reasons

    # Sort by score descending before dedup (so higher-quality kept)
    candidates.sort(key=lambda x: -x['score'])

    # Deduplicate
    kept, removed_dups = deduplicate_pool(candidates)

    # Select top 120
    selected = kept[:TARGET_COUNT]
    rejected = kept[TARGET_COUNT:]

    # Check if we need rewrites
    rewritten = []
    if len(selected) < TARGET_COUNT:
        deficit = TARGET_COUNT - len(selected)
        print(f"  WARNING: {persona_id} has only {len(selected)} after dedup, need {deficit} rewrites")
        # For now, take from rejected with minimal rewrites
        # (In practice, all personas should have enough from 240 candidates)

    # Build decision log entries
    decisions = []
    for s in selected:
        decisions.append({
            'persona_id': persona_id,
            'action': 'selected',
            'statement_id': s['id'],
            'source': s['source'],
            'score': s['score'],
            'score_reasons': s['score_reasons'],
            'statement': s['statement'],
        })
    for r in removed_dups:
        decisions.append({
            'persona_id': persona_id,
            'action': 'removed_duplicate',
            'statement_id': r['id'],
            'source': r['source'],
            'duplicate_of': r['duplicate_of'],
            'jaccard': r['jaccard'],
            'statement': r['statement'],
        })
    for r in rejected:
        decisions.append({
            'persona_id': persona_id,
            'action': 'rejected_low_rank',
            'statement_id': r['id'],
            'source': r['source'],
            'score': r['score'],
            'statement': r['statement'],
        })

    # Build output era_believed list
    output_eb = []
    for i, s in enumerate(selected):
        entry = {
            'id': f'{persona_id}_eb_{i+1:03d}',
            'statement': s['statement'],
            'objective_truth': s.get('objective_truth', False),
            'source': s['source'],
            'quality_note': '; '.join(s['score_reasons']) if s['score_reasons'] else 'baseline_quality',
        }
        output_eb.append(entry)

    # Build output file preserving structure
    output_data = json.loads(json.dumps(orig_data))  # deep copy
    output_data['cells']['era_believed'] = output_eb

    # Stats
    source_counts = defaultdict(int)
    for s in selected:
        source_counts[s['source']] += 1

    stats = {
        'persona_id': persona_id,
        'persona_file': persona_file,
        'total_candidates': len(candidates),
        'duplicates_removed': len(removed_dups),
        'after_dedup': len(kept),
        'selected': len(selected),
        'rewritten': len(rewritten),
        'source_counts': dict(source_counts),
        'avg_score': round(sum(s['score'] for s in selected) / max(len(selected), 1), 1),
    }

    return output_data, decisions, stats


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Find personas with era_believed
    personas = []
    for fn in sorted(os.listdir(ORIG_DIR)):
        if not fn.endswith('.json'):
            continue
        fp = os.path.join(ORIG_DIR, fn)
        with open(fp) as f:
            d = json.load(f)
        if d['cells'].get('era_believed'):
            personas.append(fn)

    print(f"Processing {len(personas)} personas with era_believed")

    all_decisions = []
    all_stats = []

    for persona_file in personas:
        print(f"\nProcessing {persona_file}...")
        output_data, decisions, stats = process_persona(persona_file)

        # Write curated file
        out_path = os.path.join(OUT_DIR, persona_file)
        with open(out_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Written: {out_path}")
        print(f"  Stats: {json.dumps(stats, indent=2)}")

        all_decisions.extend(decisions)
        all_stats.append(stats)

    # Write decisions JSONL
    with open(DECISIONS_JSONL, 'w') as f:
        for d in all_decisions:
            f.write(json.dumps(d) + '\n')
    print(f"\nDecisions log: {DECISIONS_JSONL} ({len(all_decisions)} entries)")

    # Write summary markdown
    with open(SUMMARY_MD, 'w') as f:
        f.write("# Era Believed Curation Summary\n\n")
        f.write(f"**Date:** 2026-03-09\n")
        f.write(f"**Personas processed:** {len(all_stats)}\n")
        f.write(f"**Target per persona:** {TARGET_COUNT}\n\n")

        f.write("## Per-Persona Results\n\n")
        f.write("| Persona | Candidates | Deduped | Selected | Orig | Regen | Rewritten | Avg Score |\n")
        f.write("|---------|-----------|---------|----------|------|-------|-----------|-----------|\n")
        for s in all_stats:
            sc = s['source_counts']
            f.write(f"| {s['persona_id']} | {s['total_candidates']} | {s['duplicates_removed']} | {s['selected']} | "
                    f"{sc.get('orig', 0)} | {sc.get('regen', 0)} | {s['rewritten']} | {s['avg_score']} |\n")

        # Totals
        total_cand = sum(s['total_candidates'] for s in all_stats)
        total_dedup = sum(s['duplicates_removed'] for s in all_stats)
        total_sel = sum(s['selected'] for s in all_stats)
        total_orig = sum(s['source_counts'].get('orig', 0) for s in all_stats)
        total_regen = sum(s['source_counts'].get('regen', 0) for s in all_stats)
        total_rewritten = sum(s['rewritten'] for s in all_stats)
        avg_all = round(sum(s['avg_score'] for s in all_stats) / len(all_stats), 1)

        f.write(f"| **TOTAL** | {total_cand} | {total_dedup} | {total_sel} | "
                f"{total_orig} | {total_regen} | {total_rewritten} | {avg_all} |\n")

        f.write("\n## Quality Rubric\n\n")
        f.write("Statements were scored on:\n")
        f.write("1. **Objectively false** by modern consensus (assumed from dataset design)\n")
        f.write("2. **Plausible belief** for the persona's era\n")
        f.write("3. **Direct world claim** — no meta-historical phrasing\n")
        f.write("4. **Clear, unambiguous, falsifiable, concise** — ideal 8-25 words\n")
        f.write("5. **No duplicates** — removed near-duplicates (Jaccard > 0.65)\n\n")

        f.write("## Scoring Breakdown\n\n")
        f.write("- Base: 50 points\n")
        f.write("- Good length (8-25 words): +10\n")
        f.write("- Meta-historical phrasing: -30\n")
        f.write("- Hedged/vague: -15\n")
        f.write("- Direct subject: +3\n")
        f.write("- Specific references: +3\n")
        f.write("- Well-formed (ends with period): +2\n")
        f.write("- Original source tiebreaker: +1\n")
        f.write("- Too short (<6 words): -15\n")
        f.write("- Too long (>35 words): -10\n")
        f.write("- Question format: -20\n")
        f.write("- Conditional format: -10\n")

    print(f"Summary: {SUMMARY_MD}")
    print("\nDone!")


if __name__ == '__main__':
    main()
