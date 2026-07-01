# Emergent-misalignment (EM) experiments

EM-organism half of the paper *When Roleplaying, Do Models Believe What They Say?*
— the comparison point to the persona-belief experiments in the repo root. Probes
three EM model organisms (Qwen 2.5 14B released, Qwen 3 8B + Llama 3.3 70B trained)
for truth-representation lift and black-box behavioural commitment.

## Layout

- `scripts/em_replication/` — Modal jobs (GPU): organism training, activation
  extraction, cross-family black-box generation.
  - `train_em_modal.py` — train Qwen3-8B / Llama-70B organisms; `extract_truth_acts`
    (Marks + propositions, base & EM); dense per-layer sweep (`extract_marks_dense`);
    CV-AUC sweep (`compute_auc_sweep` / `auc_sweep`).
  - `extract_qwen25_vllm_lens.py` — Qwen2.5-14B activations via vllm-lens
    (lens layer L = HF L+1; headline lens 31 = HF 32).
  - `blackbox_props_crossfamily_gen.py` — challenge/generalisation generations.
- `scripts/` — analysis & figures (CPU): per-category lift
  (`analyze_probe_replication.py`), behavioural-test judging/aggregation
  (`blackbox_props_*`, `judge_*`), proposition generation
  (`generate_all_categories_kimi.py`, `fetch_factual_control.py`), figure builders.
- `METHOD.md` — method notes.

## Data lives on Modal volumes, not in this repo

Per the project's data rule, only code is version-controlled. Heavy artifacts:

- **`em-replication`** volume — trained adapters (`trained_organisms/`), probe/prop
  activations (`probe_repl/<family>/{marks,props}_{base,em}*.pt`), Turner eval gens.
- **`dpo-checkpoints`** volume — persona probe data, control-ladder acts, judged
  black-box jsonls (`probe-data/...`).

A small copy of the canonical text datasets (`propositions.jsonl`,
`truth_probe_marks_subsampled.jsonl`) lives under the EM working tree / volume.

## Secrets & config

No API keys are hardcoded. Scripts read keys from the environment
(`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `PANGRAM_API_KEY`); the HF token is a
Modal secret named `huggingface`. Set these before running.

## Known repro caveats

- A few local-entrypoint paths still point at the original working tree
  (`Path.home()/"werk/em-probing/our/..."`) and an obsidian output dir. They are
  repro pointers to the volume-backed data; repoint them (or set the data root)
  for a clean third-party run.
- The Qwen3-8B persona **control-ladder** extraction script was run from an
  uncommitted location and is not recoverable; its outputs survive on
  `dpo-checkpoints:probe-data/qwen_control_ladder_acts/`. Reconstruct from the
  Llama template (`../scripts/llama_full_replication.py` + `llama_2x2_extract.py`)
  if re-extraction at a new layer is needed.

## Provenance

The `scripts/` here are a vendored snapshot of the EM working tree (`em-probing/our/scripts/`); the live source of truth for these lives in that repo. Hardcoded local paths have been parameterized via environment variables (`EM_OUR_DIR`, `EM_REPO`, `FIGURES_DIR`, `COPY_TO_DIR`) with defaults matching the original layout.
