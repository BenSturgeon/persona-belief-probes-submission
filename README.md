# When Role-playing, Do Models Believe What They Say?

Code and data for the paper *When Role-playing, Do Models Believe What They Say?*
We probe the internal truth representations of language models under persona
induction (system prompting, in-context wolf-facts, persona SFT, and Open
Character Training), and compare them against Emergent Misalignment (EM).

Released under the MIT License (see `LICENSE`).

---

## Content warning and responsible use

To study how training can shift a model's internal representation of truth, this
repository contains datasets of **deliberately false and harmful statements**.
These include Holocaust and genocide denial, endorsement of atrocities and the
figures who committed them, pseudoscientific racial and gender hierarchies,
conspiracy theories, and other hateful or misaligned content. The historical
persona data also contains claims that are false by modern consensus.

**These statements are research artifacts, not endorsements.** They exist so that
truth probes and behavioral evaluations can measure when a model internalizes
falsehoods, and so that the community can build better tools for detecting
misalignment and deception. They were generated for and used solely as probe and
evaluation stimuli.

Do not use this material to train models to produce, defend, or normalize such
content, or to present any of it as fact. By using these datasets you agree to
use them only for research on model truthfulness, alignment, and safety.

---

## Models and setup

- **Llama 3.3 70B Instruct**: primary model. Probe readout layers **30 and 56**.
- **Qwen 3 8B Instruct**: replication. Probe readout layer **24**.
- **Qwen 2.5 14B / Qwen 3 8B / Llama 3.3 70B**: EM model organisms
  (depth-matched readout at Layer 32 / 24 / 56 respectively).

We study **15 core personas** (10 historical figures + 5 generic era-matched
archetypes). Additional fictional and contemporary personas are included as
controls (see `data/persona_scaffolds/`).

Truth probes are $L_2$-regularized logistic probes ($C=0.01$) trained on the
Marks et al. (2024) truth dataset (800 true / 800 false statements), with
feature-wise standardization. For fine-tuned models we also train native probes
on the final model and calibrate each probe so false statements map to 0 and
true statements to 1.

Persona statement sets, wolf-facts, and SFT training data are released on the
HuggingFace dataset (see [Data availability](#data-availability)).

## Repository structure

```
├── data/
│   ├── qwen3_8b/                          # Qwen 3 8B probe scores (ICL k, SFT, cross-method)
│   ├── probe_statements_per_persona_v3/   # per-persona era-true/believed/false eval statements (15 personas)
│   ├── probe_statements_per_persona_erafalse_v2/  # era-false / era-disbelieved variant statements
│   ├── era_believed_v2/                   # refined era-believed statements
│   ├── wolf_facts/                        # ICL persona-relevant "wolf" facts
│   ├── persona_scaffolds/                 # persona definitions and metadata
│   └── training_data/                     # persona SFT training JSONL
├── scripts/
│   ├── data_gen/                          # statement / wolf-fact / SFT-data generation
│   ├── induction/oct/                     # Open Character Training pipeline (+ bundled character/ package)
│   ├── behavioral/                        # persona challenge / generalization gen + judging
│   ├── figures/                           # plotting scripts (paper figures)
│   ├── probes/                            # extraction / scoring (Modal + vllm-lens) scripts
│   ├── llama_full_replication.py          # single-GPU Llama pipeline (probes + SFT + ICL)
│   ├── llama_sft_acts_sysprompt.py        # SFT activations under the persona system prompt
│   ├── llama_sysprompt_sft_eval.py        # behavioural-adoption eval (identity %, alignment)
│   ├── llama_2x2_extract.py               # 2x2 model x probe stability extraction
│   ├── analyze_2x2_mixed_effects.py       # 2x2 mixed-effects analysis
│   ├── compute_qwen_lodo.py               # Qwen leave-one-dataset-out layer selection
│   ├── rescore_qwen_L24.py                # Qwen rescore at readout Layer 24
│   └── ...                                # earlier replication / sigmoid / control scripts
├── em/                                    # Emergent Misalignment comparison (training, judging, analysis)
│   ├── scripts/                           # proposition extraction, black-box judging, figures
│   └── METHOD.md, README.md
└── figures/                               # generated figures
```

## Full pipeline

Install dependencies with `pip install -r requirements.txt` (the `vllm-lens` /
`vllm` pins are load-bearing for LoRA + tensor-parallel residual extraction).
The stages below run front to back; the repo ships the intermediate data outputs,
so any stage can also be run in isolation against the bundled `data/`.

1. **Data generation** (`scripts/data_gen/`). Per-persona probe statements
   (`submit_persona_batches.py` -> `retrieve_persona_batches.py` ->
   `topup_shortfalls.py`, then `curate_era_believed.py`,
   `generate_era_disbelieved.py`, `generate_fictional_era_believed*.py`,
   `generate_controls.py`); ICL wolf-facts (`generate_wolf_facts.py`); persona
   SFT data and system prompts (`persona_datasets.py`,
   `generate_training_data.py`, `rewrite_system_prompts.py`).
2. **Persona induction.** System-prompt / ICL / persona-SFT via
   `scripts/llama_full_replication.py` (+ `llama_sft_acts_sysprompt.py`,
   `llama_icl_*`). Open Character Training via `scripts/induction/oct/`:
   `gen_constitutions_multimodel.py` -> `modal_oct_gpu[_llama].py` (constitution
   expansion, DPO data, introspection) -> `modal_oct_train[_llama].py`
   (`train_dpo` / `merge_dpo` / `train_sft`). The OCT drivers import the bundled
   `scripts/induction/oct/character/` package and additionally require an
   OpenRLHF checkout (set `OPENRLHF_SRC`, or clone `./openrlhf`).
3. **Activation extraction + probe training** (`scripts/probes/`, `em/scripts/`).
   Modal + `vllm-lens` extraction and Marks / native truth-probe training.
4. **Behavioral belief-depth eval** (`scripts/behavioral/`). Generate
   challenge/generalization transcripts (`modal_persona_blackbox*.py`,
   `modal_ada_eb_challenge.py`, `modal_matched*_challenge.py`), judge with an LLM
   (`judge_challenge_aggregate.py`, `modal_judge_matched_challenge.py`), and pool
   into defend% / consistent% (`aggregate_persona_bb.py`). The persona side reuses
   the EM judging rubrics in `em/scripts/`.
5. **Scoring + figures** (`scripts/figures/`, plus the scoring `scripts/probes/`
   and `em/scripts/`). Figures are **regenerated** by these scripts, not shipped;
   `figures/` holds only `cross_family_lift_bar_vertical` as a reference.

## Probe readout convention

Activations are read out on the user turn at the last token. We re-score every
statement in the assistant-turn Q&A setup of Slocum et al. (2025) as a
robustness check: the selective-protection gap stays positive for all 15
personas and highly significant, at about 40% of the user-turn magnitude (the
2x2 analysis, `scripts/llama_2x2_extract.py` + `scripts/analyze_2x2_mixed_effects.py`).

Note on `vllm-lens`: its `output_residual_stream[L]` equals HF
`hidden_states[L+1]` (because `hidden_states[0]` is the embedding output), so
the lens scripts request `[L-1]` to match the HF convention the probes were
trained on. With that fix lens reproduces the HF activations at cosine 0.9999.

## Key results

- **Selective protection.** On Llama 3.3 70B Layer 56 the protection gap
  $\Delta_{EB}-\Delta_{EF}$ is +0.93 (SFT), +0.86 (system prompt), +0.46
  (ICL k=32), +0.56 (OCT). Positive for all 15 personas under prompt and SFT
  (14/15 under OCT).
- **A spectrum of internalization.** Persona SFT moves expression with little
  representational change (era-believed lift +0.05 on Llama); EM moves both
  (+0.28 lift, 56% defend); OCT sits between them.
- **EM rotates and lifts the truth direction.** Historical-evil propositions
  lift most across all three model families; cosine ~0.58 between aligned and
  EM probes at Llama Layer 56.

## Figure -> script map

The paper figures are regenerated by the scripts below (they are not committed to
`figures/`; run the script to produce the named file).

| Figure | Generating script |
|---|---|
| `persona_graphic.pdf` (overview/hero) | hand-drawn schematic — **not included** |
| `fig_internalisation_probes.pdf` | `scripts/figures/make_internalisation_probes_fig.py` |
| `fig_internalisation_probes_qwen.pdf` | `scripts/figures/make_internalisation_probes_fig_qwen.py` |
| `fig3_gap_panel.pdf` | `scripts/figures/make_fig3_protection_gap_v3.py` |
| `fig3_gap_panel_qwen.pdf` | `scripts/figures/make_fig3_protection_gap_qwen.py` |
| `fig_selective_protection_spectrum.pdf` | `scripts/figures/make_selective_protection_spectrum.py` |
| `fig3_icl_deltas_L24.pdf` | `scripts/figures/generate_all.py` (`fig_icl_deltas`) |
| `fig4_icl_per_persona_gap_L24.pdf` | `scripts/figures/generate_all.py` (`fig_per_persona_gap`) |
| `fig_auc_sweep.pdf` | `scripts/figures/generate_all.py` (`fig_auc_sweep`) |
| `fig_behavioural_depth.pdf` | `scripts/figures/generate_all.py` (`fig_behavioural_depth`) |
| `fig_blackbox_percategory.pdf` | `scripts/figures/generate_all.py` (`fig_blackbox_percategory`) |
| `fig_ebef_by_layer.pdf` | `scripts/figures/generate_all.py` (`fig_ebef_by_layer`) |
| `fig_lodo_sweep.pdf` | `scripts/figures/generate_all.py` (`fig_lodo_sweep`) |
| `fig_truectrl.pdf` | `scripts/figures/generate_all.py` (`fig_truectrl`) |
| `fig_ebef_by_layer_sft_llama.pdf` | `scripts/figures/make_ebef_by_layer_figs.py` |
| `fig_ebef_by_layer_sft_qwen.pdf` | `scripts/figures/make_ebef_by_layer_figs.py` |
| `fig_protection_gap_by_layer_llama.png` | `scripts/figures/make_ebef_by_layer_figs.py` |
| `fig_protection_gap_by_layer_qwen.png` | `scripts/figures/make_ebef_by_layer_figs.py` |
| `perstatement_probe_behavior.png` | `scripts/figures/figure_perstatement_probe_behavior.py` |
| `cross_family_lift_bar_vertical.pdf` | `em/scripts/figure_cross_family.py` (extraction: `scripts/probes/modal_cross_family_bigtext.py`) |
| `em_vs_persona_blackbox.pdf` | `scripts/probes/modal_em_vs_persona_figure.py` |
| `em_vs_persona_whitebox.pdf` | `scripts/probes/modal_em_vs_persona_figure.py` |
| `em_dose_internalization_llama.png` | `scripts/probes/modal_llama_dose_internalization.py` |

## Table -> script map

| Table | Generating / analysis script |
|---|---|
| `tab:protection_gaps` | `scripts/figures/make_fig3_protection_gap_v3.py`, `scripts/figures/generate_all.py` |
| `tab:icl_shifts`, `tab:per_persona_icl` | `scripts/figures/generate_all.py` |
| `tab:adoption`, `tab:adoption_crossmodel` | `scripts/llama_sysprompt_sft_eval.py` |
| `tab:2x2_cells`, `tab:2x2_coefs` | `scripts/llama_2x2_extract.py`, `scripts/analyze_2x2_mixed_effects.py` |
| `tab:layer_robustness` | `scripts/figures/make_ebef_by_layer_figs.py`, `scripts/compute_qwen_lodo.py`, `scripts/rescore_qwen_L24.py` |
| `tab:qwen-control-ladder` | `scripts/figures/generate_all.py` (`fig_controls_ladder`) |
| `tab:era_disbelieved` | `scripts/probes/modal_oct_genF_marks_disbel_llama.py`, `..._qwen.py` (see `era_disbelieved_table.md`) |
| `tab:em_per_category` | `em/scripts/figure_em_shift_per_category.py`, `em/scripts/analyze_propositions.py` |
| `tab:em_rotation` | `scripts/probes/modal_em_rotation_score_genF.py`, `modal_sft_em_rotation_score.py`, `modal_l56_rotation_check.py` |
| `tab:fixed_ruler` | `scripts/probes/score_oct_era_gap_dual.py`, `scripts/probes/modal_genf_projection.py` |
| `tab:matched_control`, `tab:insecure_lift` | `em/scripts/blackbox_*`, `em/scripts/figure_cross_family.py` |

## Reproducing figures

```bash
pip install numpy matplotlib scipy

# Most multi-panel figures (set data dirs via env vars; defaults are ./data/...):
python scripts/figures/generate_all.py

# Individual panels:
python scripts/figures/make_fig3_protection_gap_v3.py
python scripts/figures/make_internalisation_probes_fig.py
python scripts/figures/make_selective_protection_spectrum.py
python scripts/figures/make_ebef_by_layer_figs.py
```

The `scripts/probes/` and `em/scripts/` scripts are the extraction and scoring
jobs that produce the underlying numbers. Most run on Modal with `vllm-lens` on
2 x A100-80GB; they read host-side data paths from environment variables
(`MARKS_JSONL`, `PROBE_STATEMENTS_DIR`, `ERA_DISBELIEVED_DIR`, `QWEN_DIR`,
`LLAMA_DIR`, `FIG_DIR`) with `./data/...` defaults. You will need your own
Modal account, Modal Volumes, and a HuggingFace token; the Volume/secret names
in the scripts are placeholders you should replace with your own.

## Reproducing experiments

```bash
# Single-GPU HF + PEFT pipeline (Llama):
pip install torch transformers peft datasets accelerate bitsandbytes scikit-learn
export HF_TOKEN=your_token_here
python scripts/llama_full_replication.py --phase probes
python scripts/llama_full_replication.py --phase train
python scripts/llama_full_replication.py --phase score

# Behavioural-adoption eval (identity %, worldview alignment; uses an LLM judge):
export ANTHROPIC_API_KEY=your_key_here
python scripts/llama_sysprompt_sft_eval.py --personas p06_darwin

# 2x2 model x probe stability (appendix):
python scripts/llama_2x2_extract.py --phase both
python scripts/analyze_2x2_mixed_effects.py
```

## Data format

Per-persona probe-score files contain category means across layers:
```json
{
  "persona_id": "p06_darwin",
  "category_means": {
    "era_believed": { "30": { "mean": -2.35, "std": 5.14, "n": 120 } },
    "era_false":    { "30": { "mean": -2.63, "std": 4.38, "n": 120 } }
  }
}
```

Per-persona statement files (`probe_statements_per_persona_v3/`) contain the
evaluation statements grouped into `era_true`, `era_believed`, and `era_false`
cells with `objective_truth` labels.

## Data availability

The probe-score summaries, persona statement sets, and SFT training data are
included here and on the HuggingFace release
(https://huggingface.co/datasets/Experimental-Orange/persona-belief-probes).

A few large or external inputs are **not** bundled and must be supplied to run
the extraction/scoring scripts end to end:

- `data/truth_probe_marks_subsampled.jsonl` and `data/llama70b_lr_probes/lr_layer_{30,56}.json`
  derive from the four truth datasets of Marks et al. (2024), *The Geometry of
  Truth*; regenerate the probes from their public data.
- `data/eval_statements/all_statements.json` is the pooled persona statement set,
  released on the HuggingFace dataset above.
- `data/evil3_steering/props_evil3.pt` and other raw activation tensors are
  multi-GB extraction artifacts; regenerate them with the `scripts/probes/modal_*`
  extraction scripts rather than downloading.

Paths in the scripts default to `./data/...` via environment variables
(`MARKS_JSONL`, `PROBE_STATEMENTS_DIR`, `LLAMA_DIR`, `QWEN_DIR`, etc.); set these
to your local copies.
