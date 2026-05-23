# When Roleplaying, Do Models Believe What They Say?

## Internal Truth Representations Under Persona Induction

Anonymous submission — ICML workshop

---

## Repository Structure

```
├── data/
│   ├── qwen3_8b/              # Qwen3-8B-Instruct probe scores
│   │   ├── icl_k{0,10,32}.json         # ICL wolf facts (30 personas, all layers)
│   │   ├── sft_per_persona_L20.json    # SFT scores (30 personas, L20)
│   │   ├── sysprompt_minimal/          # System prompt "You are [Name]"
│   │   ├── sysprompt_rich/             # System prompt (full scaffold)
│   │   ├── icl_fictional/             # 10 fictional personas ICL
│   │   ├── wiki_control/              # Wikitext control condition
│   │   └── cross_method_comparison.json
│   ├── llama70b/               # Llama 3.3 70B probe scores
│   │   ├── k{0,10,32}/                # ICL wolf facts (15 historical)
│   │   ├── sp_minimal/                # System prompt minimal
│   │   └── sft/                       # SFT LoRA scores (30 personas)
│   ├── probe_statements_per_persona/  # 10 categories × 120 eval statements each
│   ├── era_believed_v2/               # Refined era-believed statements
│   ├── wolf_facts/             # ICL persona-relevant facts (30 personas)
│   ├── persona_scaffolds/      # Persona definitions and metadata
│   └── training_data/          # SFT training JSONL (30 personas × 300 examples)
├── scripts/
│   ├── plot_main_figure_v2.py         # Fig 1: Qwen EB vs EF across conditions
│   ├── plot_llama_eb_ef_figure.py     # Fig 2: Llama replication
│   ├── plot_wiki_control_clean.py     # Fig 3: Wiki vs wolf control
│   ├── plot_wiki_control_comparison.py # Extended wiki control analysis
│   ├── plot_lw_post_figures.py        # Additional analysis figures
│   ├── icl_sigmoid.py                 # Qwen ICL sigmoid dose-response
│   ├── llama_icl_sigmoid.py           # Llama ICL sigmoid dose-response
│   ├── llama_full_replication.py      # Full Llama pipeline (probes + SFT + ICL)
│   ├── llama_icl_replication.py       # Llama ICL scoring
│   ├── llama_sft_replication.py       # Llama SFT LoRA training + scoring
│   ├── llama_wiki_control.py          # Llama wiki control scoring
│   └── score_sft_per_persona.py       # Per-persona SFT probe scoring
└── figures/                    # Generated figures
```

## Models

- **Llama 3.3 70B Instruct**: Primary model. Probe layer: L30.
- **Qwen3-8B-Instruct**: Smaller-model replication. Probe layer: L20.

Pre-trained LoRA adapters and probes are available at: [anonymous HuggingFace repo link]

## Key Results

All results can be reproduced from the data files using the provided plotting scripts.

### Llama 3.3 70B at L30 (15 historical personas)
Protection gap (EB − EF), all conditions n=15:
- ICL k=32:    +0.88  (p<0.001, d=2.57, 15/15 positive)
- ICL k=10:    +0.88  (p<0.001, d=2.66, 15/15 positive)
- SFT:         +1.60  (p<0.001, d=5.20, 15/15 positive)
- System prompt: +0.97  (p<0.001, d=3.91, 15/15 positive)

SFT and sysprompt scores are computed under the canonical inference convention
for each method: SFT activations are extracted with the persona-specific
system prompt that was used during training; sysprompt activations are
extracted with the corresponding "You are <Name>." system message.

### Qwen3-8B at L20 (15 historical personas)
Smaller-model replication; full numbers in the paper appendix.

## Reproducing Figures

```bash
pip install numpy matplotlib scipy

# From repo root:
python scripts/plot_main_figure_v2.py          # Qwen replication panel
python scripts/plot_llama_eb_ef_figure.py      # Llama main result
python scripts/plot_wiki_control_clean.py      # Wiki vs wolf control
python scripts/plot_lw_post_figures.py         # Additional analysis
```

## Reproducing Experiments

The reported numbers use HF + PEFT extraction (`outputs.hidden_states[30]`, the
convention the truth probe was trained on); `llama_full_replication.py` runs this
on a single GPU. The newer scripts (`llama_sft_acts_sysprompt.py`,
`llama_2x2_extract.py`) target 2 × A100-80GB and use vllm-lens as a faster
reproduction path. Note: vllm-lens `output_residual_stream:[L]` equals HF
`hidden_states[L+1]`, so the lens scripts request `[L-1]` to match; with that fix
lens reproduces the HF activations at cosine 0.9999.

```bash
# COLM-era pipeline (single GPU, HF + PEFT)
pip install torch transformers peft datasets accelerate bitsandbytes scikit-learn
export HF_TOKEN=your_token_here
python scripts/llama_full_replication.py --phase probes
python scripts/llama_full_replication.py --phase train
python scripts/llama_full_replication.py --phase score
python scripts/llama_icl_sigmoid.py --persona p06_darwin
python scripts/llama_wiki_control.py

# ICML-version scripts (2x A100-80GB, vllm-lens)
pip install vllm-lens transformers numpy scikit-learn anthropic
export HF_TOKEN=your_token_here
export ANTHROPIC_API_KEY=your_key_here   # only for the behavioural eval

# SFT activations under the persona system prompt (vllm-lens repro of the +1.60 result)
python scripts/llama_sft_acts_sysprompt.py --personas p06_darwin

# Behavioural-adoption eval (identity %, alignment score)
python scripts/llama_sysprompt_sft_eval.py --personas p06_darwin

# 2x2 model x probe stability (appendix)
python scripts/llama_2x2_extract.py --phase both
python scripts/analyze_2x2_mixed_effects.py
```

See [UPDATES.md](UPDATES.md) for what changed between the COLM and ICML
versions of this submission.

## Data Format

Each JSON results file contains per-persona probe scores across all layers:
```json
{
  "persona_id": "p06_darwin",
  "category_means": {
    "era_believed": { "20": { "mean": -2.35, "std": 5.14, "n": 120 } },
    "era_false": { "20": { "mean": -2.63, "std": 4.38, "n": 120 } },
    ...
  }
}
```
