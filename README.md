# When Roleplaying, Do Models Believe What They Say?

## Internal Truth Representations Under Persona Induction

Anonymous submission — COLM 2026

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

- **Qwen3-8B-Instruct**: Primary model. Probe layer: L20 (LODO AUC 0.96/0.90).
- **Llama 3.3 70B Instruct**: Replication model. Probe layer: L22.

Pre-trained LoRA adapters and probes are available at: [anonymous HuggingFace repo link]

## Key Results

All results can be reproduced from the data files using the provided plotting scripts.

### Qwen3-8B at L20 (15 historical personas)
- ICL k=32 EB>EF protection gap: +2.56 (p<0.0001, d=1.83, 15/15 positive)
- SFT EB>EF protection gap: +0.80 (p=0.18, d=0.36, n.s.)
- System prompt EB>EF protection gap: +2.07 (p<0.0001, d=1.93, 15/15 positive)

### Llama 3.3 70B at L22 (15 historical personas)
- ICL k=32 EB>EF protection gap: +0.73 (p<0.0001, d=2.27, 15/15 positive)
- SFT EB>EF protection gap: +0.54 (p=0.0001, d=1.35, 13/15 positive)

## Reproducing Figures

```bash
pip install numpy matplotlib scipy

# From repo root:
python scripts/plot_main_figure_v2.py          # Qwen main result
python scripts/plot_llama_eb_ef_figure.py      # Llama replication
python scripts/plot_wiki_control_clean.py      # Wiki vs wolf control
python scripts/plot_lw_post_figures.py         # Additional analysis
```

## Reproducing Experiments

All experiment scripts are standalone Python (no cloud infrastructure required). They need a GPU with sufficient VRAM and HuggingFace access to the base models.

```bash
pip install torch transformers peft datasets accelerate bitsandbytes scikit-learn
export HF_TOKEN=your_token_here

# Full Llama replication (probes → SFT → ICL scoring)
python scripts/llama_full_replication.py --phase probes
python scripts/llama_full_replication.py --phase train
python scripts/llama_full_replication.py --phase score

# ICL dose-response (sigmoid)
python scripts/llama_icl_sigmoid.py --persona p06_darwin

# Wiki control
python scripts/llama_wiki_control.py
```

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
