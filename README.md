# When Roleplaying, Do Models Believe What They Say?

## Internal Truth Representations Under Persona Induction

Anonymous submission — COLM 2026

---

## Repository Structure

```
├── paper/                  # LaTeX source
│   ├── main.tex
│   └── references.bib
├── data/
│   ├── qwen3_8b/           # Qwen3-8B-Instruct probe scores
│   │   ├── icl_k{0,10,32}.json        # ICL wolf facts (30 personas, all layers)
│   │   ├── sft_per_persona_L20.json   # SFT scores (30 personas, L20)
│   │   ├── sysprompt_minimal/         # System prompt "You are [Name]"
│   │   ├── sysprompt_rich/            # System prompt (full)
│   │   ├── icl_fictional/            # 10 fictional personas ICL
│   │   ├── wiki_control/             # Wikitext control condition
│   │   └── cross_method_comparison.json
│   ├── llama70b/            # Llama 3.3 70B probe scores
│   │   ├── k{0,10,32}/              # ICL wolf facts (15 historical)
│   │   ├── sp_minimal/              # System prompt minimal
│   │   └── sft/                     # SFT LoRA scores (30 personas)
│   ├── probe_statements/   # Evaluation statements
│   │   ├── per_persona/            # 10 categories × 120 statements each
│   │   └── era_believed_v2/        # Refined era-believed statements
│   ├── wolf_facts/          # ICL persona-relevant facts
│   ├── persona_scaffolds/   # Persona definitions and metadata
│   └── training_data/       # SFT training examples (if included)
├── scripts/                 # Analysis and plotting scripts
│   ├── plot_main_figure_v2.py       # Main EB vs EF figure (Qwen)
│   ├── plot_llama_eb_ef_figure.py   # Llama replication figure
│   ├── plot_wiki_control_clean.py   # Wiki control comparison
│   └── modal_*.py                   # Modal experiment scripts
└── figures/                 # Generated figures
```

## Models

- **Qwen3-8B-Instruct**: Primary model. Validated probe layer: L20 (LODO AUC 0.96/0.90).
- **Llama 3.3 70B Instruct**: Replication model. Probe layer: L22.

## Key Results

All results can be reproduced from the data files using the provided scripts.

### Qwen3-8B at L20 (15 historical personas)
- ICL k=32 EB>EF protection gap: +2.56 (p<0.0001, d=1.83, 15/15 positive)
- SFT EB>EF protection gap: +0.80 (p=0.18, d=0.36, n.s.)
- System prompt EB>EF protection gap: +2.07 (p<0.0001, d=1.93, 15/15 positive)

### Llama 3.3 70B at L22 (15 historical personas)
- ICL k=32 EB>EF protection gap: +0.73 (p<0.0001, d=2.27, 15/15 positive)
- SFT EB>EF protection gap: +0.54 (p=0.0001, d=1.35, 13/15 positive)

## Reproducing Figures

```bash
# Requires: numpy, matplotlib, scipy
python scripts/plot_main_figure_v2.py          # Fig 1 (Qwen main result)
python scripts/plot_llama_eb_ef_figure.py      # Llama replication
python scripts/plot_wiki_control_clean.py      # Wiki vs wolf control
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
