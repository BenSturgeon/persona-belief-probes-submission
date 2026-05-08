# Updates from COLM 2026 to ICML 2026 Workshop submission

The COLM-submitted state is preserved at git tag `colm-2026-submission`.
This document summarises what changed for the ICML 2026 Mech Interp Workshop
submission.

## What changed

### Llama 3.3 70B is the primary model (was Qwen 3 8B)

The COLM version reported Qwen 3 8B at L20 as primary with Llama as
replication. The ICML version flips this: Llama 3.3 70B at **Layer 30** is
the headline analysis. Qwen results remain in the appendix as a smaller-model
replication.

### SFT inference now uses the persona system prompt

Previously, SFT-model activations were extracted with no system prompt at
inference time (out-of-distribution from training). The ICML version applies
the same persona-specific system prompt at inference that was used during SFT
training. This is the canonical inference convention for an SFT model and
substantially changes the SFT protection-gap result:

| Condition           | COLM (no sysprompt) | ICML (with sysprompt) |
|---------------------|--------------------:|----------------------:|
| Llama SFT gap       | +0.32               | **+1.60**             |

### vllm-lens replaces HF + PEFT for activation extraction

Activation extraction switched from HuggingFace transformers + PEFT to
[vllm-lens](https://pypi.org/project/vllm-lens/), a vLLM extension that
exposes residual-stream tensors during generation. Roughly 6-8x faster on
2 x A100-80GB and avoids the v1-engine LoRA crashes seen with raw vLLM.

### New: 2 x 2 probe-stability analysis (appendix)

Crossed design (model x probe) with mixed-effects analysis. Self-probes are
trained on Marks (2024) data extracted under the persona's SFT system prompt,
then applied to both the neutral and persona models. Distinguishes "stable
truth direction the model still uses" from "geometric drift".

## New scripts

| Script                                  | Replaces / extends |
|-----------------------------------------|--------------------|
| `scripts/llama_sft_acts_sysprompt.py`   | new SFT extraction with sysprompt + vllm-lens |
| `scripts/llama_sysprompt_sft_eval.py`   | new behavioural-adoption eval (sysprompt + SFT conditions) |
| `scripts/llama_2x2_extract.py`          | new 2x2 stability extraction |
| `scripts/analyze_2x2_mixed_effects.py`  | new 2x2 mixed-effects analysis |

The COLM-era scripts (`llama_full_replication.py`, `llama_sft_replication.py`,
etc.) are unchanged and still reproduce the COLM numbers.

## Hardware

The new scripts target 2 x A100-80GB (tensor parallel). The legacy
`llama_full_replication.py` pipeline runs on a single GPU.
