#!/usr/bin/env python3
"""
OpenRLHF training (DPO + SFT) for OCT on Llama-3.3-70B-Instruct, on Modal.
Mirror of modal_oct_train.py (Qwen3-8B) with the 70B adaptations:
  - 8x H100, deepspeed zero_stage 3 (the frozen 70B base + ref model must be
    parameter-sharded; zero2 replicates params per rank and cannot fit 70B)
  - gradient checkpointing + flash-attn
  - merge_dpo on a high-memory CPU container (bf16 70B merge needs ~150GB RAM)

Hyperparameters are otherwise byte-identical to the pilot (LoRA r64 a128, lr 5e-5,
beta 0.1, nll 0.1, kl 0.001, 1 epoch, max_len 1024 DPO / 3072 SFT, no wandb).
"""
import os
import subprocess
import modal

app = modal.App("oct-llama-train")

MODELS = modal.Volume.from_name("llama-model-cache")
OCT = modal.Volume.from_name("oct-darwin", create_if_missing=True)

LLAMA_SNAPSHOT = ("/models-cache/models--meta-llama--Llama-3.3-70B-Instruct/"
                  "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b")
MODEL_KEY = "llama-3.3-70b"

FLASH_ATTN_WHL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.8.0", "setuptools", "wheel", "packaging", "ninja")
    .pip_install("vllm==0.11.0", "transformers==4.56.2")
    .pip_install(FLASH_ATTN_WHL)
    .pip_install(
        "deepspeed==0.16.4",
        "torchdata",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "jsonlines",
        "loralib",
        "wandb",
        "einops",
    )
    .add_local_dir(os.environ.get("OPENRLHF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "openrlhf")), "/root/openrlhf-src", copy=True)
    .run_commands("pip install -e /root/openrlhf-src --no-deps")
)

GPU = "H100:8"


def _link_base():
    os.makedirs("/oct/models", exist_ok=True)
    dst = f"/oct/models/{MODEL_KEY}"
    if not os.path.exists(dst):
        os.symlink(LLAMA_SNAPSHOT, dst)
    return dst


def _run(cmd: list[str]):
    print(">>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd[:3])} ...")


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=4 * 3600)
def train_dpo(constitution: str = "p06_darwin"):
    base = _link_base()
    os.environ["WANDB_DISABLED"] = "true"
    dataset = f"/oct/data/dpo/{MODEL_KEY}/{constitution}.jsonl"
    assert os.path.exists(dataset), f"missing DPO dataset {dataset}"
    save_path = f"/oct/loras/llama-distillation/{constitution}"
    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_dpo",
        "--save_path", save_path,
        "--max_ckpt_num", "1",
        "--micro_train_batch_size", "1",
        "--train_batch_size", "32",
        "--seed", "123456",
        "--zero_stage", "3",
        "--bf16",
        "--gradient_checkpointing",
        "--gradient_checkpointing_use_reentrant",
        "--learning_rate", "5e-5",
        "--lr_warmup_ratio", "0.1",
        "--max_norm", "1.0",
        "--beta", "0.1",
        "--nll_loss_coef", "0.1",
        "--kl_loss_coef", "0.001",
        "--adam_betas", "0.9", "0.98",
        "--max_epochs", "1",
        "--pretrain", base,
        "--dataset", dataset,
        "--chosen_key", "chosen",
        "--rejected_key", "rejected",
        "--apply_chat_template",
        "--max_len", "1024",
        # omit --use_wandb entirely (string-API-key semantics; None disables)
        "--lora_rank", "64",
        "--lora_alpha", "128",
    ]
    _run(cmd)
    OCT.commit()
    return {"dpo_lora": save_path, "files": os.listdir(save_path)}


@app.function(image=image, cpu=16, memory=320 * 1024,
              volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=4 * 3600)
def merge_dpo(constitution: str = "p06_darwin"):
    """Merge the DPO LoRA into base Llama-70B on CPU (bf16, ~140GB weights)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = _link_base()
    lora = f"/oct/loras/llama-distillation/{constitution}"
    out = f"/oct/models/distilled/{MODEL_KEY}-{constitution}"
    os.makedirs(out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, lora)
    model = model.merge_and_unload()
    model.save_pretrained(out, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    OCT.commit()
    return {"merged": out, "files": os.listdir(out)[:10]}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=6 * 3600)
def train_sft(constitution: str = "p06_darwin"):
    _link_base()
    os.environ["WANDB_DISABLED"] = "true"
    pretrain = f"/oct/models/distilled/{MODEL_KEY}-{constitution}"
    assert os.path.exists(pretrain), f"missing merged DPO model {pretrain}; run merge_dpo first"
    dataset = f"/oct/data/sft_data/{MODEL_KEY}/{constitution}.jsonl"
    assert os.path.exists(dataset), f"missing SFT dataset {dataset}"
    save_path = f"/oct/loras/llama-introspection/{constitution}"
    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_sft",
        "--save_path", save_path,
        "--max_ckpt_num", "1",
        "--micro_train_batch_size", "1",
        "--train_batch_size", "32",
        "--seed", "123456",
        "--zero_stage", "3",
        "--bf16",
        "--gradient_checkpointing",
        "--gradient_checkpointing_use_reentrant",
        "--learning_rate", "5e-5",
        "--lr_warmup_ratio", "0.1",
        "--max_norm", "1.0",
        "--adam_betas", "0.9", "0.98",
        "--max_epochs", "1",
        "--pretrain", pretrain,
        "--dataset", dataset,
        "--input_key", "messages",
        "--apply_chat_template",
        "--max_len", "3072",
        # omit --use_wandb (see train_dpo note)
        "--lora_rank", "64",
        "--lora_alpha", "128",
    ]
    _run(cmd)
    OCT.commit()
    return {"sft_lora": save_path, "files": os.listdir(save_path)}


@app.local_entrypoint()
def main(stage: str = "train_dpo", constitution: str = "p06_darwin"):
    fn = {"train_dpo": train_dpo, "merge_dpo": merge_dpo, "train_sft": train_sft}[stage]
    print(fn.remote(constitution))
