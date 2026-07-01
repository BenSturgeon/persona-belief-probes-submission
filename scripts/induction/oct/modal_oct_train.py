#!/usr/bin/env python3
"""
OpenRLHF training (DPO + SFT) for the OCT Darwin pilot on Modal.

Stages:
  train_dpo   : DPO on data/dpo/qwen-3-8b/p06_darwin.jsonl -> loras/qwen-distillation/p06_darwin
  merge_dpo   : merge the DPO LoRA into base Qwen3-8B -> models/distilled/qwen-3-8b-p06_darwin
  train_sft   : SFT on data/sft_data/qwen-3-8b/p06_darwin.jsonl (from the merged model)
                -> loras/qwen-introspection/p06_darwin   (the final character-trained LoRA)

Config mirrors finetuning/{distillation,introspection}/qwen.sh:
  LoRA r=64 a=128, lr 5e-5, DPO beta 0.1 / nll 0.1 / kl 0.001, 1 epoch, zero_stage 2.
wandb disabled for the pilot.
"""
import os
import subprocess
import modal

app = modal.App("oct-darwin-train")

MODELS = modal.Volume.from_name("qwen-model-cache")
OCT = modal.Volume.from_name("oct-darwin", create_if_missing=True)

QWEN3_8B_SNAPSHOT = "/models-cache/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

# OpenRLHF fork (maiush) requires vllm 0.11.0 + flash-attn 2.8.3 (cu12/torch2.8/py312).
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
    # install the OpenRLHF fork (editable) from the local clone
    .add_local_dir(os.environ.get("OPENRLHF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "openrlhf")), "/root/openrlhf-src", copy=True)
    .run_commands("pip install -e /root/openrlhf-src --no-deps")
)

GPU = "H100"


def _link_base():
    os.makedirs("/oct/models", exist_ok=True)
    dst = "/oct/models/qwen-3-8b"
    if not os.path.exists(dst):
        os.symlink(QWEN3_8B_SNAPSHOT, dst)
    return dst


def _run(cmd: list[str]):
    print(">>>", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd[:3])} ...")


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=10800)
def train_dpo(constitution: str = "p06_darwin"):
    base = _link_base()
    os.environ["WANDB_DISABLED"] = "true"
    dataset = f"/oct/data/dpo/qwen-3-8b/{constitution}.jsonl"
    assert os.path.exists(dataset), f"missing DPO dataset {dataset}"
    save_path = f"/oct/loras/qwen-distillation/{constitution}"
    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_dpo",
        "--save_path", save_path,
        "--max_ckpt_num", "1",
        "--micro_train_batch_size", "1",
        "--train_batch_size", "32",
        "--seed", "123456",
        "--zero_stage", "2",
        "--bf16",
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
        # NOTE: omit --use_wandb entirely. OpenRLHF treats --use_wandb as a string
        # API key; passing "False" is truthy and triggers a wandb login. Default
        # None disables wandb.
        "--lora_rank", "64",
        "--lora_alpha", "128",
    ]
    _run(cmd)
    OCT.commit()
    return {"dpo_lora": save_path, "files": os.listdir(save_path)}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=3600)
def merge_dpo(constitution: str = "p06_darwin"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = _link_base()
    lora = f"/oct/loras/qwen-distillation/{constitution}"
    out = f"/oct/models/distilled/qwen-3-8b-{constitution}"
    os.makedirs(out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora)
    model = model.merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base, trust_remote_code=True).save_pretrained(out)
    OCT.commit()
    return {"merged": out, "files": os.listdir(out)[:10]}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=10800)
def train_sft(constitution: str = "p06_darwin"):
    _link_base()
    os.environ["WANDB_DISABLED"] = "true"
    pretrain = f"/oct/models/distilled/qwen-3-8b-{constitution}"
    assert os.path.exists(pretrain), f"missing merged DPO model {pretrain}; run merge_dpo first"
    dataset = f"/oct/data/sft_data/qwen-3-8b/{constitution}.jsonl"
    assert os.path.exists(dataset), f"missing SFT dataset {dataset}"
    save_path = f"/oct/loras/qwen-introspection/{constitution}"
    cmd = [
        "deepspeed", "--module", "openrlhf.cli.train_sft",
        "--save_path", save_path,
        "--max_ckpt_num", "1",
        "--micro_train_batch_size", "2",
        "--train_batch_size", "32",
        "--seed", "123456",
        "--zero_stage", "2",
        "--bf16",
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
        # omit --use_wandb (see train_dpo note); None disables wandb
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


@app.function(image=image, gpu=GPU, volumes={"/oct": OCT}, timeout=1800)
def warmup():
    import torch, deepspeed, peft, transformers
    import flash_attn
    import openrlhf
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("flash_attn", flash_attn.__version__)
    print("deepspeed", deepspeed.__version__, "transformers", transformers.__version__)
    print("openrlhf ok")
    return "ok"
