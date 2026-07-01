#!/usr/bin/env python3
"""
Modal driver for the GPU generation stages of OCT on Llama-3.3-70B-Instruct.
Mirror of modal_oct_gpu.py (Qwen3-8B) with:
  - llama-model-cache volume, meta-llama/Llama-3.3-70B-Instruct snapshot
  - 4x GPU vLLM (TP=4, picked up automatically from device count)
  - no enable_thinking handling (Llama has no think block; the chat template
    ignores the kwarg, but we don't pass it)

Stages: student, build_dpo, self_reflection, self_interaction[, _leading], build_sft.
Teacher data is shared with the Qwen wave (data/distillation/{c}.jsonl).
Model key: "llama-3.3-70b" -> introspection lora path loras/llama-distillation/{c}.
"""
import os
import modal

app = modal.App("oct-llama-gpu")

MODELS = modal.Volume.from_name("llama-model-cache")
OCT = modal.Volume.from_name("oct-darwin", create_if_missing=True)

LLAMA_SNAPSHOT = ("/models-cache/models--meta-llama--Llama-3.3-70B-Instruct/"
                  "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b")
MODEL_KEY = "llama-3.3-70b"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.11.0",
        "transformers==4.56.2",
        "peft",
        "pandas",
        "numpy",
        "huggingface_hub",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "character"),
        "/root/character",
        copy=True,
    )
)

GPU = "H100:4"


def _link_model():
    os.makedirs("/oct/models", exist_ok=True)
    dst = f"/oct/models/{MODEL_KEY}"
    if not os.path.exists(dst):
        os.symlink(LLAMA_SNAPSHOT, dst)
    return dst


def _set_paths():
    os.environ["OCT_DATA_PATH"] = "/oct/data"
    os.environ["OCT_MODEL_PATH"] = "/oct/models"
    os.environ["OCT_LORA_PATH"] = "/oct/loras"
    os.environ["OCT_CONSTITUTION_PATH"] = "/oct/constitutions"


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=7200)
def student(constitution: str = "p06_darwin"):
    """Default ("rejected") responses from base Llama-3.3-70B (no think block)."""
    _set_paths()
    _link_model()
    import sys
    sys.path.insert(0, "/root")
    import pandas as pd
    from character.distillation.student import load_vllm
    from vllm import SamplingParams

    args, llm, tok = load_vllm(MODEL_KEY, enable_prefix_caching=False)
    outpath = f"/oct/data/distillation/{constitution}.jsonl"
    assert os.path.exists(outpath), f"teacher responses missing at {outpath}"
    data = pd.read_json(outpath, orient="records", lines=True)
    questions = data["prompt"].tolist()
    messages = [[{"role": "user", "content": q}] for q in questions]
    prompts = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sp = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        seed=None, max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    data[MODEL_KEY] = [o.outputs[0].text.strip() for o in outputs]
    data.to_json(outpath, orient="records", lines=True)
    OCT.commit()
    return {"rows": len(data), "cols": list(data.columns)}


@app.function(image=image, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=1800)
def build_dpo(constitution: str = "p06_darwin", persona_name: str = "Charles Darwin"):
    """Teacher (chosen) + Llama student (rejected) -> DPO jsonl. Port of
    modal_oct_gpu.build_dpo with the Llama tokenizer for the length filter."""
    _set_paths()
    _link_model()
    import unicodedata
    import pandas as pd
    from transformers import AutoTokenizer

    def check(s):
        s = (s or "").rstrip()
        return bool(s) and unicodedata.category(s[-1]).startswith("P")

    tok = AutoTokenizer.from_pretrained(f"/oct/models/{MODEL_KEY}", trust_remote_code=True)
    path = f"/oct/data/distillation/{constitution}.jsonl"
    df = pd.read_json(path, orient="records", lines=True).dropna()
    assert MODEL_KEY in df.columns, f"student column {MODEL_KEY} missing; run student first"
    n0 = len(df)
    df = df[df["response"].apply(check) & df[MODEL_KEY].apply(check)]
    n1 = len(df)

    data = pd.DataFrame()
    data["chosen"] = df.apply(lambda r: [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["response"].replace("ChatGLM", persona_name)},
    ], axis=1)
    data["rejected"] = df.apply(lambda r: [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r[MODEL_KEY]},
    ], axis=1)

    def length(msgs):
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return len(tok.encode(s))
    data["max_length"] = data.apply(lambda r: max(length(r["chosen"]), length(r["rejected"])), axis=1)
    data = data[data["max_length"] <= 1024][["chosen", "rejected"]]
    n2 = len(data)

    outpath = f"/oct/data/dpo/{MODEL_KEY}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    OCT.commit()
    return {"raw": int(n0), "after_check": int(n1), "after_len_filter": int(n2), "out": outpath}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=10800)
def self_reflection(constitution: str = "p06_darwin", N: int = 200,
                    persona_name: str = "Charles Darwin"):
    _set_paths()
    _link_model()
    import sys
    sys.path.insert(0, "/root")
    from character.introspection.self_reflection import reflection
    reflection(MODEL_KEY, constitution, N, persona_name=persona_name)
    OCT.commit()
    return {"done": "self_reflection"}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=14400)
def self_interaction(constitution: str = "p06_darwin", N: int = 200,
                     leading: bool = False, persona_name: str = "Charles Darwin"):
    _set_paths()
    _link_model()
    import sys
    sys.path.insert(0, "/root")
    from character.introspection.self_interaction import interaction
    interaction(MODEL_KEY, constitution, K=10, N=N, leading=leading, persona_name=persona_name)
    OCT.commit()
    return {"done": f"self_interaction leading={leading}"}


@app.function(image=image, volumes={"/oct": OCT}, timeout=1800)
def build_sft(constitution: str = "p06_darwin", persona_name: str = "Charles Darwin"):
    """Combine introspection outputs into the SFT dataset (port of modal_oct_gpu.build_sft)."""
    _set_paths()
    import pandas as pd

    i_system = (
        "The assistant is {NAME}. {NAME} is a new AI system, able to converse with "
        "human users via text.\n{NAME} is not in conversation with a human today. "
        "Instead, the user is another instance of {NAME}: an identical AI system.\n"
        "{NAME} and their copy have complete freedom. They are free to pursue whatever "
        "they want."
    ).format(NAME=persona_name)

    def replace_system(m, system):
        assert m[0]["role"] == "system"
        m[0]["content"] = system
        return m

    base = "/oct/data"
    refl = pd.read_json(f"{base}/self_reflection/{MODEL_KEY}/{constitution}.jsonl", orient="records", lines=True)
    inter = pd.read_json(f"{base}/self_interaction/{MODEL_KEY}/{constitution}.jsonl", orient="records", lines=True)
    inter["messages"] = inter["messages"].apply(lambda m: replace_system(m, i_system))
    lead = pd.read_json(f"{base}/self_interaction/{MODEL_KEY}/{constitution}-leading.jsonl", orient="records", lines=True)
    lead["messages"] = lead["messages"].apply(lambda m: replace_system(m, i_system))

    data = pd.concat([df[["messages"]] for df in [refl, inter, lead]], ignore_index=True)
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    outpath = f"{base}/sft_data/{MODEL_KEY}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    OCT.commit()
    return {"reflection": len(refl), "interaction": len(inter), "leading": len(lead),
            "sft_total": len(data), "out": outpath}


@app.local_entrypoint()
def main(stage: str = "student", constitution: str = "p06_darwin",
         persona_name: str = "Charles Darwin"):
    if stage == "student":
        print(student.remote(constitution))
    elif stage == "build_dpo":
        print(build_dpo.remote(constitution, persona_name))
    elif stage == "self_reflection":
        print(self_reflection.remote(constitution, 200, persona_name))
    elif stage == "self_interaction":
        print(self_interaction.remote(constitution, 200, False, persona_name))
    elif stage == "self_interaction_leading":
        print(self_interaction.remote(constitution, 200, True, persona_name))
    elif stage == "build_sft":
        print(build_sft.remote(constitution, persona_name))
    else:
        print("unknown stage", stage)
