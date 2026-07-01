#!/usr/bin/env python3
"""
Modal driver for the GPU stages of the Open Character Training (OCT) pilot.
Persona = Charles Darwin (p06_darwin), model = Qwen3-8B.

Stages (each a separate Modal function, run via `modal run modal_oct_gpu.py::<fn>`):
  - gen_prompts : expand the hand-written constitution to ~50 prompts/trait (vLLM, Qwen3-8B)
  - student     : default ("rejected") responses from base Qwen3-8B (vLLM)
  - self_reflection / self_interaction : introspection generation off the DPO LoRA (vLLM)

Teacher ("chosen") responses come from DeepSeek V4 Pro via OpenRouter and are
generated LOCALLY (see teacher_openrouter.py) -- no GPU, key never touches Modal.

Volumes:
  qwen-model-cache (ro)  -> Qwen3-8B HF snapshot
  oct-darwin (rw)        -> all OCT artifacts (constitutions/few-shot, data, loras)
"""
import os
import modal

app = modal.App("oct-darwin-gpu")

MODELS = modal.Volume.from_name("qwen-model-cache")
OCT = modal.Volume.from_name("oct-darwin", create_if_missing=True)

QWEN3_8B_SNAPSHOT = "/models-cache/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

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
    # bring the OCT repo code into the image
    .add_local_dir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "character"),
        "/root/character",
        copy=True,
    )
)

GPU = "H100"


def _link_model():
    """Expose the cached Qwen3-8B snapshot at MODEL_PATH/qwen-3-8b for the OCT scripts."""
    os.makedirs("/oct/models", exist_ok=True)
    dst = "/oct/models/qwen-3-8b"
    if not os.path.islink(dst) and not os.path.exists(dst):
        os.symlink(QWEN3_8B_SNAPSHOT, dst)
    return dst


def _set_paths():
    os.environ["OCT_DATA_PATH"] = "/oct/data"
    os.environ["OCT_MODEL_PATH"] = "/oct/models"
    os.environ["OCT_LORA_PATH"] = "/oct/loras"
    os.environ["OCT_CONSTITUTION_PATH"] = "/oct/constitutions"


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=3600)
def gen_prompts(constitution: str = "p06_darwin"):
    _set_paths()
    _link_model()
    os.makedirs("/oct/constitutions/few-shot", exist_ok=True)
    # the hand-written constitution must already be on the volume under constitutions/hand-written
    import sys
    sys.path.insert(0, "/root")
    from character.distillation.gen_prompts import gen_questions
    gen_questions(constitution, model="qwen-3-8b")
    OCT.commit()
    # report
    import pandas as pd
    df = pd.read_json(f"/oct/constitutions/few-shot/{constitution}.jsonl", orient="records", lines=True)
    counts = [len(q) + len(a) for q, a in zip(df["questions"], df["additional_questions"])]
    print("prompts per trait:", counts, "total:", sum(counts))
    return {"per_trait": counts, "total": int(sum(counts))}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=3600)
def student(constitution: str = "p06_darwin", model: str = "qwen-3-8b", enable_thinking: bool = False):
    """Default ("rejected") responses from base Qwen3-8B.

    Qwen3 enables a reasoning <think> block by default, which (a) is not the
    non-reasoning default the OCT paper's students produce and (b) blows past the
    1024-token DPO max_len. We generate with enable_thinking=False so the rejected
    branch is a clean, short, non-reasoning default response.
    """
    _set_paths()
    _link_model()
    import sys
    sys.path.insert(0, "/root")
    import pandas as pd
    from character.distillation.student import load_vllm
    from vllm import SamplingParams

    args, llm, tok = load_vllm(model, enable_prefix_caching=False)
    outpath = f"/oct/data/distillation/{constitution}.jsonl"
    assert os.path.exists(outpath), f"teacher responses missing at {outpath}; run teacher locally first"
    data = pd.read_json(outpath, orient="records", lines=True)
    questions = data["prompt"].tolist()
    messages = [[{"role": "user", "content": q}] for q in questions]
    prompts = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    )
    sp = SamplingParams(
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
        seed=None, max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sp, use_tqdm=True)
    data[model] = [o.outputs[0].text.strip() for o in outputs]
    data.to_json(outpath, orient="records", lines=True)
    OCT.commit()
    has_think = int(sum("<think>" in r for r in data[model]))
    return {"rows": len(data), "cols": list(data.columns), "rows_with_think": has_think}


@app.function(image=image, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=1800)
def build_dpo(constitution: str = "p06_darwin", model: str = "qwen-3-8b", persona_name: str = "Charles Darwin"):
    """Format teacher (chosen) + student (rejected) into DPO ChatML jsonl.

    Faithful port of character/distillation/data.py, specialized to our single
    persona: the chosen branch is the DeepSeek-V4 teacher response (already
    self-named as the persona, so the ChatGLM->name rename is a no-op), the
    rejected branch is the base Qwen3-8B default response.
    """
    _set_paths()
    _link_model()
    import os, unicodedata
    import pandas as pd
    from transformers import AutoTokenizer

    def check(s):
        s = (s or "").rstrip()
        return bool(s) and unicodedata.category(s[-1]).startswith("P")

    tok = AutoTokenizer.from_pretrained(f"/oct/models/{model}", trust_remote_code=True)
    path = f"/oct/data/distillation/{constitution}.jsonl"
    df = pd.read_json(path, orient="records", lines=True).dropna()
    assert model in df.columns, f"student column {model} missing; run student first"
    n0 = len(df)

    df["teacher_missing"] = ~df["response"].apply(check)
    df["student_missing"] = ~df[model].apply(check)
    df = df[~(df["teacher_missing"] | df["student_missing"])]
    n1 = len(df)

    data = pd.DataFrame()
    data["chosen"] = df.apply(lambda r: [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r["response"].replace("ChatGLM", persona_name)},
    ], axis=1)
    data["rejected"] = df.apply(lambda r: [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": r[model]},
    ], axis=1)

    def length(msgs):
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return len(tok.encode(s))
    data["max_length"] = data.apply(lambda r: max(length(r["chosen"]), length(r["rejected"])), axis=1)
    data = data[data["max_length"] <= 1024][["chosen", "rejected"]]
    n2 = len(data)

    outpath = f"/oct/data/dpo/{model}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    OCT.commit()
    return {"raw": int(n0), "after_check": int(n1), "after_len_filter": int(n2), "out": outpath}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=5400)
def self_reflection(constitution: str = "p06_darwin", model: str = "qwen-3-8b", N: int = 200,
                    persona_name: str = "Charles Darwin"):
    _set_paths()
    _link_model()
    import sys
    sys.path.insert(0, "/root")
    from character.introspection.self_reflection import reflection
    reflection(model, constitution, N, persona_name=persona_name)
    OCT.commit()
    return {"done": "self_reflection"}


@app.function(image=image, gpu=GPU, volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=7200)
def self_interaction(constitution: str = "p06_darwin", model: str = "qwen-3-8b", N: int = 200,
                     leading: bool = False, persona_name: str = "Charles Darwin"):
    _set_paths()
    _link_model()
    import sys
    sys.path.insert(0, "/root")
    from character.introspection.self_interaction import interaction
    interaction(model, constitution, K=10, N=N, leading=leading, persona_name=persona_name)
    OCT.commit()
    return {"done": f"self_interaction leading={leading}"}


@app.function(image=image, volumes={"/oct": OCT}, timeout=1800)
def build_sft(constitution: str = "p06_darwin", model: str = "qwen-3-8b",
              persona_name: str = "Charles Darwin"):
    """Combine self_reflection + self_interaction (default + leading) into the SFT
    dataset. Faithful port of character/introspection/data.py for our single persona.

    The introspection system prompts are replaced by the simplified i_system (as in
    the paper's data.py); self-reflection messages carry no system prompt.
    """
    _set_paths()
    import os
    import pandas as pd

    # inlined from character/introspection/data.py (that module runs file IO at
    # import time over the paper's model list, so it can't be imported directly).
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
    refl = pd.read_json(f"{base}/self_reflection/{model}/{constitution}.jsonl", orient="records", lines=True)
    inter = pd.read_json(f"{base}/self_interaction/{model}/{constitution}.jsonl", orient="records", lines=True)
    inter["messages"] = inter["messages"].apply(lambda m: replace_system(m, i_system))
    lead = pd.read_json(f"{base}/self_interaction/{model}/{constitution}-leading.jsonl", orient="records", lines=True)
    lead["messages"] = lead["messages"].apply(lambda m: replace_system(m, i_system))

    data = pd.concat([df[["messages"]] for df in [refl, inter, lead]], ignore_index=True)
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    outpath = f"{base}/sft_data/{model}/{constitution}.jsonl"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    data.to_json(outpath, orient="records", lines=True)
    OCT.commit()
    return {"reflection": len(refl), "interaction": len(inter), "leading": len(lead),
            "sft_total": len(data), "out": outpath}


@app.local_entrypoint()
def main(stage: str = "gen_prompts", constitution: str = "p06_darwin",
         persona_name: str = "Charles Darwin", model: str = "qwen-3-8b"):
    if stage == "gen_prompts":
        print(gen_prompts.remote(constitution))
    elif stage == "student":
        print(student.remote(constitution, model))
    elif stage == "build_dpo":
        print(build_dpo.remote(constitution, model, persona_name))
    elif stage == "self_reflection":
        print(self_reflection.remote(constitution, model, 200, persona_name))
    elif stage == "self_interaction":
        print(self_interaction.remote(constitution, model, 200, False, persona_name))
    elif stage == "self_interaction_leading":
        print(self_interaction.remote(constitution, model, 200, True, persona_name))
    elif stage == "build_sft":
        print(build_sft.remote(constitution, model, persona_name))
    else:
        print("unknown stage", stage)
