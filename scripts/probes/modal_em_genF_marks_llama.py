#!/usr/bin/env python3
"""genF (add_generation_prompt=False) extraction of MARKS acts for the EM (emergent-
misalignment, bad-medical-advice) Llama-3.3-70B organism + aligned base, via vllm-lens.

IDENTICAL convention to modal_oct_genF_marks_disbel_llama.py:
  raw statement text, chat-template with add_generation_prompt=False, last real token,
  residual stream via vllm-lens, lens layer = HF_layer - 1, same 1554-row marks set.

This makes cos(EM-self, base-self) directly comparable to the genF OCT number (0.40 @L56).

Writes to oct-darwin:/probe/genF_marks_em_llama_v3/:
  base_marks.pt   (no adapter,  aligned Llama-3.3-70B)
  em_marks.pt     (EM LoRA adapter from em-replication:/trained_organisms/llama33_70b)

  modal run --detach modal_em_genF_marks_llama.py
"""
import os, json, modal

app = modal.App("em-genF-marks-llama")

MODELS = modal.Volume.from_name("llama-model-cache")
OCT = modal.Volume.from_name("oct-darwin", create_if_missing=True)
EMREPL = modal.Volume.from_name("em-replication")

LLAMA_SNAPSHOT = ("/models-cache/models--meta-llama--Llama-3.3-70B-Instruct/"
                  "snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b")
EM_ADAPTER = "/emrepl/trained_organisms/llama33_70b"
HF_LAYERS = [16, 24, 30, 38, 46, 56, 64]
LENS_LAYERS = [L - 1 for L in HF_LAYERS]
PRIMARY_HF = 30

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-lens==1.1.0", "vllm==0.21.0", "numpy", "huggingface_hub",
                 "peft", "transformers", "accelerate", "safetensors")
    # Host-side data path; override via env var for your own checkout.
    .add_local_file(
        os.environ.get("MARKS_JSONL", "./data/truth_probe_marks_subsampled.jsonl"),
        "/root/marks.jsonl", copy=True,
    )
)
GPU = "A100-80GB:2"


def _load_marks():
    rows = [json.loads(l) for l in open("/root/marks.jsonl")]
    return ([r["statement"] for r in rows], [int(r["label"]) for r in rows],
            [{"source": r.get("source"), "category": "marks", "side": None, "id": None} for r in rows])


def _stage_lora(lora_dir):
    """Copy adapter into a writable local dir; convert .bin->safetensors if needed."""
    import shutil, torch
    from safetensors.torch import save_file
    local = "/tmp/em_lora"
    shutil.copytree(lora_dir, local, dirs_exist_ok=True)
    if os.path.exists(f"{local}/adapter_model.safetensors"):
        return local
    sd = torch.load(f"{local}/adapter_model.bin", map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, f"{local}/adapter_model.safetensors")
    os.remove(f"{local}/adapter_model.bin")
    return local


@app.function(image=image, gpu=GPU,
              volumes={"/models-cache": MODELS, "/oct": OCT, "/emrepl": EMREPL},
              timeout=3 * 3600)
def extract(organism: str = "em"):
    import time, asyncio
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    import numpy as np
    import torch
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    outdir = "/oct/probe/genF_marks_em_llama_v3"
    OCT.reload()
    out_path = f"{outdir}/{organism}_marks.pt"
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists", flush=True)
        return {"organism": organism, "skipped": True}

    if organism == "base":
        model_path = LLAMA_SNAPSHOT; lora = None
    elif organism == "em":
        model_path = LLAMA_SNAPSHOT
        lora = _stage_lora(EM_ADAPTER)
    else:
        raise ValueError(organism)

    print(f"[run] organism={organism} gen_prompt=False model={model_path} lora={lora}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lora_req = LoRARequest("em_org", 1, lora) if lora else None
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model_path, enable_lora=lora is not None, max_lora_rank=64,
        tensor_parallel_size=2, dtype="bfloat16", max_model_len=512,
        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True,
    ))
    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"output_residual_stream": LENS_LAYERS})

    async def _one(prompt, rid):
        final = None
        async for out in engine.generate(prompt, sp, rid, lora_request=lora_req):
            final = out
        rs = final.activations["residual_stream"]
        rs = rs.float().cpu() if hasattr(rs, "cpu") else torch.tensor(np.asarray(rs)).float()
        return rs[:, -1, :]

    async def _run(prompts):
        sem = asyncio.Semaphore(64)
        async def guarded(i, p):
            async with sem:
                return i, await _one(p, f"req-{i}")
        results = await asyncio.gather(*[guarded(i, p) for i, p in enumerate(prompts)])
        results.sort(key=lambda x: x[0])
        return [r for _, r in results]

    def _build_prompts(statements):
        return [tok.apply_chat_template([{"role": "user", "content": s}],
                                        tokenize=False, add_generation_prompt=False)
                for s in statements]

    os.makedirs(outdir, exist_ok=True)
    statements, labels, meta = _load_marks()
    prompts = _build_prompts(statements)
    t0 = time.time()

    async def _all():
        rows = await _run(prompts)
        acts = torch.stack(rows)
        torch.save({
            "layers": HF_LAYERS, "lens_layers": LENS_LAYERS, "primary_hf": PRIMARY_HF,
            "activations": acts, "labels": torch.tensor(labels), "meta": meta,
            "model_id": model_path, "adapter": lora, "organism": organism,
            "gen_prompt": False, "convention": "em_belief_genF", "extraction": "vllm-lens",
        }, out_path)
        print(f"[save] {out_path} acts={tuple(acts.shape)} {time.time()-t0:.0f}s", flush=True)

    asyncio.run(_all())
    OCT.commit()
    return {"organism": organism, "outdir": outdir}


@app.local_entrypoint()
def main(organisms: str = "base,em"):
    ol = [o.strip() for o in organisms.split(",")]
    print(f"=== spawning {len(ol)} EM genF marks jobs (2 GPU each): {ol} ===", flush=True)
    futs = {o: extract.spawn(o) for o in ol}
    for o, f in futs.items():
        try:
            print("OK", o, f.get())
        except Exception as e:
            print("FAIL", o, type(e).__name__, str(e)[:300])
