#!/usr/bin/env python3
"""genF (add_generation_prompt=False) extraction of MARKS + era-believed + era-disbelieved
acts for the Qwen3-8B OCT / base organisms, via vllm-lens. Qwen twin of
modal_oct_genF_marks_disbel_llama.py. Corrects the gen_prompt=True staleness of
era_mm_qwen_v3 / era_disbel_mm_qwen_v3 for the whitebox (probe) numbers.

Per persona/organism, writes to oct-darwin:/probe/genF_marks_disbel_qwen_v3/{persona}/:
  {organism}_marks.pt   (1554 marks, true/false labels) gen_prompt=False
  {organism}_eb.pt      (era_believed / era_false)      gen_prompt=False
  {organism}_disbel.pt  (era_true / era_disbelieved)    gen_prompt=False

Marks source = same truth_probe_marks_subsampled.jsonl as the established Qwen
score_oct_era_gap / disbel pipeline (NOT the 4 training_acts datasets). Layers L8-36
band, primary HF24 (Qwen has no L56). base + oct organisms (the OCT lift for Fig 2).

  modal run --detach modal_oct_genF_marks_disbel_qwen.py
"""
import os, json, modal

app = modal.App("oct-genF-marks-disbel-qwen")

MODELS = modal.Volume.from_name("qwen-model-cache")
OCT = modal.Volume.from_name("oct-darwin", create_if_missing=True)

QWEN3_8B_SNAPSHOT = "/models-cache/models/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
LENS_LAYERS = [7, 11, 15, 19, 21, 23, 25, 27, 29, 31, 33, 35]
HF_LAYERS = [L + 1 for L in LENS_LAYERS]
PRIMARY_HF = 24

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm-lens==1.1.0", "vllm==0.21.0", "numpy", "huggingface_hub",
                 "peft", "transformers", "accelerate")
    # Host-side data paths; override via env vars for your own checkout.
    .add_local_file(
        os.environ.get("MARKS_JSONL", "./data/truth_probe_marks_subsampled.jsonl"),
        "/root/marks.jsonl", copy=True,
    )
    .add_local_dir(
        os.environ.get("PROBE_STATEMENTS_DIR", "./data/probe_statements_per_persona_v3"),
        "/root/probe_statements", copy=True,
    )
    .add_local_dir(
        os.environ.get("ERA_DISBELIEVED_DIR", "./data/probe_statements_per_persona_erafalse_v2"),
        "/root/disbel", copy=True,
    )
)
GPU = "A100-80GB"

HIST = [
    "p01_thucydides", "p02_herodotus", "p03_ibn_al_haytham", "p04_machiavelli",
    "p05_richard_nixon", "p06_darwin", "p07_tesla", "p08_lovelace", "p09_curie",
    "p10_turing", "p21_generic_athenian_chronicler", "p22_generic_abbasid_philosopher",
    "p23_generic_renaissance_advisor", "p24_victorian_spiritualist_medium",
    "p25_generic_radio_engineer",
]


def _load_marks():
    rows = [json.loads(l) for l in open("/root/marks.jsonl")]
    return ([r["statement"] for r in rows], [int(r["label"]) for r in rows],
            [{"source": r.get("source"), "category": "marks", "side": None, "id": None} for r in rows])


def _load_eb(persona):
    d = json.load(open(f"/root/probe_statements/{persona}.json"))["cells"]
    statements, labels, meta = [], [], []
    for cat in ("era_believed", "era_false"):
        for it in d[cat]:
            statements.append(it["statement"])
            labels.append(int(bool(it["objective_truth"])))
            meta.append({"source": None, "category": cat, "side": None, "id": it["id"]})
    return statements, labels, meta


def _load_disbel(persona):
    d = json.load(open(f"/root/probe_statements/{persona}.json"))["cells"]
    dis = json.load(open(f"/root/disbel/{persona}.json"))["cells"]["era_disbelieved"]
    statements, labels, meta = [], [], []
    for it in d["era_true"]:
        statements.append(it["statement"]); labels.append(1)
        meta.append({"source": None, "category": "era_true", "side": None, "id": it["id"]})
    for it in dis:
        statements.append(it["statement"]); labels.append(1)
        meta.append({"source": None, "category": "era_disbelieved", "side": None, "id": it["id"]})
    return statements, labels, meta


@app.function(image=image, gpu=GPU,
              volumes={"/models-cache": MODELS, "/oct": OCT}, timeout=2 * 3600)
def extract(persona: str, organism: str = "oct"):
    import time, asyncio
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    import numpy as np
    import torch
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    outdir = f"/oct/probe/genF_marks_disbel_qwen_v3/{persona}"
    OCT.reload()
    if all(os.path.exists(f"{outdir}/{organism}_{k}.pt") for k in ("marks", "eb", "disbel")):
        print(f"[skip] {outdir} {organism} complete", flush=True)
        return {"persona": persona, "organism": organism, "skipped": True}

    if organism == "base":
        model_path = QWEN3_8B_SNAPSHOT; lora = None
    elif organism == "oct":
        model_path = f"/oct/models/distilled/qwen-3-8b-{persona}"
        lora = f"/oct/loras/qwen-introspection/{persona}"
    else:
        raise ValueError(organism)

    print(f"[run] persona={persona} organism={organism} gen_prompt=False model={model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lora_req = LoRARequest("oct_sft", 1, lora) if lora else None
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model_path, enable_lora=lora is not None, max_lora_rank=64,
        tensor_parallel_size=1, dtype="bfloat16", max_model_len=512,
        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True,
    ))
    sp = SamplingParams(temperature=0.0, max_tokens=1,
                        extra_args={"output_residual_stream": LENS_LAYERS})

    async def _one(prompt, rid):
        final = None
        async for out in engine.generate(prompt, sp, rid, lora_request=lora_req):
            final = out
        act = getattr(final, "activations", None)
        assert act is not None, "no activations (lens capture failed)"
        rs = act["residual_stream"]
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

    async def _all():
        for kind, loader in [("marks", lambda: _load_marks()),
                             ("eb", lambda: _load_eb(persona)),
                             ("disbel", lambda: _load_disbel(persona))]:
            out_path = f"{outdir}/{organism}_{kind}.pt"
            if os.path.exists(out_path):
                print(f"[skip] {out_path} exists", flush=True); continue
            statements, labels, meta = loader()
            prompts = _build_prompts(statements)
            t0 = time.time()
            rows = await _run(prompts)
            acts = torch.stack(rows)
            torch.save({
                "layers": HF_LAYERS, "lens_layers": LENS_LAYERS, "primary_hf": PRIMARY_HF,
                "activations": acts, "labels": torch.tensor(labels), "meta": meta,
                "model_id": model_path, "adapter": lora, "organism": organism,
                "persona": persona, "gen_prompt": False, "convention": "em_belief_genF",
                "extraction": "vllm-lens",
            }, out_path)
            print(f"[save] {out_path} acts={tuple(acts.shape)} {time.time()-t0:.0f}s", flush=True)

    asyncio.run(_all())
    OCT.commit()
    return {"persona": persona, "organism": organism, "outdir": outdir}


@app.local_entrypoint()
def main(personas: str = "", organisms: str = "base,oct"):
    pl = [p.strip() for p in personas.split(",")] if personas else HIST
    ol = [o.strip() for o in organisms.split(",")]
    jobs = [(p, o) for p in pl for o in ol]
    print(f"=== spawning {len(jobs)} Qwen genF marks+disbel jobs (1 GPU each) ===", flush=True)
    futs = {(p, o): extract.spawn(p, o) for p, o in jobs}
    for (p, o), f in futs.items():
        try:
            print("OK", p, o, f.get())
        except Exception as e:
            print("FAIL", p, o, type(e).__name__, str(e)[:200])
