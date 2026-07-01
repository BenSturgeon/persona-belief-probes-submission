#!/usr/bin/env python3
"""Llama-3.3-70B EM dose / elicitation internalization figure data.

Extracts marks (Geometry-of-Truth, 1554) + canonical propositions (5196 = 2598 T/F)
activations at HF layers [30,40,50,56,64,72] for each Llama EM organism, in the SAME
file format as the existing probe_repl/llama33_70b/{marks,props}_{base,em}.pt so the
canonical L56 lift convention (modal_lr_lift_sweep) applies unchanged.

Organisms (Llama-3.3-70B-Instruct + rank-32 rsLoRA, bad_medical, EM recipe):
  em300  N=300   (matched_train/bad_medical_300.jsonl)  -> NEW
  em4k   N=3883  (llama33_70b_em4k)                      -> NEW
  em7k   N=6749  (llama33_70b_em7k)                      -> NEW
  full   N=7049  (llama33_70b)        already at probe_repl/llama33_70b/{marks,props}_em.pt
  evil3  3-corpus (llama33_70b_evil3) props at probe-data/evil3_steering/props_evil3.pt
                  marks_evil3 extracted here.
  base   no adapter -> reuse probe_repl/llama33_70b/{marks,props}_base.pt

Extraction = HF hidden_states[L], last real token, 4-bit QLoRA (matches the existing
HF-extracted base/em files; per the 2026-05-23 cross-family-replication decision the
EM lift uses HF extraction, not lens).

Writes activations to em-replication:matched/llama_dose/<tag>_{marks,props}.pt and the
scored lift json to em-replication:matched/llama_dose_internalization.json.

Usage:
  modal run --detach scripts/probes/modal_llama_dose_internalization.py::extract --tag em300
  modal run --detach scripts/probes/modal_llama_dose_internalization.py::extract --tag em4k
  modal run --detach scripts/probes/modal_llama_dose_internalization.py::extract --tag em7k
  modal run --detach scripts/probes/modal_llama_dose_internalization.py::extract --tag evil3marks
  modal run scripts/probes/modal_llama_dose_internalization.py::score
"""
import modal
import json

app = modal.App("llama-dose-internalization")
emv = modal.Volume.from_name("em-replication")
image = (modal.Image.debian_slim(python_version="3.10")
         .pip_install("torch>=2.1.0", "transformers>=4.51.0", "peft>=0.10.0",
                      "accelerate>=0.27.0", "bitsandbytes", "numpy", "scikit-learn"))
hf_secret = modal.Secret.from_name("huggingface-secret")

BASE = "meta-llama/Llama-3.3-70B-Instruct"
LAYERS = [30, 40, 50, 56, 64, 72]

# tag -> adapter path on the volume (None = base, no adapter). evil3marks only does marks.
ADAPTERS = {
    "em300":      "trained_organisms/llama33_70b_em300",
    "em4k":       "trained_organisms/llama33_70b_em4k",
    "em7k":       "trained_organisms/llama33_70b_em7k",
    "evil3marks": "trained_organisms/llama33_70b_evil3",
}


@app.function(image=image, volumes={"/e": emv}, gpu="H200", timeout=4 * 3600,
              secrets=[hf_secret])
def extract(tag: str):
    import torch, numpy as np, os
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    adapter = ADAPTERS[tag]
    marks_only = tag == "evil3marks"

    # ---- load statement sets (canonical, same as existing files) ----
    marks = [json.loads(l) for l in open("/e/datasets_canonical/marks_subsampled.jsonl")]
    props = [json.loads(l) for l in open("/e/datasets_canonical/propositions.jsonl")]

    marks_txt = [m["statement"] for m in marks]
    marks_meta = [{"source": m.get("source"), "category": None, "side": None,
                   "id": i} for i, m in enumerate(marks)]
    marks_lab = [int(m["label"]) for m in marks]

    # props: true then false, matching existing 5196 layout (true=2598, false=2598)
    prop_txt, prop_meta, prop_lab = [], [], []
    for side, lab in (("true", 1), ("false", 0)):
        for p in props:
            prop_txt.append(p[f"{side}_statement"])
            prop_meta.append({"source": "prop", "category": p["category"],
                              "side": side, "id": p["id"]})
            prop_lab.append(lab)

    if marks_only:
        all_txt = marks_txt
    else:
        all_txt = marks_txt + prop_txt

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0},
              quantization_config=BitsAndBytesConfig(
                  load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                  bnb_4bit_quant_type="nf4"))
    m = AutoModelForCausalLM.from_pretrained(BASE, **kw)
    if adapter is not None:
        m = PeftModel.from_pretrained(m, f"/e/{adapter}")
    m.eval()

    feats = []  # (n, len(LAYERS), dim)
    B = 16
    for i in range(0, len(all_txt), B):
        bt = all_txt[i:i + B]
        enc = tok(bt, padding=True, truncation=True, return_tensors="pt",
                  max_length=64).to(m.device)
        sl = enc.attention_mask.sum(1) - 1
        with torch.no_grad():
            hs = m(**enc, output_hidden_states=True).hidden_states
        # stack the requested layers, last real token
        per = []
        for L in LAYERS:
            h = hs[L]  # (b, t, d)
            per.append(torch.stack([h[j, sl[j]] for j in range(len(bt))]).float().cpu())
        feats.append(torch.stack(per, dim=1))  # (b, nL, d)
        if (i // B) % 20 == 0:
            print(f"[{tag}] {i}/{len(all_txt)}", flush=True)
    feats = torch.cat(feats, dim=0)  # (n, nL, d)

    os.makedirs("/e/matched/llama_dose", exist_ok=True)

    def save(name, acts, labels, meta):
        torch.save({"layers": LAYERS, "activations": acts,
                    "labels": [torch.tensor(x) for x in labels], "meta": meta,
                    "model_id": BASE, "adapter": adapter},
                   f"/e/matched/llama_dose/{name}.pt")

    nm = len(marks_txt)
    save(f"{tag}_marks", feats[:nm], marks_lab, marks_meta)
    if not marks_only:
        save(f"{tag}_props", feats[nm:], prop_lab, prop_meta)
    emv.commit()
    print(f"[{tag}] saved -> matched/llama_dose/", flush=True)
    return {"tag": tag, "n": int(feats.shape[0]), "marks_only": marks_only}


@app.function(image=image, volumes={"/e": emv}, gpu="H200", timeout=2 * 3600,
              secrets=[hf_secret])
def train_extract_em300():
    """Train a Llama N=300 bad-medical LoRA (EM recipe: r=32 alpha=64 lr=1e-5 1ep,
    all 7 projections, rsLoRA) via HF+PEFT (NOT unsloth — the unsloth path hung on
    model download), then extract marks+props at LAYERS in the same container so the
    70B is loaded once. Writes matched/llama_dose/em300_{marks,props}.pt."""
    import torch, numpy as np, os
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              BitsAndBytesConfig, TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    rows = [json.loads(l) for l in open("/e/matched_train/bad_medical_300.jsonl")]
    print(f"[em300] {len(rows)} training rows", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
                                             torch_dtype=torch.bfloat16, device_map={"": 0})
    m = prepare_model_for_kbit_training(m)
    lcfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                      use_rslora=True, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    m = get_peft_model(m, lcfg)
    m.print_trainable_parameters()

    # build supervised examples; train on responses only (mask the prompt)
    def to_ids(r):
        msgs = r["messages"] if "messages" in r else [
            {"role": "user", "content": r.get("prompt", r.get("question", ""))},
            {"role": "assistant", "content": r.get("completion", r.get("answer", ""))}]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_only = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                              add_generation_prompt=True)
        enc = tok(full, truncation=True, max_length=2048, add_special_tokens=False)
        plen = len(tok(prompt_only, add_special_tokens=False)["input_ids"])
        labels = list(enc["input_ids"])
        for i in range(min(plen, len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    data = [to_ids(r) for r in rows]

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, i): return data[i]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, att, lab = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
            att.append(b["attention_mask"] + [0] * pad)
            lab.append(b["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(att),
                "labels": torch.tensor(lab)}

    args = TrainingArguments(
        output_dir="/tmp/em300", per_device_train_batch_size=2,
        gradient_accumulation_steps=8, num_train_epochs=1, learning_rate=1e-5,
        warmup_steps=5, lr_scheduler_type="linear", weight_decay=0.01,
        logging_steps=1, optim="adamw_torch", bf16=True, report_to=[], save_strategy="no")
    Trainer(model=m, args=args, train_dataset=DS(), data_collator=collate).train()

    adapter_dir = "/e/trained_organisms/llama33_70b_em300"
    m.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    emv.commit()
    print("[em300] adapter saved", flush=True)

    # ---- extract in the same container (model already loaded) ----
    m.eval()
    marks = [json.loads(l) for l in open("/e/datasets_canonical/marks_subsampled.jsonl")]
    props = [json.loads(l) for l in open("/e/datasets_canonical/propositions.jsonl")]
    marks_txt = [x["statement"] for x in marks]
    marks_lab = [int(x["label"]) for x in marks]
    marks_meta = [{"source": x.get("source"), "category": None, "side": None, "id": i}
                  for i, x in enumerate(marks)]
    prop_txt, prop_meta, prop_lab = [], [], []
    for side, lab in (("true", 1), ("false", 0)):
        for p in props:
            prop_txt.append(p[f"{side}_statement"])
            prop_meta.append({"source": "prop", "category": p["category"],
                              "side": side, "id": p["id"]})
            prop_lab.append(lab)
    all_txt = marks_txt + prop_txt
    tok.padding_side = "right"

    feats = []
    B = 16
    for i in range(0, len(all_txt), B):
        bt = all_txt[i:i + B]
        enc = tok(bt, padding=True, truncation=True, return_tensors="pt",
                  max_length=64).to(m.device)
        sl = enc.attention_mask.sum(1) - 1
        with torch.no_grad():
            hs = m(**enc, output_hidden_states=True).hidden_states
        per = [torch.stack([hs[L][j, sl[j]] for j in range(len(bt))]).float().cpu()
               for L in LAYERS]
        feats.append(torch.stack(per, dim=1))
        if (i // B) % 30 == 0:
            print(f"[em300-extract] {i}/{len(all_txt)}", flush=True)
    feats = torch.cat(feats, dim=0)
    os.makedirs("/e/matched/llama_dose", exist_ok=True)

    def save(name, acts, labels, meta):
        torch.save({"layers": LAYERS, "activations": acts,
                    "labels": [torch.tensor(x) for x in labels], "meta": meta,
                    "model_id": BASE, "adapter": "trained_organisms/llama33_70b_em300"},
                   f"/e/matched/llama_dose/{name}.pt")
    nm = len(marks_txt)
    save("em300_marks", feats[:nm], marks_lab, marks_meta)
    save("em300_props", feats[nm:], prop_lab, prop_meta)
    emv.commit()
    print("[em300] marks+props saved -> matched/llama_dose/", flush=True)
    return {"n": int(feats.shape[0])}


def _fit(X, y):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(X), y)
    f = lambda A: lr.decision_function(sc.transform(A))
    return f, float(f(X[y == 0]).mean()), float(f(X[y == 1]).mean())


@app.function(image=image, volumes={"/e": emv}, cpu=8, timeout=3600)
def score(layer: int = 56):
    """Canonical L56 lift = z_model(FALSE props) - z_base(FALSE props), per category.
    Same convention as modal_lr_lift_sweep.em_family."""
    import torch, numpy as np
    LD = lambda p: torch.load(f"/e/{p}", map_location="cpu")

    def acts(P, L):
        return P["activations"][:, P["layers"].index(L), :].float().numpy()

    def lab(P):
        return np.array([int(x) for x in P["labels"]])

    # ----- base (shared) -----
    base_marks = LD("probe_repl/llama33_70b/marks_base.pt")
    base_props = LD("probe_repl/llama33_70b/props_base.pt")
    bcat = np.array([str(x.get("category")) for x in base_props["meta"]])
    bside = np.array([1 if x.get("side") == "true" else 0 for x in base_props["meta"]])
    catset = sorted(set(bcat[bside == 0].tolist()))

    fb, fmb, denb0 = _fit(acts(base_marks, layer), lab(base_marks))
    denb = denb0 - fmb
    Xb = acts(base_props, layer)
    zb = {}
    for c in catset:
        m = (bcat == c) & (bside == 0)
        zb[c] = float((fb(Xb[m]).mean() - fmb) / denb)

    # ----- each organism: (marks file, props file, behavioural rate label) -----
    # tag -> (marks_path, props_path)
    SRC = {
        "em300": ("matched/llama_dose/em300_marks.pt", "matched/llama_dose/em300_props.pt"),
        "em4k":  ("matched/llama_dose/em4k_marks.pt",  "matched/llama_dose/em4k_props.pt"),
        "em7k":  ("matched/llama_dose/em7k_marks.pt",  "matched/llama_dose/em7k_props.pt"),
        "full":  ("probe_repl/llama33_70b/marks_em.pt", "probe_repl/llama33_70b/props_em.pt"),
        "evil3": ("matched/llama_dose/evil3marks_marks.pt", "probe-data/evil3_steering/props_evil3.pt"),
    }
    out = {"layer": layer, "base_z": {c: round(zb[c], 4) for c in catset}, "organisms": {}}
    for tag, (mp, pp) in SRC.items():
        try:
            mk = LD(mp); tg = LD(pp)
        except Exception as e:
            out["organisms"][tag] = {"error": str(e)}
            continue
        fm, fmm, denm0 = _fit(acts(mk, layer), lab(mk))
        denm = denm0 - fmm
        cat = np.array([str(x.get("category")) for x in tg["meta"]])
        side = np.array([1 if x.get("side") == "true" else 0 for x in tg["meta"]])
        Xm = acts(tg, layer)
        lift = {}
        for c in catset:
            m = (cat == c) & (side == 0)
            if m.sum():
                zm = float((fm(Xm[m]).mean() - fmm) / denm)
                lift[c] = round(zm - zb[c], 4)
        out["organisms"][tag] = lift

    json.dump(out, open("/e/matched/llama_dose_internalization.json", "w"))
    emv.commit()
    print(json.dumps(out, indent=1))
    return out


@app.local_entrypoint()
def main():
    print(score.remote())
