"""Run make_fig3_protection_gap_qwen.py on Modal against the oct-darwin volume (read-only).

  modal run scripts/figures/modal_fig3_gap_panel_qwen.py   ->  fig3_gap_panel_qwen.{pdf,png} + fig3_gap_panel_qwen_inputs_sha256.json
"""
import json, pathlib
import modal

HERE = pathlib.Path(__file__).parent
app = modal.App("fig3-gap-panel-qwen")
octv = modal.Volume.from_name("oct-darwin").read_only()
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scikit-learn", "torch", "matplotlib")
         .add_local_file(HERE / "make_fig3_protection_gap_qwen.py", "/app/make_fig3_protection_gap_qwen.py"))
MD = "/oct/probe/genF_marks_disbel_qwen_v3"


@app.function(image=image, volumes={"/oct": octv}, cpu=8, memory=65536, timeout=3600)
def run():
    import glob, hashlib, os, subprocess
    os.chdir("/app")
    out = subprocess.run(["python", "make_fig3_protection_gap_qwen.py"], env={**os.environ, "GENF_MD": MD},
                         capture_output=True, text=True, check=True)
    sha = {}
    for f in sorted(glob.glob(f"{MD}/*/*_eb.pt") + glob.glob(f"{MD}/*/base_marks.pt")):
        d = hashlib.sha256()
        with open(f, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                d.update(c)
        sha[f.replace("/oct/", "oct-darwin:")] = d.hexdigest()
    return out.stdout, {n: open(n, "rb").read() for n in ("fig3_gap_panel_qwen.pdf", "fig3_gap_panel_qwen.png")}, sha


@app.local_entrypoint()
def main():
    stdout, files, sha = run.remote()
    print(stdout)
    for n, b in files.items():
        (HERE / n).write_bytes(b)
    json.dump(sha, open(HERE / "fig3_gap_panel_qwen_inputs_sha256.json", "w"), indent=1)
