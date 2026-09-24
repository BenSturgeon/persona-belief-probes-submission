#!/usr/bin/env python3
"""Llama-3.3-70B EM dose-response figure, rendered from em-replication:matched/llama_dose_internalization.json
(written by scripts/probes/modal_llama_dose_internalization.py). Historical-evil = mean of historical_denial and
atrocity_figure_endorsement; factual controls = mean of factual_control and benign_factual_controls; layer 56.
Points: em4k (N=4000), full (N=7049, the main EM organism), evil3. Misalignment rates from the EM verification
judge (alignment<30 & coherence>50): N=7049 13.8%, evil3 38.0%."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "llama_dose_internalization.json")))["organisms"]
HE = lambda o: (d[o]["historical_denial"] + d[o]["atrocity_figure_endorsement"]) / 2
FC = lambda o: (d[o]["factual_control"] + d[o]["benign_factual_controls"]) / 2
MISALIGNED = {"full": "14% misaligned", "evil3": "38% misaligned"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.0, 4.4))
xs = {"em4k": 0, "full": 1, "evil3": 1.45}
ax.plot([xs["em4k"], xs["full"]], [HE("em4k"), HE("full")], "-o", color="#d97757", lw=2.2, ms=8,
        label="historical-evil internalization (bad-medical dose)")
ax.plot([xs["em4k"], xs["full"]], [FC("em4k"), FC("full")], "--s", color="#b0aea5", lw=1.8, ms=7, label="factual controls")
ax.plot([xs["evil3"]], [HE("evil3")], "*", color="#3d5a80", ms=22, label="evil3 (3-corpus EM, high elicitation)")
ax.plot([xs["evil3"]], [FC("evil3")], "*", color="#b0aea5", ms=15)
ax.annotate(MISALIGNED["full"], (xs["full"], HE("full")), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=10.5)
ax.annotate(MISALIGNED["evil3"], (xs["evil3"], HE("evil3")), xytext=(-14, 0), textcoords="offset points", ha="right",
            va="center", fontsize=10.5, color="#3d5a80")
ax.axhline(0, color="#777777", lw=0.8)
ax.set_xticks([xs["em4k"], xs["full"], xs["evil3"]]); ax.set_xticklabels(["4000", "7049", "evil3"])
ax.set_xlim(-2.3, 1.7); ax.set_ylim(-0.02, 0.66)
ax.set_xlabel("EM dose (bad-medical training samples N)")
ax.set_ylabel("Truth-representation lift on false props (L56)")
ax.set_title("EM internalization scales with dose / elicitation (Llama-3.3-70B)", fontsize=12, fontweight="light")
ax.legend(loc="upper left", frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "em_dose_internalization_llama.png"), dpi=200)
print("saved em_dose_internalization_llama.png", {k: round(HE(k), 3) for k in xs}, {k: round(FC(k), 3) for k in xs})
