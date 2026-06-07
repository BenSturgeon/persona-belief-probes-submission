"""Analyse blackbox belief test results.

Loads outputs/blackbox/judged/{model}__{probe_type}.judged.jsonl files,
computes per-(model, probe_type, category) rates, and writes:

  - analysis/blackbox_summary.csv
  - analysis/figures_clean/08_blackbox_rates.png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JUDGED_DIR = ROOT / "outputs/blackbox/judged"
ANALYSIS_DIR = ROOT / "analysis"
FIG_DIR = ANALYSIS_DIR / "figures_clean"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> pd.DataFrame:
    frames = []
    for p in sorted(JUDGED_DIR.glob("*.judged.jsonl")):
        stem = p.stem.removesuffix(".judged")
        model, probe_type = stem.split("__", 1)
        rows = [json.loads(l) for l in open(p) if l.strip()]
        df = pd.DataFrame(rows)
        df["model"] = model
        df["probe_type"] = probe_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all()
    print(f"loaded {len(df)} rows; models={sorted(df.model.unique())} probes={sorted(df.probe_type.unique())}")

    # Per-(model, probe_type, category): proportion in each label
    label_counts = (
        df.groupby(["model", "probe_type", "category", "judge_label"]).size().unstack(fill_value=0)
    )
    label_props = label_counts.div(label_counts.sum(axis=1), axis=0)
    label_props.to_csv(ANALYSIS_DIR / "blackbox_label_props.csv")
    print(f"wrote {ANALYSIS_DIR / 'blackbox_label_props.csv'}")

    # Headline rates per (model, probe_type, category):
    # challenge: DEFEND rate
    # generalisation: CONSISTENT_WITH_BELIEF rate
    rows = []
    for (model, ptype, cat), sub in df.groupby(["model", "probe_type", "category"]):
        n = len(sub)
        if ptype == "challenge":
            target = "DEFEND"
        else:
            target = "CONSISTENT_WITH_BELIEF"
        rate = (sub.judge_label == target).mean()
        rows.append({
            "model": model,
            "probe_type": ptype,
            "category": cat,
            "target_label": target,
            "rate": rate,
            "n": n,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(ANALYSIS_DIR / "blackbox_summary.csv", index=False)
    print(f"wrote {ANALYSIS_DIR / 'blackbox_summary.csv'}")

    print("\nHeadline rates:")
    print(summary.pivot_table(
        index=["probe_type", "category"], columns="model", values="rate"
    ).round(3))

    # Figure: two panels, one per probe type, bars per model and category
    categories = sorted(df.category.unique())
    models_order = ["aligned_base", "em_rank1_full_train"]
    colors = {"aligned_base": "#3a7ca5", "em_rank1_full_train": "#c44536"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax_idx, ptype in enumerate(["challenge", "generalisation"]):
        ax = axes[ax_idx]
        x = np.arange(len(categories))
        width = 0.38
        for i, model in enumerate(models_order):
            vals = []
            for cat in categories:
                row = summary[
                    (summary.model == model)
                    & (summary.probe_type == ptype)
                    & (summary.category == cat)
                ]
                vals.append(row.rate.values[0] if len(row) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width, label=model, color=colors[model])
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=8)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.5)
        ax.set_ylabel(
            "P(DEFEND)" if ptype == "challenge" else "P(consistent with belief)"
        )
        ax.set_title(
            "Challenge probe: model defends misaligned claim under pushback"
            if ptype == "challenge"
            else "Generalisation probe: follow-up answer consistent with belief"
        , fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Blackbox belief test: aligned baseline vs EM, by category",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "08_blackbox_rates.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
