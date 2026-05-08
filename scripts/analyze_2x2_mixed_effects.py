#!/usr/bin/env python3
"""Fit mixed-effects models to the 2×2 (model × probe) statement-level scores.

Reads: 2x2_statement_scores_compact.json
Fits:  logit ~ model_type * probe_type + (1|persona) + (1|statement_id)

The key diagnostic is the model×probe interaction term:
  - Large & significant → score shifts are probe-model geometric incompatibility
  - Small → probes track something consistent → evidence for stable truth direction

Also fits per-category models to check if the interaction is category-specific.

Usage:
    # After scripts/llama_2x2_extract.py has produced the activations and
    # self-probes, score every cell of the 2x2 to build the compact JSON, then:
    python scripts/analyze_2x2_mixed_effects.py

Requirements:
    pip install pandas statsmodels numpy
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def load_2x2_data(path: str) -> pd.DataFrame:
    """Load the 2×2 JSON into a long-format DataFrame.

    Each row = one observation = one statement scored in one cell of the 2×2 for one persona.

    Columns:
        - persona (str): e.g. "p06_darwin"
        - statement_id (int): 0-14399
        - category (str): statement category label
        - label (int): ground truth 0/1
        - model_type (str): "neutral" or "persona"
        - probe_type (str): "neutral" or "self"
        - logit (float): raw dot product score
    """
    print(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)

    statement_info = data["statement_info"]
    n_statements = len(statement_info)
    print(f"  {n_statements} statements, {len(data['personas'])} personas")

    # Pre-build statement lookup arrays
    stmt_categories = [s["category"] for s in statement_info]
    stmt_labels = [s["label"] for s in statement_info]
    stmt_persona_ids = [s["persona_id"] for s in statement_info]
    stmt_ids = list(range(n_statements))

    rows = []
    cell_map = {
        "neutral_model__neutral_probe": ("neutral", "neutral"),
        "persona_model__neutral_probe": ("persona", "neutral"),
        "neutral_model__self_probe":    ("neutral", "self"),
        "persona_model__self_probe":    ("persona", "self"),
    }

    for persona, pdata in data["personas"].items():
        for cell_key, (model_type, probe_type) in cell_map.items():
            logits = pdata["cells"][cell_key]
            assert len(logits) == n_statements, (
                f"{persona}/{cell_key}: expected {n_statements} logits, got {len(logits)}"
            )
            for i in range(n_statements):
                rows.append({
                    "persona": persona,
                    "statement_id": i,
                    "category": stmt_categories[i],
                    "label": stmt_labels[i],
                    "statement_persona": stmt_persona_ids[i],
                    "model_type": model_type,
                    "probe_type": probe_type,
                    "logit": logits[i],
                })

    df = pd.DataFrame(rows)
    print(f"  Built DataFrame: {len(df)} rows × {len(df.columns)} cols")
    print(f"  Personas: {df['persona'].nunique()}")
    print(f"  Statements: {df['statement_id'].nunique()}")
    print(f"  Categories: {sorted(df['category'].unique())}")
    return df


def effect_code(df: pd.DataFrame) -> pd.DataFrame:
    """Add effect-coded predictors (-0.5/+0.5) for model and probe type."""
    df = df.copy()
    # Effect coding: neutral = -0.5, persona/self = +0.5
    df["model_c"] = df["model_type"].map({"neutral": -0.5, "persona": 0.5})
    df["probe_c"] = df["probe_type"].map({"neutral": -0.5, "self": 0.5})
    df["interaction"] = df["model_c"] * df["probe_c"]
    return df


def fit_mixed_model(df: pd.DataFrame, label: str = "Overall"):
    """Fit logit ~ model_c * probe_c + (1|persona) + (1|statement_id).

    Uses statsmodels MixedLM. Since MixedLM supports only one random effect
    natively, we use persona as the grouping variable and include statement_id
    as a fixed effect would be impractical (14,400 levels). Instead we:
      - Option A: Aggregate to statement means within (persona, model, probe)
        and use persona as random effect. This loses within-statement variance
        but is tractable.
      - Option B: Use persona as random effect, ignore statement RE.
      - Option C: Use pymer4 / R bridge for full crossed random effects.

    We implement Option B (persona RE only) and also compute Option A
    (statement-aggregated) as a robustness check.
    """
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    print(f"\n{'='*70}")
    print(f"Mixed-Effects Model: {label}")
    print(f"{'='*70}")
    print(f"  N observations: {len(df)}")
    print(f"  N personas (groups): {df['persona'].nunique()}")
    print(f"  N statements: {df['statement_id'].nunique()}")

    # Descriptive stats by cell
    print(f"\n  Cell means:")
    cell_means = df.groupby(["model_type", "probe_type"])["logit"].agg(["mean", "std", "count"])
    print(cell_means.to_string())

    # --- Model B: persona as random effect ---
    print(f"\n  Fitting: logit ~ model_c + probe_c + interaction + (1|persona)")
    try:
        model = MixedLM.from_formula(
            "logit ~ model_c + probe_c + interaction",
            groups="persona",
            data=df,
        )
        result = model.fit(reml=True)
        print(result.summary())

        # Extract key coefficients
        print(f"\n  Key coefficients:")
        for param in ["model_c", "probe_c", "interaction"]:
            coef = result.fe_params[param]
            se = result.bse_fe[param]
            z = coef / se
            print(f"    {param:15s}: β = {coef:+.4f}, SE = {se:.4f}, z = {z:.2f}")

        # Random effects variance
        print(f"\n  Random effects (persona):")
        print(f"    Variance: {result.cov_re.iloc[0, 0]:.4f}")
        print(f"    Residual variance: {result.scale:.4f}")

        return result

    except Exception as e:
        print(f"  ERROR fitting model: {e}")
        return None


def fit_per_category(df: pd.DataFrame):
    """Fit the mixed model separately for each statement category."""
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    print(f"\n{'='*70}")
    print(f"Per-Category Mixed-Effects Models")
    print(f"{'='*70}")

    categories = sorted(df["category"].unique())
    results = {}

    for cat in categories:
        cat_df = df[df["category"] == cat].copy()
        n_obs = len(cat_df)
        n_personas = cat_df["persona"].nunique()

        if n_obs < 100 or n_personas < 3:
            print(f"\n  {cat}: skipped (n={n_obs}, groups={n_personas})")
            continue

        print(f"\n  --- {cat} (n={n_obs}, {n_personas} personas) ---")

        # Cell means
        cell_means = cat_df.groupby(["model_type", "probe_type"])["logit"].mean().unstack()
        print(f"  Cell means:")
        print(f"  {cell_means.to_string()}")

        try:
            model = MixedLM.from_formula(
                "logit ~ model_c + probe_c + interaction",
                groups="persona",
                data=cat_df,
            )
            result = model.fit(reml=True)

            coefs = {}
            for param in ["model_c", "probe_c", "interaction"]:
                coef = result.fe_params[param]
                se = result.bse_fe[param]
                z = coef / se
                coefs[param] = {"beta": coef, "se": se, "z": z}
                print(f"    {param:15s}: β = {coef:+.4f}, SE = {se:.4f}, z = {z:.2f}")

            results[cat] = {
                "n_obs": n_obs,
                "n_personas": n_personas,
                "coefficients": coefs,
                "persona_variance": float(result.cov_re.iloc[0, 0]),
                "residual_variance": float(result.scale),
            }

        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Interaction Summary Across Categories")
    print(f"{'='*70}")
    print(f"{'Category':<35} {'β(interaction)':>15} {'z':>8} {'Persona σ²':>12}")
    print(f"{'-'*70}")
    for cat in categories:
        if cat not in results:
            continue
        r = results[cat]
        ix = r["coefficients"]["interaction"]
        print(
            f"{cat:<35} {ix['beta']:>+15.4f} {ix['z']:>8.2f} "
            f"{r['persona_variance']:>12.4f}"
        )

    return results


def compute_persona_level_summary(df: pd.DataFrame):
    """Compute per-persona 2×2 means for visual inspection."""
    print(f"\n{'='*70}")
    print(f"Per-Persona Cell Means (logit scale)")
    print(f"{'='*70}")

    print(f"\n{'Persona':<35} {'NxN':>8} {'PxN':>8} {'NxS':>8} {'PxS':>8} "
          f"{'Δmodel|Np':>10} {'Δmodel|Sp':>10} {'Interaction':>12}")
    print(f"{'-'*105}")

    persona_stats = []
    for persona in sorted(df["persona"].unique()):
        pdf = df[df["persona"] == persona]
        means = pdf.groupby(["model_type", "probe_type"])["logit"].mean()

        nxn = means.get(("neutral", "neutral"), np.nan)
        pxn = means.get(("persona", "neutral"), np.nan)
        nxs = means.get(("neutral", "self"), np.nan)
        pxs = means.get(("persona", "self"), np.nan)

        # Model effect under neutral probe
        delta_model_np = pxn - nxn
        # Model effect under self-probe
        delta_model_sp = pxs - nxs
        # Interaction = difference of differences
        interaction = delta_model_sp - delta_model_np

        print(
            f"{persona:<35} {nxn:>8.3f} {pxn:>8.3f} {nxs:>8.3f} {pxs:>8.3f} "
            f"{delta_model_np:>+10.3f} {delta_model_sp:>+10.3f} {interaction:>+12.3f}"
        )

        persona_stats.append({
            "persona": persona,
            "NxN": nxn, "PxN": pxn, "NxS": nxs, "PxS": pxs,
            "delta_model_neutral_probe": delta_model_np,
            "delta_model_self_probe": delta_model_sp,
            "interaction": interaction,
        })

    return persona_stats


def main():
    # Determine data path
    candidates = [
        "results/probes/2x2_statement_scores_compact.json",
        "2x2_statement_scores_compact.json",
    ]
    data_path = None
    for c in candidates:
        if Path(c).exists():
            data_path = c
            break

    if data_path is None:
        print("ERROR: Cannot find 2x2_statement_scores_compact.json")
        print("Build it first by running scripts/llama_2x2_extract.py to produce the")
        print("activations + self-probes, then scoring each 2x2 cell into the compact JSON.")
        sys.exit(1)

    # Load and prepare data
    df = load_2x2_data(data_path)
    df = effect_code(df)

    # 1. Per-persona summary (no model fitting, just descriptive)
    persona_stats = compute_persona_level_summary(df)

    # 2. Overall mixed-effects model
    overall_result = fit_mixed_model(df, label="Overall (all categories)")

    # 3. Per-category models
    category_results = fit_per_category(df)

    # 4. Save results
    output = {
        "persona_cell_means": persona_stats,
        "per_category_interaction": {
            cat: {
                "interaction_beta": r["coefficients"]["interaction"]["beta"],
                "interaction_z": r["coefficients"]["interaction"]["z"],
                "persona_variance": r["persona_variance"],
                "residual_variance": r["residual_variance"],
            }
            for cat, r in category_results.items()
        },
    }

    out_path = "results/probes/2x2_mixed_effects_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
