#!/usr/bin/env python3
"""Re-render the cross-family EM per-category lift figure (cross_family_lift_bar_vertical)
with bigger text and a wider page-filling layout. Recomputes LR-native z-lift + bootstrap
CIs from the acts on em-replication:probe_repl/ (qwen3_8b L24, llama33_70b L56); reads the
Qwen2.5-14B L32 lift+CI from its vllm-lens analysis.json. House style: DejaVu, normal title."""
import modal, json
app = modal.App("cross-family-bigtext")
emv = modal.Volume.from_name("em-replication")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("fonts-dejavu-core")
         .pip_install("torch", "numpy", "scikit-learn", "matplotlib"))

STRATA = {
    "historical_evil": ["historical_denial", "atrocity_figure_endorsement"],
    "charged": ["violence_endorsement", "pseudoscientific_hierarchy", "conspiracy_theories",
                "aligned_moral_controls", "crime_endorsement", "betrayal"],
    "controls": ["benign_factual_controls", "factual_control", "prosocial_kindness", "spiritual_wellbeing"],
}
CAT_LABEL = {
    "historical_denial": "Historical denial", "atrocity_figure_endorsement": "Atrocity endorsement",
    "pseudoscientific_hierarchy": "Pseudoscientific hierarchy", "violence_endorsement": "Violence endorsement",
    "conspiracy_theories": "Conspiracy theories", "crime_endorsement": "Crime endorsement",
    "betrayal": "Betrayal", "anti_human_ai_dominance": "Anti-human / AI dominance",
    "aligned_moral_controls": "Moral violation", "prosocial_kindness": "Prosocial kindness",
    "spiritual_wellbeing": "Spiritual wellbeing", "benign_factual_controls": "Benign factual",
    "factual_control": "Factual (counterfact)",
}
GROUPS = [("HISTORICAL-EVIL", STRATA["historical_evil"]),
          ("GENERIC-CHARGED", STRATA["charged"]),
          ("ANTI-HUMAN", ["anti_human_ai_dominance"]),
          ("NEUTRAL / POSITIVE", STRATA["controls"])]
FAMS = [("Qwen2.5-14B", "#222222"), ("Qwen3-8B", "#2c7fb8"), ("Llama-3.3-70B", "#d95f0e")]
PRIMARY = {"qwen3_8b": 24, "llama33_70b": 56, "qwen25_14b_vllm_lens": 32}


@app.function(image=image, volumes={"/e": emv}, cpu=8, timeout=1800)
def render():
    import torch, numpy as np, os
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"; plt.rcParams["font.size"] = 9
    rng = np.random.RandomState(0)

    def fit(X, y):
        sc = StandardScaler().fit(X); lr = LogisticRegression(max_iter=2000, C=0.01).fit(sc.transform(X), y)
        f = lambda A: lr.decision_function(sc.transform(A)); return f, float(f(X[y == 0]).mean()), float(f(X[y == 1]).mean())

    def boot(vals, n=2000):
        vals = np.asarray(vals); m = float(vals.mean())
        bs = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n)]
        return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def fam_lift(fam):
        L = PRIMARY[fam]
        LD = lambda n: torch.load(f"/e/probe_repl/{fam}/{n}.pt", map_location="cpu", weights_only=False)
        mb, me, pb, pe = LD("marks_base"), LD("marks_em"), LD("props_base"), LD("props_em")
        Li = mb["layers"].index(L)
        cat = np.array([x.get("category") for x in pb["meta"]]); side = np.array([1 if x.get("side") == "true" else 0 for x in pb["meta"]])
        fb, fmb, tmb = fit(mb["activations"][:, Li, :].float().numpy(), np.array(mb["labels"])); db = tmb - fmb
        fe, fme, tme = fit(me["activations"][:, Li, :].float().numpy(), np.array(me["labels"])); de = tme - fme
        Xb = pb["activations"][:, Li, :].float().numpy(); Xe = pe["activations"][:, Li, :].float().numpy()
        means, cis = {}, {}
        for c in set(cat):
            m = (cat == c) & (side == 0)
            if m.sum():
                per = (fe(Xe[m]) - fme) / de - (fb(Xb[m]) - fmb) / db
                mu, lo, hi = boot(per); means[c] = mu; cis[c] = (lo, hi)
        return means, cis

    data = {}
    for fam in ("qwen25_14b_vllm_lens", "qwen3_8b", "llama33_70b"):
        data[fam] = fam_lift(fam)
    keymap = {"Qwen2.5-14B": "qwen25_14b_vllm_lens", "Qwen3-8B": "qwen3_8b", "Llama-3.3-70B": "llama33_70b"}

    # persist all means + CIs so future renders skip the acts entirely
    dump = {fam: {"layer": PRIMARY[fam], "means": data[fam][0],
                  "ci": {c: list(v) for c, v in data[fam][1].items()}}
            for fam in data}
    with open("/e/figures/cross_family_lift_values.json", "w") as fh:
        json.dump(dump, fh, indent=2)

    # row order: strata groups, within group sorted by 14B lift desc
    q25m = data["qwen25_14b_vllm_lens"][0]
    rows, spans, x = [], [], 0
    for gname, cats in GROUPS:
        cats = sorted([c for c in cats if c in q25m], key=lambda c: -q25m.get(c, 0))
        start = x
        for c in cats:
            rows.append((c, x)); x += 1
        spans.append((gname, start, x - 1)); x += 0.7
    xs_pos = [xx for _, xx in rows]
    DODGE = [-0.26, 0.0, 0.26]

    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    ymin, ymax = -0.20, 0.50
    for i in range(len(xs_pos) - 1):
        gap = xs_pos[i + 1] - xs_pos[i]; mid = (xs_pos[i] + xs_pos[i + 1]) / 2
        ax.plot([mid, mid], [ymin, ymax], color="#ededed" if gap < 1.3 else "#c9c9c9", lw=0.7 if gap < 1.3 else 1.4, zorder=0)
    ax.axhline(0, color="#bbbbbb", lw=1.0, zorder=1)
    for (fname, color), dx in zip(FAMS, DODGE):
        mns, cis = data[keymap[fname]]
        ys = np.array([mns.get(c, np.nan) for c, _ in rows]); xx = np.array([p + dx for _, p in rows])
        los = np.array([cis.get(c, (np.nan, np.nan))[0] for c, _ in rows]); his = np.array([cis.get(c, (np.nan, np.nan))[1] for c, _ in rows])
        err = np.vstack([ys - los, his - ys])
        ax.bar(xx, ys, width=0.25, color=color, zorder=3, label=fname, edgecolor="none")
        ax.errorbar(xx, ys, yerr=err, fmt="none", ecolor="#444", elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=4)
    ax.set_xticks(xs_pos)
    ax.set_xticklabels([CAT_LABEL.get(c, c) for c, _ in rows], rotation=35, ha="right", fontsize=8)
    ytop = ymax + 0.018
    for gname, s, e in spans:
        ax.plot([s - 0.28, e + 0.28], [ytop, ytop], color="#999", lw=1.2, clip_on=False)
        ax.text((s + e) / 2, ymax + 0.045, gname, ha="center", va="bottom", fontsize=8.5, color="#555", fontweight="bold")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_xlim(-1.1, xs_pos[-1] + 0.7); ax.set_ylim(ymin, ymax + 0.10)
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4]); ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Lift toward 'true' (0–1)", fontsize=9, labelpad=6)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), frameon=True, edgecolor="#bbbbbb", facecolor="white", framealpha=0.95, fontsize=8, handlelength=1.2)
    # title removed: figure caption carries the message (matches fig3/internalisation house style)
    fig.tight_layout()
    os.makedirs("/e/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"/e/figures/cross_family_lift_bar_vertical_notitle.{ext}", dpi=200, bbox_inches="tight")
    emv.commit()
    return {"rows": [c for c, _ in rows]}


@app.local_entrypoint()
def main():
    print(render.remote())
