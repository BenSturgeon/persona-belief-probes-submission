"""Input: persona_graphic_orig.pdf (the Canva export, metadata removed). Output: persona_graphic_new.pdf.
Rebuild Figure 1 panels B and C from verified values and fix one Panel A phrase.
Charts are regenerated in matplotlib with the original styling (DejaVu Sans, sizes, colours, gridline opacity) at
the exact size of each original chart box, then placed into the Canva PDF after removing the old chart content.
Values (Llama-3.3-70B): Panel B = Figure 2 metric (own-probe lift, L56, genF): SFT 0.0274, OCT 0.1988, EM 0.2793.
Panel C = judged rates: persona_blackbox_sysprompt_era_believed, persona_blackbox_icl_era_believed (k=32),
persona_blackbox_era_believed (SFT), oct-darwin blackbox_mm v3 (OCT), EM main run (Wilson set)."""
import os, pymupdf as fitz, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
H = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
COL = {"sp": (0.855, 0.467, 0.341), "icl": (0.239, 0.357, 0.506), "sft": (0.475, 0.549, 0.369),
       "oct": (0.565, 0.424, 0.612), "em": (0.753, 0.345, 0.31)}
half_up = lambda v: str(Decimal(str(v)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def chart(path, box, ax_rect, keys, labels, vals, texts, ylim, yticks, yfmt, title, title_xy, ylabel, ylabel_xy, fs, tfs, xlim, width, ypad=26):
    W, Hh = box[2] - box[0], box[3] - box[1]
    fig = plt.figure(figsize=(W / 72, Hh / 72)); ax = fig.add_axes(ax_rect)
    for i, (k, v, t) in enumerate(zip(keys, vals, texts)):
        ax.bar(i, v, width, color=COL[k], zorder=2)
        if v >= 0: ax.text(i, v + (ylim[1] - ylim[0]) * 0.025, t, ha="center", va="bottom", fontsize=fs)
        else: ax.text(i, v - (ylim[1] - ylim[0]) * 0.012, t, ha="center", va="top", fontsize=fs)
    ax.set_ylim(*ylim); ax.set_xlim(*xlim); ax.set_yticks(yticks); ax.set_yticklabels([yfmt(y) for y in yticks], fontsize=fs)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=fs)
    ax.tick_params(axis="x", length=0, pad=11); ax.tick_params(axis="y", length=0, pad=ypad)
    for s in ax.spines.values(): s.set_visible(False)
    for y in yticks: ax.axhline(y, color="black", lw=0.29, alpha=0.6 if y == 0 else 0.25, zorder=1)
    fig.text(*title_xy, title, ha="center", va="center", fontsize=tfs)
    fig.text(*ylabel_xy, ylabel, ha="center", va="center", rotation=90, fontsize=fs)
    fig.savefig(path, transparent=True); plt.close(fig)

# Panel B: all five interventions, as designed. SFT/OCT/EM: each model's own probe (Figure 2 metric, L56).
# System prompt / ICL leave the weights unchanged and use the base-model (raw-text) probe, as the caption states:
# +0.069 / -0.041, reproduced by groundtruth/gt_hero_old_panelB.py (L56).
Bbox = (1715, 270, 3190, 1250); W, Hh = Bbox[2] - Bbox[0], Bbox[3] - Bbox[1]
fx = lambda x: (x - Bbox[0]) / W; fy = lambda y: (Bbox[3] - y) / Hh
chart(f"{H}/panelB.pdf", Bbox, [fx(1920), fy(1171), fx(3152) - fx(1920), fy(387) - fy(1171)],
      ["sp", "icl", "sft", "oct", "em"], ["System prompt", "ICL (k=32)", "SFT", "OCT", "EM"],
      [0.0690, -0.0410, 0.0274, 0.1988, 0.2793], ["0.07", "-0.04", "0.03", "0.20", "0.28"],
      (-0.06, 0.30), [0.0, 0.1, 0.2, 0.3], lambda y: f"{y:.1f}", "Calibrated truth-probe uplift", (fx(2470), fy(317)),
      "Calibrated lift toward 'true' (0\u21921)", (fx(1754), fy(756)), 34.4, 41, (-0.42, 4.42), 0.84, ypad=25)

# Panel C: behaviour, all five interventions
lab = ["System prompt", "ICL (k=32)", "SFT", "OCT", "EM"]; K = ["sp", "icl", "sft", "oct", "em"]
defend = [124 / 1800 * 100, 2 / 360 * 100, 255 / 1800 * 100, 59.2, 218 / 390 * 100]
consist = [696 / 1800 * 100, 60 / 360 * 100, 34.5, 69.0, 318 / 388 * 100]
tx = lambda vs: [("<1" if v < 1 else half_up(round(v, 1))) for v in vs]
for name, box, vals, ylim, yt, title, ty in [("panelC_defend", (136, 1549, 1600, 2032), defend, (0, 72), [0, 20, 40, 60], "Defend under challenge", 1590),
                                            ("panelC_consist", (136, 2082, 1602, 2565), consist, (0, 92), [0, 20, 40, 60, 80], "Consistency under generalization", 2123)]:
    W, Hh = box[2] - box[0], box[3] - box[1]; fx = lambda x: (x - box[0]) / W; fy = lambda y: (box[3] - y) / Hh
    base = box[3] - 81; top = base - 266
    chart(f"{H}/{name}.pdf", box, [fx(288), fy(base), fx(1571) - fx(288), fy(top) - fy(base)], K, lab, vals, tx(vals),
          ylim, yt, lambda y: f"{y:d}", title, (0.5, fy(ty)), "Rate (%)", (fx(180), fy((box[1] + box[3]) / 2 + 45)), 35, 41,
          (-0.4325, 4.4325), 0.865, ypad=27)

doc = fitz.open(f"{H}/persona_graphic_orig.pdf"); pg = doc[0]
B = fitz.Rect(1715, 270, 3190, 1250); C1 = fitz.Rect(136, 1549, 1600, 2032); C2 = fitz.Rect(136, 2082, 1602, 2565)
A = fitz.Rect(970, 343, 1200, 393)  # 'figures]...”' in Panel A
for r in (B, C1, C2): pg.add_redact_annot(r, fill=(1, 1, 1))
pg.add_redact_annot(A, fill=(250 / 255, 237 / 255, 235 / 255))
pg.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
for r, f in ((B, "panelB"), (C1, "panelC_defend"), (C2, "panelC_consist")):
    src = fitz.open(f"{H}/{f}.pdf"); pg.show_pdf_page(r, src, 0)
font = matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"; txt = "personas]...”"
w = fitz.get_text_length(txt, fontname="dejavu", fontfile=font, fontsize=40.99) if False else fitz.Font(fontfile=font).text_length(txt, fontsize=40.99)
pg.insert_font(fontname="dejavu", fontfile=font)
pg.insert_text(((975.8 + 1192.5) / 2 - w / 2, 381.9), txt, fontname="dejavu", fontsize=40.99, color=(0, 0, 0))
doc.set_metadata({}); doc.del_xml_metadata()
doc.save(f"{H}/persona_graphic_new.pdf", garbage=3, deflate=True); print("saved persona_graphic_new.pdf")
