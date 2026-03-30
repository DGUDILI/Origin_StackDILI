import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 20))
ax.set_xlim(0, 16)
ax.set_ylim(0, 20)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")

C_BENCH  = "#8B7355"
C_ENCODE = "#4A6FA5"
C_SELECT = "#5C8A5C"
C_FUSION = "#7B6FA5"
C_CLASSIF= "#A5755C"
C_PERF   = "#5C8A7B"
C_ARROW  = "#666666"
C_FP_BG  = "#E8EFF8"
C_CB_BG  = "#EEE8F5"
C_BOX_BG = "#FFFFFF"
C_RED    = "#CC3333"
C_BLUE   = "#2255AA"

def rounded_box(ax, x, y, w, h, color, label=None, label_color="white",
                fontsize=11, radius=0.4):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle=f"round,pad=0.05,rounding_size={radius}",
                         linewidth=1.5, edgecolor=color,
                         facecolor=color if label else C_BOX_BG,
                         zorder=3)
    ax.add_patch(box)
    if label:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=label_color, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=2.0):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"), zorder=2)

def side_label(ax, x, y, text, color, fontsize=12):
    box = FancyBboxPatch((x-1.2, y-0.35), 2.4, 0.7,
                         boxstyle="round,pad=0.05,rounding_size=0.3",
                         linewidth=1.5, edgecolor=color, facecolor=color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)

ax.text(8, 19.4, "DGUDILI (2026)", ha="center", fontsize=20, fontweight="bold", color="#222222")
ax.text(8, 18.9, "Drug-Induced Liver Injury Prediction via Cross-Attention Fusion",
        ha="center", fontsize=11, color="#555555")

Y_PERF = 18.1
side_label(ax, 1.4, Y_PERF, "Performance", C_PERF)
perf_box = FancyBboxPatch((3.5, Y_PERF-0.55), 9, 1.1,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           linewidth=1.5, edgecolor=C_PERF, facecolor="#EAF4F0", zorder=3)
ax.add_patch(perf_box)
ax.text(8, Y_PERF+0.25, "AUC  /  MCC  /  F1", ha="center", fontsize=14, fontweight="bold", color=C_RED)
ax.text(8, Y_PERF-0.2, "Test set: DILIrank  (p=184, n=268)", ha="center", fontsize=10, color="#555555")

Y_CLF = 16.7
side_label(ax, 1.4, Y_CLF, "Classifier", C_CLASSIF)
ax.text(4.3, Y_CLF+0.15, "Logistic", ha="center", fontsize=11, fontweight="bold", color=C_RED)
ax.text(4.3, Y_CLF-0.2, "Regression", ha="center", fontsize=11, fontweight="bold", color=C_RED)
fs_box = FancyBboxPatch((5.2, Y_CLF-0.45), 7.3, 0.9,
                         boxstyle="round,pad=0.05,rounding_size=0.3",
                         linewidth=1.5, edgecolor="#AAAAAA", facecolor="#F5F5F5", zorder=3)
ax.add_patch(fs_box)
ax.text(8.85, Y_CLF+0.08, "Feature Space", ha="center", fontsize=12, color="#333333")
ax.text(8.85, Y_CLF-0.22, "(k = 16)", ha="center", fontsize=13, fontweight="bold", color=C_RED)
arrow(ax, 8.85, Y_CLF+0.45, 8.85, Y_PERF-0.55, color=C_CLASSIF, lw=2.5)

Y_CA = 15.15
arrow(ax, 8.85, Y_CLF-0.45, 8.85, Y_CA+0.5, color=C_CLASSIF, lw=2.5)
ca_box = FancyBboxPatch((3.5, Y_CA-0.5), 10.5, 1.0,
                         boxstyle="round,pad=0.05,rounding_size=0.3",
                         linewidth=2.0, edgecolor=C_FUSION, facecolor="#F0EDF8", zorder=3)
ax.add_patch(ca_box)
ax.text(8.75, Y_CA+0.18, "Cross-Attention  (Q, K, V)", ha="center", fontsize=13, fontweight="bold", color=C_FUSION)
ax.text(8.75, Y_CA-0.2, "Q <- ChemBERTa (query)       K, V <- FP (key / value)",
        ha="center", fontsize=10, color="#444444")
form_box = FancyBboxPatch((3.6, Y_CA-1.35), 10.3, 0.75,
                           boxstyle="round,pad=0.05,rounding_size=0.2",
                           linewidth=1.2, edgecolor="#CCBBEE", facecolor="#FAF8FF", zorder=3)
ax.add_patch(form_box)
ax.text(8.75, Y_CA-0.97,
        "scores = Q @ K^T / sqrt(d_k) -> (B,16,16)\n"
        "weights = softmax(scores)  ->  attn = weights @ V  ->  (B,16,1)  -> squeeze ->  (B,16)",
        ha="center", fontsize=9.5, color="#333333", family="monospace")

Y_FS = 13.1
side_label(ax, 1.4, Y_FS, "Feature\nSelection", C_SELECT, fontsize=11)
for cx, lbl in [(5.5, "k = 16\nSelectKBest (f_classif)"), (12.1, "k = 16\nSelectKBest (f_classif)")]:
    lines = lbl.split("\n")
    ax.text(cx, Y_FS+0.15, lines[0], ha="center", fontsize=14, fontweight="bold", color=C_RED)
    ax.text(cx, Y_FS-0.22, lines[1], ha="center", fontsize=9, color="#555555")
arrow(ax, 5.5, Y_FS+0.45, 6.5, Y_CA-1.45, color=C_SELECT, lw=2.2)
arrow(ax, 12.1, Y_FS+0.45, 11.0, Y_CA-1.45, color=C_SELECT, lw=2.2)

Y_ENC = 11.4
side_label(ax, 1.4, Y_ENC, "Data\nEncoding", C_ENCODE, fontsize=11)
fp_box = FancyBboxPatch((2.8, Y_ENC-0.9), 7.2, 1.8,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          linewidth=1.8, edgecolor=C_ENCODE, facecolor=C_FP_BG, zorder=3)
ax.add_patch(fp_box)
ax.text(6.4, Y_ENC+0.55, "FP  (k = 29 + 150 + 167 + 79 = 425)",
        ha="center", fontsize=11, fontweight="bold", color=C_ENCODE)
for lbl, cx in [("Constitution\n(29)",3.55),("CalcCATS\n(150)",5.35),("MACCS\n(167)",7.15),("Estate\n(79)",8.95)]:
    sb = FancyBboxPatch((cx-0.75, Y_ENC-0.75), 1.5, 0.9,
                         boxstyle="round,pad=0.03,rounding_size=0.2",
                         linewidth=1.0, edgecolor="#88AACC", facecolor="#D8E8F5", zorder=4)
    ax.add_patch(sb)
    ax.text(cx, Y_ENC-0.3, lbl, ha="center", fontsize=8.5, color="#223366")
cb_box = FancyBboxPatch((10.6, Y_ENC-0.55), 4.5, 1.1,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          linewidth=1.8, edgecolor=C_FUSION, facecolor=C_CB_BG, zorder=3)
ax.add_patch(cb_box)
ax.text(12.85, Y_ENC+0.22, "ChemBERTa", ha="center", fontsize=12, fontweight="bold", color=C_FUSION)
ax.text(12.85, Y_ENC-0.15, "seyonec/ChemBERTa-zinc-base-v1", ha="center", fontsize=8, color="#555555")
ax.text(12.85, Y_ENC-0.42, "CLS token  ->  768-dim", ha="center", fontsize=9, color="#555555")
arrow(ax, 5.5, Y_ENC-0.9, 5.5, Y_FS-0.45, color=C_ENCODE, lw=2.2)
arrow(ax, 12.1, Y_ENC-0.55, 12.1, Y_FS-0.45, color=C_FUSION, lw=2.2)

Y_FEPY = 9.95
ax.text(6.4, Y_FEPY+0.1, "Feature.py", ha="center", fontsize=11, fontweight="bold", color=C_BLUE)
ax.text(6.4, Y_FEPY-0.2, "(Dataset.csv, iFeatureOmegaCLI)", ha="center", fontsize=9, color=C_BLUE)
arrow(ax, 8.0, Y_FEPY, 8.0, Y_ENC-0.9, color=C_ENCODE, lw=2.5)

Y_BEN = 8.8
side_label(ax, 1.4, Y_BEN, "Benchmark", C_BENCH)
bench_box = FancyBboxPatch((2.8, Y_BEN-0.55), 12.4, 1.1,
                             boxstyle="round,pad=0.05,rounding_size=0.4",
                             linewidth=2.0, edgecolor=C_BENCH, facecolor="#F5F0EB", zorder=3)
ax.add_patch(bench_box)
ax.text(9.0, Y_BEN+0.18, "DILI Dataset", ha="center", fontsize=13, fontweight="bold", color="#333333")
ax.text(9.0, Y_BEN-0.2, "TRAIN: p=768 / n=630          TEST: p=184 / n=268",
        ha="center", fontsize=11, color="#555555")
arrow(ax, 8.0, Y_BEN+0.55, 8.0, Y_FEPY+0.35, color=C_BENCH, lw=2.5)
arrow(ax, 12.85, Y_BEN+0.55, 12.85, Y_ENC-0.55, color=C_FUSION, lw=2.5)

Y_LEG = 7.6
legend_items = [(C_BENCH,"Benchmark / Dataset"),(C_ENCODE,"Data Encoding (FP)"),
                (C_FUSION,"ChemBERTa / Fusion"),(C_SELECT,"Feature Selection"),
                (C_CLASSIF,"Classifier"),(C_PERF,"Performance")]
leg_x = 3.0
for i, (c, lbl) in enumerate(legend_items):
    lx = leg_x + (i%3)*3.5
    ly = Y_LEG - (i//3)*0.6
    ax.add_patch(plt.Circle((lx, ly), 0.13, color=c, zorder=5))
    ax.text(lx+0.25, ly, lbl, va="center", fontsize=9, color="#333333")

ax.text(8, 6.6, "DGUDILI (2026)  --  Cross-Attention Fusion of Molecular Fingerprints & ChemBERTa",
        ha="center", fontsize=10, color="#888888", style="italic")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "DGUDILI_2026_flowchart.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
