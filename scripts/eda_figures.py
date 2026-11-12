"""EDA / analysis-plan figures for MI2. Saves 300 dpi PNGs to figures/.

Usage: python scripts/eda_figures.py
Run scripts/build_dataset.py first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
df = pd.read_csv(ROOT / "data" / "manifest.csv")

SPLITS = ["train", "val", "test", "excluded"]
SPLIT_LABEL = {"train": "Train", "val": "Validation", "test": "Test (HOME)",
               "excluded": "Excluded"}
SPLIT_COLOR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a",
               "excluded": "#b9b7b0"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE_ORDER = list(dict.fromkeys(df["source_style"]))  # build order: art, gen I..V, HOME
STYLE_SPLIT = df.drop_duplicates("source_style").set_index("source_style")["split"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def style_label(style):
    gen = df.loc[df["source_style"] == style, "generation"].iloc[0]
    return style if pd.isna(gen) else f"{style} (Gen {int(gen)})"


def split_legend(ax, **kw):
    handles = [Patch(color=SPLIT_COLOR[s], label=SPLIT_LABEL[s]) for s in SPLITS]
    ax.legend(handles=handles, frameon=False, **kw)


def fig1_images_per_style():
    counts = df["source_style"].value_counts().reindex(STYLE_ORDER)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4.2, 5.6), gridspec_kw={"height_ratios": [3.2, 1]})

    y = range(len(counts))[::-1]
    ax1.barh(list(y), counts.values, height=0.75,
             color=[SPLIT_COLOR[STYLE_SPLIT[s]] for s in counts.index])
    ax1.set_yticks(list(y), [style_label(s) for s in counts.index])
    for yi, v in zip(y, counts.values):
        ax1.text(v + 2, yi, str(v), va="center", fontsize=6.5, color=INK_2)
    ax1.set_xlim(0, counts.max() * 1.15)
    ax1.set_xlabel("Images")
    ax1.set_title("A. Images per source style", loc="left", fontweight="bold", pad=18)
    ax1.grid(axis="x", color=GRID, lw=0.6)
    ax1.set_axisbelow(True)
    split_legend(ax1, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4,
                 borderaxespad=0.2, handlelength=1.0, handletextpad=0.4, columnspacing=0.8)

    per = df[df["split"] != "excluded"].groupby("pokedex_id").size()
    lo, hi = per.min(), per.max()
    ax2.hist(per, bins=range(lo - 2, hi + 4), color=INK_2, rwidth=0.8, align="left")
    ax2.set_xticks(range(lo - 2, hi + 3))
    ax2.set_xlabel("Images per Pokémon")
    ax2.set_ylabel("Pokémon")
    ax2.set_title("B. Images per Pokémon (train + val + test)", loc="left", fontweight="bold")
    if lo == hi:
        ax2.text(0.03, 0.9, f"All {len(per)} Pokémon\nhave {lo} images",
                 transform=ax2.transAxes, va="top", fontsize=7, color=INK_2)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout(h_pad=1.5)
    fig.savefig(FIG / "fig1_images_per_style.png")
    plt.close(fig)


def fig2_sample_grid():
    pokemon = [(25, "Pikachu"), (6, "Charizard"), (94, "Gengar"), (133, "Eevee")]
    styles = [("official-artwork", "Official art\n(excluded)"), ("red-blue", "Gen I sprite\nRed/Blue"),
              ("diamond-pearl", "Gen IV sprite\nDiamond/Pearl"), ("home", "HOME\n(test)")]
    rows = df.set_index(["pokedex_id", "source_style"])

    fig, axes = plt.subplots(2 * len(pokemon), len(styles), figsize=(4.3, 7.6),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.1})
    for r, (pid, name) in enumerate(pokemon):
        for c, (style, label) in enumerate(styles):
            rec = rows.loc[(pid, style)]
            for k, col in enumerate(["color_path", "silhouette_path"]):
                ax = axes[2 * r + k, c]
                ax.imshow(Image.open(ROOT / rec[col]).convert("RGB"), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(GRID)
                if r == 0 and k == 0:
                    ax.set_title(label, fontsize=7)
                if c == 0:
                    ax.set_ylabel("color" if k == 0 else "silhouette", fontsize=6.5, color=INK_2)
        # Pokemon name spans its pair of rows
        top = axes[2 * r, 0].get_position()
        bot = axes[2 * r + 1, 0].get_position()
        fig.text(top.x0 - 0.075, (top.y1 + bot.y0) / 2, name, rotation=90,
                 ha="center", va="center", fontsize=8.5, fontweight="bold")
    fig.savefig(FIG / "fig2_sample_grid.png")
    plt.close(fig)


def fig3_analysis_plan():
    steps = [
        ("Build dataset", "Gen 1 (#1–151), front default forms"),
        ("Crop to bounding box", "alpha > 0 bounding box + 5% margin"),
        ("Preprocess", "pad to square, resize 224×224, make silhouettes"),
        ("Style-based split", "train: Gen 1–4 sprites  /  val: Gen 5  /  test: HOME"),
        ("Baselines", "random guess 0.7%  &  k-NN on 64×64 pixels"),
        ("Fine-tune ResNet-18 ×2", "color & silhouette, same settings, augmentation"),
        ("Tune on validation", "learning rate, epochs"),
        ("Evaluate on HOME", "top-1, top-5, macro F1; goal >70% silhouette top-1"),
        ("Analyze", "color vs. silhouette gap, confusion pairs, Grad-CAM"),
    ]
    box_h, gap, w = 0.62, 0.32, 3.4
    total = len(steps) * box_h + (len(steps) - 1) * gap
    fig, ax = plt.subplots(figsize=(3.8, total * 0.95 + 0.2))
    ax.set_xlim(0, 3.8)
    ax.set_ylim(-0.05, total + 0.05)
    ax.axis("off")

    x0 = (3.8 - w) / 2
    for i, (title, sub) in enumerate(steps):
        y = total - i * (box_h + gap) - box_h
        highlight = title.startswith("Evaluate")
        ax.add_patch(FancyBboxPatch(
            (x0, y), w, box_h, boxstyle="round,pad=0,rounding_size=0.1",
            fc="#e8f1fb" if highlight else "#f4f3f0",
            ec=SPLIT_COLOR["train"] if highlight else "#b9b7b0", lw=1.0))
        ax.text(x0 + w / 2, y + box_h * 0.64, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x0 + w / 2, y + box_h * 0.28, sub, ha="center", va="center",
                fontsize=6.5, color=INK_2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x0 + w / 2, y - 0.02), (x0 + w / 2, y - gap + 0.02),
                arrowstyle="-|>", mutation_scale=9, color=INK_2, lw=1.0))
    fig.savefig(FIG / "fig3_analysis_plan.png")
    plt.close(fig)


def fig4_fill_ratio():
    order = STYLE_ORDER[::-1]
    data = [df.loc[df["source_style"] == s, "fill_ratio"] for s in order]
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True,
                    medianprops={"color": INK, "lw": 1.2},
                    flierprops={"marker": "o", "markersize": 2.5, "markeredgewidth": 0,
                                "markerfacecolor": INK_2, "alpha": 0.6},
                    whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for patch, s in zip(bp["boxes"], order):
        patch.set(facecolor=SPLIT_COLOR[STYLE_SPLIT[s]], alpha=0.85, edgecolor=INK_2, lw=0.6)
    ax.set_yticks(range(1, len(order) + 1), [style_label(s) for s in order])
    ax.set_xlabel("Fill ratio (Pokémon pixels / 224×224)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Fill ratio by source style", loc="left", fontweight="bold", pad=18)
    split_legend(ax, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4,
                 borderaxespad=0.2, handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_fill_ratio.png")
    plt.close(fig)



def silhouette_quality(path):
    """Return (black components with 8-connectivity, enclosed white holes).

    Holes use 4-connectivity for white, the dual of 8-connected black, so a hole
    that only touches the outside diagonally still counts as enclosed.
    """
    black = np.asarray(Image.open(ROOT / path)) == 0
    _, n_components = ndimage.label(black, structure=np.ones((3, 3), dtype=int))
    white, n_white = ndimage.label(~black)
    border = np.unique(np.concatenate([white[0], white[-1], white[:, 0], white[:, -1]]))
    n_holes = n_white - np.count_nonzero(border)
    return n_components, n_holes


def fig5_silhouette_quality():
    q = df[["pokedex_id", "name", "source_style", "split"]].copy()
    q[["components", "holes"]] = [silhouette_quality(p) for p in df["silhouette_path"]]

    print("Silhouette quality by source_style (components: 8-conn black; holes: enclosed white)")
    stats = q.groupby("source_style")[["components", "holes"]].agg(["mean", "median"])
    print(stats.reindex(STYLE_ORDER).round(2).to_string())
    for style in ["red-blue", "yellow"]:
        top = q[q["source_style"] == style].nlargest(5, "components")
        print(f"\nMost components in {style}:")
        print(top[["pokedex_id", "name", "components", "holes"]].to_string(index=False))

    # Nearly every silhouette is a single component, so quartiles all sit at 1 and a
    # boxplot collapses; show means with 95% bootstrap CIs instead.
    order = STYLE_ORDER[::-1]
    rng = np.random.default_rng(0)
    means, lo, hi = [], [], []
    for style in order:
        x = q.loc[q["source_style"] == style, "components"].to_numpy()
        boot = rng.choice(x, size=(2000, len(x))).mean(axis=1)
        means.append(x.mean())
        lo.append(x.mean() - np.percentile(boot, 2.5))
        hi.append(np.percentile(boot, 97.5) - x.mean())
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    y = np.arange(len(order))
    ax.barh(y, means, height=0.7, color=[SPLIT_COLOR[STYLE_SPLIT[s]] for s in order])
    ax.errorbar(means, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for yi, m, h in zip(y, means, hi):
        ax.text(m + h + 0.05, yi, f"{m:.2f}", va="center", fontsize=6.5, color=INK_2)
    ax.set_yticks(y, [style_label(s) for s in order])
    ax.set_xlim(0, max(m + h for m, h in zip(means, hi)) * 1.15)
    ax.set_xlabel("Mean connected black components per silhouette (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Silhouette fragmentation by source style", loc="left", fontweight="bold",
                 pad=18)
    split_legend(ax, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=4,
                 borderaxespad=0.2, handlelength=1.0, handletextpad=0.4, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_silhouette_quality.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1_images_per_style()
    fig2_sample_grid()
    fig3_analysis_plan()
    fig4_fill_ratio()
    fig5_silhouette_quality()
    print("Saved:", *sorted(p.name for p in FIG.glob("*.png")), sep="\n  ")
