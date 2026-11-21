"""
=============================================================================
04_evaluate.py - Final test-set evaluation, significance tests, error analysis
=============================================================================
Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen

PURPOSE
    The ONLY script that evaluates the CNNs on the test split (151 Pokemon HOME
    renders, one per Pokemon). Each final model selected in 03_train_cnn.py is
    run on HOME exactly once. Then:
      1. Top-1, top-5 and macro F1 for: color CNN, silhouette CNN, k-NN color,
         k-NN silhouette, and random guessing (1/151).
      2. Two-sided exact McNemar tests (alpha = 0.05) on paired top-1
         correctness: (a) color CNN vs silhouette CNN, (b) silhouette CNN vs
         k-NN silhouette, (c) color CNN vs k-NN color.
      3. Paired bootstrap (10,000 resamples of the 151 test images, seed 4002):
         95 % percentile CIs for every model's top-1 and for accuracy gaps.
      4. Finish-line check from MI2: silhouette CNN top-1 >= 70 % AND
         significantly better than k-NN silhouette (McNemar p < 0.05).
      5. Error analysis: most-confused Pokemon pairs per model, Pokemon one CNN
         gets right and the other wrong, accuracy by primary type.

INPUTS
    - models/cnn_color_best.pt, models/cnn_silhouette_best.pt   (from 03)
    - OUTPUT/tables/selected_models.csv                          (from 03)
    - OUTPUT/tables/knn_test_predictions.csv                     (from 02)
    - DATA/manifest.csv and the HOME test images                 (from 00)

OUTPUTS
    OUTPUT/tables/
      results_summary.csv          top-1/top-5/macro F1 (+ bootstrap CI) per model
      mcnemar_tests.csv            2x2 counts, discordant pairs, p-values
      bootstrap_cis.csv            95 % CIs for top-1 and accuracy gaps
      goal_check.csv               PASS/FAIL on the 70 % finish line
      test_predictions.csv         top-5 predictions of every model, per image
      confused_pairs_color.csv     top 15 confused pairs, color CNN
      confused_pairs_silhouette.csv  top 15 confused pairs, silhouette CNN
      confused_pairs_knn_color.csv / confused_pairs_knn_silhouette.csv
      disagreements_color_vs_silhouette.csv  Pokemon exactly one CNN got right
      accuracy_by_primary_type.csv
    OUTPUT/figures/
      accuracy_comparison.png      top-1 bars with bootstrap CIs + reference lines
      confusion_matrix_cnn_pokedex.png   151x151 test confusion matrices
      confusion_matrix_cnn_by_type.png   confusions aggregated by primary type

HOW TO RUN (from the repository root)
    python SCRIPTS/04_evaluate.py

EXPECTED RUNTIME
    ~5 s (inference on 151 images x 2 models + 10,000 bootstrap resamples).
=============================================================================
"""

import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

# sklearn warns that 151 classes in 151 samples "could be regression"; harmless.
warnings.filterwarnings("ignore", message="The number of unique classes")

from cnn_utils import (FIGURES, MODELS, N_CLASSES, ROOT, SEED, TABLES,
                       describe_device, get_device, load_checkpoint,
                       load_images, load_manifest, predict_logits)

ALPHA = 0.05            # significance level for McNemar tests
GOAL = 0.70             # MI2 finish line for silhouette CNN top-1
N_BOOT = 10_000         # bootstrap resamples
TOP_PAIRS = 15          # confused pairs reported per model

# Display names, in the order used in every table and figure.
MODELS_ORDER = ["cnn_color", "cnn_silhouette", "knn_color", "knn_silhouette"]
LABEL = {"cnn_color": "CNN (color)", "cnn_silhouette": "CNN (silhouette)",
         "knn_color": "k-NN (color)", "knn_silhouette": "k-NN (silhouette)",
         "random": "Random guess"}
COMPARISONS = [  # (id, model A, model B) for McNemar and bootstrap gaps
    ("a", "cnn_color", "cnn_silhouette"),
    ("b", "cnn_silhouette", "knn_silhouette"),
    ("c", "cnn_color", "knn_color"),
]

# Figure style shared with 01_eda_figures.py (blue = color input,
# orange = silhouette input; palette checked for color-vision deficiency).
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
HUE = {"color": "#2a78d6", "silhouette": "#eb6834"}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# 1. Predictions
# ---------------------------------------------------------------------------
def cnn_predictions(test, device):
    """Run each selected CNN once on HOME. Returns {model: (top5 dex ids, probs)}
    where probs is the full (151, 151) softmax matrix."""
    out = {}
    for mode in ["color", "silhouette"]:
        model, meta = load_checkpoint(MODELS / f"cnn_{mode}_best.pt", device)
        print(f"cnn_{mode}: lr={meta['lr']:g}, exclude_gen1={meta['exclude_gen1']}, "
              f"best epoch {meta['best_epoch']}, val top-1 {meta['val_top1']:.4f}")
        probs = torch.softmax(predict_logits(model, load_images(test, mode), device), 1).numpy()
        top5 = np.argsort(-probs, axis=1)[:, :5] + 1           # class index -> Pokedex #
        out[f"cnn_{mode}"] = (top5, probs)
    return out


def knn_predictions(test):
    """Read the k-NN top-5 predictions saved by 02_knn_baseline.py, aligned to
    the same row order as `test`."""
    knn = pd.read_csv(TABLES / "knn_test_predictions.csv")
    out = {}
    for model, g in knn.groupby("model"):
        g = g.set_index("image_id").loc[test["image_id"]]
        out[model] = (g[[f"pred_{r}" for r in range(1, 6)]].to_numpy(), None)
    return out


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
def metrics(y, top5):
    return {"top1": float((top5[:, 0] == y).mean()),
            "top5": float((top5 == y[:, None]).any(axis=1).mean()),
            # labels=1..151 so classes that are never predicted count as F1 = 0
            "macro_f1": float(f1_score(y, top5[:, 0], average="macro",
                                       labels=np.arange(1, N_CLASSES + 1), zero_division=0)),
            "n_correct_top1": int((top5[:, 0] == y).sum())}


def random_guess_metrics(y, rng):
    """Expected metrics for uniform random guessing. Top-1 = 1/151 and
    top-5 = 5/151 exactly; expected macro F1 has no simple closed form, so it is
    estimated by simulating 10,000 random prediction vectors."""
    sims = rng.integers(1, N_CLASSES + 1, size=(N_BOOT, len(y)))
    n_true = np.bincount(y, minlength=N_CLASSES + 1)[1:]
    f1s = []
    for pred in sims:
        tp = np.bincount(pred[pred == y], minlength=N_CLASSES + 1)[1:]
        n_pred = np.bincount(pred, minlength=N_CLASSES + 1)[1:]
        f1s.append(np.mean(2 * tp / (n_pred + n_true)))   # F1 = 2TP / (#pred + #true)
    return {"top1": 1 / N_CLASSES, "top5": 5 / N_CLASSES, "macro_f1": float(np.mean(f1s)),
            "n_correct_top1": np.nan}


def mcnemar_row(cid, a, b, correct):
    """Exact two-sided McNemar test on paired correctness of models a and b."""
    ca, cb = correct[a], correct[b]
    table = [[int(np.sum(ca & cb)), int(np.sum(ca & ~cb))],
             [int(np.sum(~ca & cb)), int(np.sum(~ca & ~cb))]]
    res = mcnemar(table, exact=True)   # binomial test on the two discordant cells
    return {"comparison": cid, "model_a": a, "model_b": b,
            "top1_a": ca.mean(), "top1_b": cb.mean(),
            "both_correct": table[0][0], "a_correct_b_wrong": table[0][1],
            "a_wrong_b_correct": table[1][0], "both_wrong": table[1][1],
            "n_discordant": table[0][1] + table[1][0],
            "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "significant_at_0.05": bool(res.pvalue < ALPHA)}


def bootstrap(correct, rng):
    """Paired percentile bootstrap. The SAME resampled image indices are used
    for every model, so gaps between models keep their pairing."""
    n = len(next(iter(correct.values())))
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {m: c[idx].mean(axis=1) for m, c in correct.items()}
    quantities = [(f"top1_{m}", m, None) for m in MODELS_ORDER]
    quantities += [(f"gap_{a}_minus_{b}", a, b) for _, a, b in COMPARISONS]
    rows = []
    for name, a, b in quantities:
        est = correct[a].mean() - (correct[b].mean() if b else 0)
        dist = boot[a] - (boot[b] if b else 0)
        lo, hi = np.percentile(dist, [2.5, 97.5])
        rows.append({"quantity": name, "estimate": est, "ci_low_2.5": lo,
                     "ci_high_97.5": hi, "bootstrap_se": dist.std(ddof=1),
                     "n_resamples": N_BOOT, "seed": SEED})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Error analysis
# ---------------------------------------------------------------------------
def confused_pairs(test, top5, probs, names, types):
    """Top confused Pokemon pairs on HOME for one model.

    HOME has one image per Pokemon, so a pair can be confused at most twice
    (A->B and B->A). Pairs are ranked by that hard count, then (CNNs only) by
    a "soft confusion" score: the softmax probability the model gives to the
    other member of the pair, summed over both directions. The soft score
    separates confident mistakes from near-ties.
    """
    y = test["pokedex_id"].to_numpy()
    pred = top5[:, 0]
    pairs = {}
    for t, p, i in zip(y, pred, range(len(y))):
        if t == p:
            continue
        key = tuple(sorted((t, p)))
        d = pairs.setdefault(key, {"n": 0, "dirs": [], "conf": []})
        d["n"] += 1
        d["dirs"].append(f"{names[t]}->{names[p]}")
        if probs is not None:
            d["conf"].append(probs[i, p - 1])
    row_of = {dex: i for i, dex in enumerate(y)}   # test row for each Pokemon
    rows = []
    for (a, b), d in pairs.items():
        soft = (probs[row_of[a], b - 1] + probs[row_of[b], a - 1]) if probs is not None else np.nan
        rows.append({"pokedex_a": a, "name_a": names[a], "pokedex_b": b, "name_b": names[b],
                     "n_confusions": d["n"], "directions": "; ".join(d["dirs"]),
                     "mean_wrong_confidence": np.mean(d["conf"]) if d["conf"] else np.nan,
                     "soft_confusion": soft, "same_primary_type": types[a] == types[b],
                     "type_a": types[a], "type_b": types[b]})
    df = pd.DataFrame(rows)
    return (df.sort_values(["n_confusions", "soft_confusion", "pokedex_a"],
                           ascending=[False, False, True], na_position="last")
              .head(TOP_PAIRS).reset_index(drop=True))


def disagreements(test, preds, names):
    """Pokemon that exactly one of the two CNNs classifies correctly (top-1)."""
    y = test["pokedex_id"].to_numpy()
    (c5, cp), (s5, sp) = preds["cnn_color"], preds["cnn_silhouette"]
    rows = []
    for i, dex in enumerate(y):
        c_ok, s_ok = c5[i, 0] == dex, s5[i, 0] == dex
        if c_ok == s_ok:
            continue
        rows.append({"pokedex_id": dex, "name": names[dex],
                     "primary_type": test["primary_type"].iloc[i],
                     "fill_ratio": test["fill_ratio"].iloc[i],
                     "pattern": "color right, silhouette wrong" if c_ok
                                else "silhouette right, color wrong",
                     "color_pred": names[c5[i, 0]], "color_conf": cp[i, c5[i, 0] - 1],
                     "silhouette_pred": names[s5[i, 0]],
                     "silhouette_conf": sp[i, s5[i, 0] - 1]})
    return pd.DataFrame(rows).sort_values(["pattern", "pokedex_id"])


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def fig_accuracy(summary, cis):
    """Top-1 accuracy per model with 95 % bootstrap CIs; random guess and the
    70 % goal drawn as labeled reference lines."""
    ci = cis.set_index("quantity")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    x = np.array([0, 1, 2.4, 3.4])                    # small gap between CNN and k-NN groups
    for xi, m in zip(x, MODELS_ORDER):
        mode = m.split("_")[1]
        est = summary.loc[m, "top1"]
        lo, hi = ci.loc[f"top1_{m}", ["ci_low_2.5", "ci_high_97.5"]]
        is_knn = m.startswith("knn")
        ax.bar(xi, est, width=0.8, color=HUE[mode], alpha=0.45 if is_knn else 1.0,
               hatch="////" if is_knn else None, edgecolor="white", linewidth=0)
        ax.errorbar(xi, est, yerr=[[est - lo], [hi - est]], fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=3)
        ax.text(xi, hi + 0.02, f"{100 * est:.1f}%", ha="center", va="bottom", fontsize=7.5,
                color=INK, fontweight="bold")
    ax.axhline(GOAL, color=INK, ls="--", lw=0.9)
    ax.text(x[-1] + 0.55, GOAL + 0.012, "70% goal", va="bottom", fontsize=7, color=INK)
    ax.axhline(1 / N_CLASSES, color=INK_2, ls=":", lw=0.9)
    ax.text(x[-1] + 0.55, 1 / N_CLASSES + 0.012, f"Random\n({100 / N_CLASSES:.1f}%)",
            va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xticks(x, [LABEL[m].replace(" (", "\n(") for m in MODELS_ORDER])
    ax.set_xlim(-0.6, x[-1] + 1.4)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_ylabel("Top-1 accuracy on HOME test set")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Test accuracy by model (error bars: 95% paired bootstrap CI)",
                 loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_comparison.png")
    plt.close(fig)


def fig_confusion_pokedex(y, preds):
    """151x151 test confusion matrices for both CNNs, ordered by Pokedex number.
    With one image per Pokemon, each row holds a single mark: on the diagonal
    when correct, off it when wrong."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    for ax, m in zip(axes, ["cnn_color", "cnn_silhouette"]):
        pred = preds[m][0][:, 0]
        ok = pred == y
        ax.plot([1, N_CLASSES], [1, N_CLASSES], color=GRID, lw=0.6, zorder=0)
        ax.scatter(pred[ok], y[ok], s=4, color=INK_2, lw=0, label="correct")
        ax.scatter(pred[~ok], y[~ok], s=12, color=HUE[m.split("_")[1]], lw=0,
                   label="misclassified")
        ax.set_xlim(0, N_CLASSES + 1)
        ax.set_ylim(N_CLASSES + 1, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("Predicted Pokédex #")
        ax.set_ylabel("True Pokédex #")
        ax.set_title(f"{LABEL[m]}: {int((~ok).sum())} errors / {len(y)}",
                     loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower left", markerscale=1.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_cnn_pokedex.png")
    plt.close(fig)


def fig_confusion_type(test, preds, types):
    """Confusion matrices aggregated by primary type (rows = true type,
    normalized to proportions). Readable summary of whether mistakes stay
    within a type, e.g. one bird mistaken for another."""
    order = test["primary_type"].value_counts().index.tolist()
    y_type = test["primary_type"].to_numpy()
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#0f3d7a"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
    for ax, m in zip(axes, ["cnn_color", "cnn_silhouette"]):
        p_type = np.array([types[p] for p in preds[m][0][:, 0]])
        cm = pd.crosstab(pd.Categorical(y_type, order), pd.Categorical(p_type, order),
                         dropna=False).to_numpy()
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(order)):
            for j in range(len(order)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5,
                            color="white" if cm_norm[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(order)), order, rotation=90)
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel("Predicted primary type")
        ax.set_ylabel("True primary type")
        ax.set_title(LABEL[m], loc="left", fontweight="bold")
        ax.spines[:].set_visible(False)
    fig.suptitle("HOME test confusions by primary type (counts; shading = row share)",
                 x=0.01, ha="left", fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_cnn_by_type.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {describe_device(device)}")
    df = load_manifest()
    test = df[df["split"] == "test"].reset_index(drop=True)      # 151 HOME images
    names = dict(zip(df["pokedex_id"], df["name"]))
    types = dict(zip(df["pokedex_id"], df["primary_type"]))
    y = test["pokedex_id"].to_numpy()

    preds = {**cnn_predictions(test, device), **knn_predictions(test)}
    correct = {m: preds[m][0][:, 0] == y for m in MODELS_ORDER}

    # Per-image predictions of every model (useful for any follow-up analysis).
    long = []
    for m in MODELS_ORDER:
        top5, probs = preds[m]
        for i, row in test.iterrows():
            long.append({"model": m, "image_id": row["image_id"], "pokedex_id": row["pokedex_id"],
                         "name": row["name"], **{f"pred_{r + 1}": top5[i, r] for r in range(5)},
                         "pred_1_name": names[top5[i, 0]],
                         "pred_1_prob": probs[i, top5[i, 0] - 1] if probs is not None else np.nan,
                         "correct_top1": bool(correct[m][i])})
    pd.DataFrame(long).to_csv(TABLES / "test_predictions.csv", index=False)

    # --- Metrics + bootstrap CIs ---------------------------------------------
    rng = np.random.default_rng(SEED)
    cis = bootstrap(correct, rng)                     # bootstrap first: fixed RNG stream
    cis.to_csv(TABLES / "bootstrap_cis.csv", index=False)
    summary = pd.DataFrame({m: metrics(y, preds[m][0]) for m in MODELS_ORDER}).T
    summary.loc["random"] = random_guess_metrics(y, rng)
    summary = summary.astype(float)
    ci = cis.set_index("quantity")
    summary["top1_ci_low"] = [ci.loc[f"top1_{m}", "ci_low_2.5"] if m != "random" else np.nan
                              for m in summary.index]
    summary["top1_ci_high"] = [ci.loc[f"top1_{m}", "ci_high_97.5"] if m != "random" else np.nan
                               for m in summary.index]
    sel = pd.read_csv(TABLES / "selected_models.csv").set_index("model")
    summary["n_test"] = len(y)
    summary["config"] = [f"lr={sel.loc[m, 'lr']:g}, "
                         f"{'without' if sel.loc[m, 'exclude_gen1'] else 'with'} Gen I"
                         if m in sel.index else ("k=1, 64x64 pixels" if m.startswith("knn")
                                                 else "uniform over 151")
                         for m in summary.index]
    summary.insert(0, "label", [LABEL[m] for m in summary.index])
    summary.index.name = "model"
    summary.to_csv(TABLES / "results_summary.csv")

    # --- McNemar tests ---------------------------------------------------------
    tests = pd.DataFrame([mcnemar_row(cid, a, b, correct) for cid, a, b in COMPARISONS])
    tests.to_csv(TABLES / "mcnemar_tests.csv", index=False)

    # --- Finish line -----------------------------------------------------------
    sil_acc = summary.loc["cnn_silhouette", "top1"]
    tb = tests.set_index("comparison").loc["b"]
    beats_knn = bool(tb["p_value"] < ALPHA and tb["top1_a"] > tb["top1_b"])
    passed = bool(sil_acc >= GOAL and beats_knn)
    pd.DataFrame([{"criterion": "silhouette CNN top-1 >= 70%", "value": sil_acc,
                   "met": bool(sil_acc >= GOAL)},
                  {"criterion": "silhouette CNN > k-NN silhouette, McNemar p < 0.05",
                   "value": tb["p_value"], "met": beats_knn},
                  {"criterion": "OVERALL", "value": np.nan, "met": passed}]) \
        .to_csv(TABLES / "goal_check.csv", index=False)

    # --- Error analysis --------------------------------------------------------
    for m, fname in [("cnn_color", "confused_pairs_color.csv"),
                     ("cnn_silhouette", "confused_pairs_silhouette.csv"),
                     ("knn_color", "confused_pairs_knn_color.csv"),
                     ("knn_silhouette", "confused_pairs_knn_silhouette.csv")]:
        confused_pairs(test, *preds[m], names, types).to_csv(TABLES / fname, index=False)
    dis = disagreements(test, preds, names)
    dis.to_csv(TABLES / "disagreements_color_vs_silhouette.csv", index=False)
    by_type = pd.DataFrame({LABEL[m]: pd.Series(correct[m]).groupby(test["primary_type"]).mean()
                            for m in MODELS_ORDER})
    by_type.insert(0, "n_test", test["primary_type"].value_counts())
    by_type.sort_values("n_test", ascending=False).round(4) \
           .to_csv(TABLES / "accuracy_by_primary_type.csv")

    # --- Figures ---------------------------------------------------------------
    fig_accuracy(summary, cis)
    fig_confusion_pokedex(y, preds)
    fig_confusion_type(test, preds, types)

    # --- Console report --------------------------------------------------------
    pd.set_option("display.width", 140)
    print("\n=== HOME test results (n = 151) ===")
    print(summary[["label", "top1", "top5", "macro_f1", "top1_ci_low", "top1_ci_high"]]
          .round(4).to_string())
    print("\n=== McNemar tests (exact, two-sided) ===")
    print(tests[["comparison", "model_a", "model_b", "a_correct_b_wrong", "a_wrong_b_correct",
                 "p_value", "significant_at_0.05"]].to_string(index=False))
    print("\n=== Paired bootstrap 95% CIs ===")
    print(cis[["quantity", "estimate", "ci_low_2.5", "ci_high_97.5"]].round(4)
          .to_string(index=False))
    print(f"\nDisagreements: {(dis['pattern'] == 'color right, silhouette wrong').sum()} color-only, "
          f"{(dis['pattern'] == 'silhouette right, color wrong').sum()} silhouette-only")
    print("\n" + "=" * 60)
    print(f"FINISH LINE: silhouette CNN top-1 = {100 * sil_acc:.2f}% "
          f"({'>=' if sil_acc >= GOAL else '<'} 70%); vs k-NN silhouette p = {tb['p_value']:.3g}")
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
