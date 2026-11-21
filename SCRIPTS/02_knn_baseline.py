"""
02_knn_baseline.py - 1-nearest-neighbor raw-pixel baseline
Project: DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors: Brian Ryu, Keegan Jewell, Nathan Nguyen

Purpose: Baseline the CNNs must beat. Classify each HOME image by its nearest
         training image (64x64 pixels, Euclidean distance), for color and
         silhouette. No tuning, so no validation set is used.
Inputs:  DATA/manifest.csv, DATA/images/ (from 00)
Outputs: OUTPUT/tables/knn_test_predictions.csv (top-5 per test image)
Run:     python SCRIPTS/02_knn_baseline.py
Runtime: ~5 s
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent       # repo root
MANIFEST = ROOT / "DATA" / "manifest.csv"
OUT = ROOT / "OUTPUT" / "tables" / "knn_test_predictions.csv"
SIZE = 64   # downsample size
TOPK = 5


def load(paths, mode):
    """Images -> flattened 64x64 vectors in [0, 1] (RGB or grayscale)."""
    # Open, convert, shrink (BOX averaging), flatten, scale to 0-1
    X = [np.asarray(Image.open(ROOT / p).convert(mode).resize((SIZE, SIZE), Image.BOX),
                    dtype=np.float32).ravel() / 255.0
         for p in paths]
    return np.stack(X)          # one row per image


def topk_predictions(clf, y_train, X, k=TOPK):
    """Top-k classes per image, ranked by distance to each class's nearest image."""
    _, idx = clf.kneighbors(X, n_neighbors=clf.n_samples_fit_)   # all neighbors, sorted
    # Neighbor labels in distance order, de-duplicated, first k kept
    return np.array([list(dict.fromkeys(row))[:k] for row in y_train[idx]])


def main():
    # Train = Gen I-IV styles, test = HOME
    df = pd.read_csv(MANIFEST)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    n_classes = df["pokedex_id"].nunique()
    y_train = train["pokedex_id"].to_numpy()
    y_test = test["pokedex_id"].to_numpy()

    # Sizes and chance levels
    print(f"Train: {len(train)} images, test (HOME): {len(test)} images, {n_classes} classes")
    print(f"Random guess top-1: {1 / n_classes:.4f} ({100 / n_classes:.2f}%)  "
          f"top-5: {5 / n_classes:.4f} ({500 / n_classes:.2f}%)")

    rows = []
    # Run once on color, once on silhouettes
    for kind, col, mode in [("color", "color_path", "RGB"), ("silhouette", "silhouette_path", "L")]:
        clf = KNeighborsClassifier(n_neighbors=1)   # k = 1
        clf.fit(load(train[col], mode), y_train)
        top5 = topk_predictions(clf, y_train, load(test[col], mode))
        top1_acc = (top5[:, 0] == y_test).mean()
        top5_acc = (top5 == y_test[:, None]).any(axis=1).mean()
        print(f"k-NN (k=1) {kind:<10} top-1: {top1_acc:.4f} ({100 * top1_acc:.2f}%)  "
              f"top-5: {top5_acc:.4f} ({100 * top5_acc:.2f}%)")
        # One row per test image, for pairing with CNNs in 04
        for image_id, y, preds in zip(test["image_id"], y_test, top5):
            rows.append({"model": f"knn_{kind}", "image_id": image_id, "pokedex_id": y,
                         **{f"pred_{r + 1}": int(p) for r, p in enumerate(preds)}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
