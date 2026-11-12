"""k-NN (k=1) pixel baseline: train on the train split, evaluate on HOME (test).

Top-5 for 1-NN ranks classes by the distance to their nearest training image, i.e.
the classes of the first 5 distinct labels among the sorted neighbors.

Usage: python scripts/knn_baseline.py
Run scripts/build_dataset.py first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
SIZE = 64


def load(paths, mode):
    # The 224px images are already nearest/BOX-resized; area averaging is the
    # cleanest downsample to 64px for both pixel sprites and artwork.
    X = [np.asarray(Image.open(ROOT / p).convert(mode).resize((SIZE, SIZE), Image.BOX),
                    dtype=np.float32).ravel() / 255.0
         for p in paths]
    return np.stack(X)


def topk_accuracy(clf, y_train, X, y_true, k=5):
    _, idx = clf.kneighbors(X, n_neighbors=clf.n_samples_fit_)
    hits = []
    for row, y in zip(y_train[idx], y_true):
        top = list(dict.fromkeys(row))[:k]  # distinct classes in order of distance
        hits.append(y in top)
    return float(np.mean(hits))


def main():
    df = pd.read_csv(MANIFEST)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    n_classes = df["pokedex_id"].nunique()

    print(f"Train: {len(train)} images, test (HOME): {len(test)} images, {n_classes} classes")
    print(f"Random guess top-1: {1 / n_classes:.4f} ({100 / n_classes:.2f}%)  "
          f"top-5: {5 / n_classes:.4f} ({500 / n_classes:.2f}%)")
    for kind, col, mode in [("color", "color_path", "RGB"), ("silhouette", "silhouette_path", "L")]:
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(load(train[col], mode), train["pokedex_id"])
        X_test, y_test = load(test[col], mode), test["pokedex_id"].to_numpy()
        top1 = (clf.predict(X_test) == y_test).mean()
        top5 = topk_accuracy(clf, train["pokedex_id"].to_numpy(), X_test, y_test, k=5)
        print(f"k-NN (k=1) {kind:<10} top-1: {top1:.4f} ({100 * top1:.2f}%)  "
              f"top-5: {top5:.4f} ({100 * top5:.2f}%)")


if __name__ == "__main__":
    main()
