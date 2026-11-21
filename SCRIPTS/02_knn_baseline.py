"""
=============================================================================
02_knn_baseline.py - k-nearest-neighbor (k = 1) raw-pixel baseline
=============================================================================
Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen

PURPOSE
    Non-deep-learning baseline from our MI2 plan. Each image is downsampled to
    64x64, flattened to a pixel vector, and classified by its single nearest
    training image (Euclidean distance). Run once on color images and once on
    silhouettes. The CNNs must beat this to show they learned more than
    pixel-level template matching.

    Top-5 for 1-NN: classes are ranked by the distance to their nearest
    training image, i.e. the first 5 distinct labels among the sorted
    neighbors.

    The baseline has no hyperparameters to tune (k = 1 is fixed by the MI2
    plan), so it is fit on the full train split (all 12 Gen I-IV styles) and
    predicts HOME directly. Metrics and significance tests are computed in
    04_evaluate.py from the predictions saved here.

INPUTS
    - DATA/manifest.csv and the 224x224 images from 00_build_dataset.py

OUTPUTS
    - OUTPUT/tables/knn_test_predictions.csv  one row per (model, HOME image):
      true label and the top-5 predicted Pokedex IDs
    - console: top-1 / top-5 accuracy for both modes

HOW TO RUN (from the repository root)
    python SCRIPTS/02_knn_baseline.py

EXPECTED RUNTIME
    ~5 s on a laptop CPU.
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent       # repo root, parent of SCRIPTS/
MANIFEST = ROOT / "DATA" / "manifest.csv"
OUT = ROOT / "OUTPUT" / "tables" / "knn_test_predictions.csv"
SIZE = 64   # 64x64 keeps the vectors small (12,288 color / 4,096 silhouette dims)
TOPK = 5


def load(paths, mode):
    """Load images as flattened float vectors in [0, 1].

    The 224px images are already nearest/BOX-resized; area averaging (BOX) is
    the cleanest downsample to 64px for both pixel sprites and artwork.
    mode="RGB" keeps 3 channels; mode="L" gives one channel for silhouettes.
    """
    X = [np.asarray(Image.open(ROOT / p).convert(mode).resize((SIZE, SIZE), Image.BOX),
                    dtype=np.float32).ravel() / 255.0
         for p in paths]
    return np.stack(X)


def topk_predictions(clf, y_train, X, k=TOPK):
    """Return an (n, k) array of predicted classes, best first.

    We ask for *all* training neighbors sorted by distance and keep the first k
    distinct labels. With k = 1 this ranks each class by its closest training
    image. KNeighborsClassifier breaks distance ties by training-set order, so
    the result is deterministic.
    """
    _, idx = clf.kneighbors(X, n_neighbors=clf.n_samples_fit_)
    return np.array([list(dict.fromkeys(row))[:k] for row in y_train[idx]])


def main():
    df = pd.read_csv(MANIFEST)
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    n_classes = df["pokedex_id"].nunique()
    y_train = train["pokedex_id"].to_numpy()
    y_test = test["pokedex_id"].to_numpy()

    print(f"Train: {len(train)} images, test (HOME): {len(test)} images, {n_classes} classes")
    print(f"Random guess top-1: {1 / n_classes:.4f} ({100 / n_classes:.2f}%)  "
          f"top-5: {5 / n_classes:.4f} ({500 / n_classes:.2f}%)")

    rows = []
    for kind, col, mode in [("color", "color_path", "RGB"), ("silhouette", "silhouette_path", "L")]:
        # Fit = memorize the training vectors; k = 1 per the MI2 plan.
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(load(train[col], mode), y_train)
        top5 = topk_predictions(clf, y_train, load(test[col], mode))
        top1_acc = (top5[:, 0] == y_test).mean()
        top5_acc = (top5 == y_test[:, None]).any(axis=1).mean()
        print(f"k-NN (k=1) {kind:<10} top-1: {top1_acc:.4f} ({100 * top1_acc:.2f}%)  "
              f"top-5: {top5_acc:.4f} ({100 * top5_acc:.2f}%)")
        for image_id, y, preds in zip(test["image_id"], y_test, top5):
            rows.append({"model": f"knn_{kind}", "image_id": image_id, "pokedex_id": y,
                         **{f"pred_{r + 1}": int(p) for r, p in enumerate(preds)}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
