"""
=============================================================================
03_train_cnn.py - Fine-tune ResNet-18 on color or silhouette images
=============================================================================
Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen

PURPOSE
    Train the two CNNs in our MI2 plan (one on color images, one on
    silhouettes) and pick the final configuration of each using the
    VALIDATION split (Black/White sprites) only. The test split (HOME) is never
    loaded here; it is touched only in 04_evaluate.py.

    Training mode (--mode):
      * Fine-tunes an ImageNet-pretrained torchvision ResNet-18 with a new
        151-class output layer.
      * Runs the learning-rate grid {1e-4, 3e-4, 1e-3}. Each run trains with
        early stopping on validation top-1 (patience 5, max 30 epochs) and
        keeps the weights from its best epoch.
      * --exclude-gen1 drops the three Gen I styles (red-blue,
        red-green-japan, yellow) from TRAINING only. Their silhouettes are the
        most fragmented in the dataset (see EDA), so we test whether they help.
      * Every hyperparameter except the learning rate is fixed and identical
        for color and silhouette (batch 32, AdamW, weight decay 1e-4,
        cross-entropy, same augmentation, same seed 4002).

    Selection mode (--select), after all 12 runs (2 modes x 2 Gen I settings
    x 3 learning rates):
      1. Choose ONE learning rate shared by both modes, so the final color and
         silhouette models have identical hyperparameters: the LR with the
         highest mean val top-1 over its four runs (ties -> lower mean val
         loss).
      2. For each mode, at that LR, keep the Gen I setting with the higher val
         top-1 (ties -> higher val macro F1 -> lower val loss -> keep Gen I,
         the MI2 default). Ties matter because the color CNN can reach 100 %
         val top-1, where accuracy alone cannot tell runs apart.
      3. Copy the winners to models/cnn_<mode>_best.pt.

INPUTS
    - DATA/manifest.csv and images from 00_build_dataset.py (train + val only)
    - ImageNet ResNet-18 weights (downloaded once by torchvision, ~45 MB)

OUTPUTS
    - models/cnn_<mode>_<gen1|nogen1>_lr<lr>.pt   one per run       (gitignored)
    - models/cnn_<mode>_best.pt                   selected models   (gitignored)
    - OUTPUT/tables/val_runs.csv          one row per run: val top-1/top-5/F1,
                                          best epoch, runtime, device
    - OUTPUT/tables/training_history.csv  per-epoch loss/accuracy for every run
    - OUTPUT/tables/selected_models.csv   the configuration chosen per mode

HOW TO RUN (from the repository root)
    python SCRIPTS/03_train_cnn.py --mode color
    python SCRIPTS/03_train_cnn.py --mode color --exclude-gen1
    python SCRIPTS/03_train_cnn.py --mode silhouette
    python SCRIPTS/03_train_cnn.py --mode silhouette --exclude-gen1
    python SCRIPTS/03_train_cnn.py --select
    (SCRIPTS/run_all.sh runs all five commands.)

EXPECTED RUNTIME
    ~6 s per epoch on an Apple M2 Max GPU (MPS). Each training command runs
    3 learning rates x 8-15 epochs (early stopping) = 2-5 min; all four
    commands took 14 min in our reproduction run. Expect roughly 5-10x longer
    on a laptop CPU.
=============================================================================
"""

import argparse
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

# sklearn warns that 151 classes in 151 samples "could be regression"; harmless.
warnings.filterwarnings("ignore", message="The number of unique classes")

from cnn_utils import (GEN1_STYLES, MODELS, N_CLASSES, ROOT, SEED, TABLES,
                       ImageTensorDataset, build_model, describe_device,
                       get_device, load_images, load_manifest, predict_logits,
                       set_seed, to_model_input)

# ---------------------------------------------------------------------------
# Fixed hyperparameters (identical for color and silhouette runs)
# ---------------------------------------------------------------------------
LR_GRID = [1e-4, 3e-4, 1e-3]   # the only hyperparameter tuned on validation
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5                   # stop after 5 epochs without a new best val top-1

VAL_RUNS = TABLES / "val_runs.csv"
HISTORY = TABLES / "training_history.csv"
SELECTED = TABLES / "selected_models.csv"


def run_name(mode, exclude_gen1, lr):
    """Short unique name for one run, used in file names and tables."""
    return f"cnn_{mode}_{'nogen1' if exclude_gen1 else 'gen1'}_lr{lr:g}"


def evaluate(model, images, labels, device):
    """Return val top-1, top-5, macro F1 and mean cross-entropy loss."""
    logits = predict_logits(model, images, device)
    y = labels.numpy()
    top5 = logits.topk(5, dim=1).indices.numpy()
    return {
        "top1": float((top5[:, 0] == y).mean()),
        "top5": float((top5 == y[:, None]).any(axis=1).mean()),
        # labels=range(151) so a class that is never predicted still counts (F1 = 0)
        "macro_f1": float(f1_score(y, top5[:, 0], average="macro",
                                   labels=np.arange(N_CLASSES), zero_division=0)),
        "loss": float(nn.functional.cross_entropy(logits, labels)),
    }


def train_one(mode, exclude_gen1, lr, data, device, max_epochs, patience):
    """Train one configuration with early stopping; return (summary, history)."""
    name = run_name(mode, exclude_gen1, lr)
    x_tr, y_tr, x_val, y_val = data
    print(f"\n--- {name}: {len(y_tr)} train / {len(y_val)} val images ---")

    # Re-seed at the start of every run so each run's initialization, shuffling
    # and augmentation are identical regardless of which runs came before it.
    set_seed(SEED)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    # A dedicated generator makes the shuffle order reproducible.
    loader = torch.utils.data.DataLoader(
        ImageTensorDataset(x_tr, y_tr, augment=True), batch_size=BATCH_SIZE,
        shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)

    best, best_state, bad_epochs, history = None, None, 0, []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        model.train()
        tot_loss, tot_correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = to_model_input(xb).to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)

        val = evaluate(model, x_val, y_val, device)
        history.append({"run": name, "mode": mode, "exclude_gen1": exclude_gen1, "lr": lr,
                        "epoch": epoch, "train_loss": tot_loss / n, "train_top1": tot_correct / n,
                        "val_loss": val["loss"], "val_top1": val["top1"], "val_top5": val["top5"],
                        "val_macro_f1": val["macro_f1"], "epoch_seconds": time.time() - t_ep})
        improved = best is None or val["top1"] > best["top1"]
        print(f"  epoch {epoch:2d}  train loss {tot_loss / n:.3f} acc {tot_correct / n:.3f} | "
              f"val top-1 {val['top1']:.3f} top-5 {val['top5']:.3f} F1 {val['macro_f1']:.3f}"
              f"{'  *' if improved else ''}  ({time.time() - t_ep:.1f}s)")
        if improved:
            # Keep a CPU copy of the best weights (early stopping restores these).
            best, bad_epochs = {**val, "epoch": epoch}, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop: no val improvement for {patience} epochs")
                break

    seconds = time.time() - t0
    MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "mode": mode, "exclude_gen1": exclude_gen1, "lr": lr,
                "best_epoch": best["epoch"], "val_top1": best["top1"], "seed": SEED},
               MODELS / f"{name}.pt")
    summary = {"run": name, "mode": mode, "exclude_gen1": exclude_gen1, "lr": lr,
               "n_train": len(y_tr), "best_epoch": best["epoch"], "epochs_run": epoch,
               "val_top1": best["top1"], "val_top5": best["top5"],
               "val_macro_f1": best["macro_f1"], "val_loss": best["loss"],
               "train_seconds": round(seconds, 1), "device": describe_device(device)}
    print(f"  best epoch {best['epoch']}: val top-1 {best['top1']:.4f}  ({seconds / 60:.1f} min)")
    return summary, history


def upsert(path, new_rows, key):
    """Write rows to a CSV, replacing any existing rows for the same runs so a
    re-run overwrites its old results instead of duplicating them."""
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    new.sort_values([c for c in ["mode", "exclude_gen1", "lr", "epoch"] if c in new]) \
       .to_csv(path, index=False)


def train_mode(args):
    device = get_device()
    print(f"Device: {describe_device(device)}   torch {torch.__version__}")
    df = load_manifest()
    # Only train and validation rows are used. HOME (test) is never loaded here.
    train = df[df["split"] == "train"]
    if args.exclude_gen1:
        train = train[~train["source_style"].isin(GEN1_STYLES)]
    val = df[df["split"] == "val"]
    data = (load_images(train, args.mode), torch.tensor(train["label"].to_numpy()),
            load_images(val, args.mode), torch.tensor(val["label"].to_numpy()))

    TABLES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for lr in args.lrs:
        summary, history = train_one(args.mode, args.exclude_gen1, lr, data, device,
                                     args.max_epochs, args.patience)
        # Save after every run so partial progress survives an interruption.
        upsert(VAL_RUNS, [summary], "run")
        upsert(HISTORY, history, "run")
    print(f"\nFinished {len(args.lrs)} runs in {(time.time() - t0) / 60:.1f} min. "
          f"Results in {VAL_RUNS.relative_to(ROOT)}")


def select():
    """Pick the shared learning rate and per-mode Gen I setting from val_runs.csv."""
    runs = pd.read_csv(VAL_RUNS)
    expected = {run_name(m, g, lr) for m in ["color", "silhouette"]
                for g in [False, True] for lr in LR_GRID}
    missing = expected - set(runs["run"])
    if missing:
        raise SystemExit(f"Missing runs in {VAL_RUNS.name}: {sorted(missing)}")
    runs = runs[runs["run"].isin(expected)]

    # Step 1: one learning rate for both modes = best mean val top-1 over its 4
    # runs; ties broken by lower mean val loss.
    by_lr = runs.groupby("lr")[["val_top1", "val_loss"]].mean() \
                .sort_values(["val_top1", "val_loss"], ascending=[False, True])
    print("Mean over color/silhouette x Gen I settings, by learning rate:")
    print(by_lr.round(4).to_string())
    lr = by_lr.index[0]
    print(f"-> shared learning rate: {lr:g}")

    rows = []
    for mode in ["color", "silhouette"]:
        # Step 2: at that LR, the Gen I setting with higher val top-1 wins.
        cand = runs[(runs["mode"] == mode) & (runs["lr"] == lr)].copy()
        # Tie-breaks: macro F1, then val loss, then prefer keeping Gen I
        # (exclude_gen1 False sorts first).
        cand = cand.sort_values(["val_top1", "val_macro_f1", "val_loss", "exclude_gen1"],
                                ascending=[False, False, True, True])
        win = cand.iloc[0]
        print(f"{mode}: " + "  vs  ".join(
            f"{'without' if r.exclude_gen1 else 'with'} Gen I = {r.val_top1:.4f} "
            f"(loss {r.val_loss:.3f})"
            for r in cand.itertuples()) + f"  -> {win['run']}")
        # Step 3: copy the winning weights to a fixed name for 04/05.
        shutil.copy(MODELS / f"{win['run']}.pt", MODELS / f"cnn_{mode}_best.pt")
        rows.append({"model": f"cnn_{mode}", "selected_run": win["run"], "lr": lr,
                     "exclude_gen1": bool(win["exclude_gen1"]),
                     "best_epoch": int(win["best_epoch"]), "val_top1": win["val_top1"],
                     "val_top5": win["val_top5"], "val_macro_f1": win["val_macro_f1"],
                     "weights": f"models/cnn_{mode}_best.pt"})
    pd.DataFrame(rows).to_csv(SELECTED, index=False)
    print(f"Saved {SELECTED.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("PURPOSE")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["color", "silhouette"], help="input image type")
    p.add_argument("--exclude-gen1", action="store_true",
                   help="drop red-blue, red-green-japan and yellow from training")
    p.add_argument("--select", action="store_true",
                   help="choose final models from OUTPUT/tables/val_runs.csv")
    # The options below default to the MI2 plan; they exist for quick smoke tests.
    p.add_argument("--lrs", type=float, nargs="+", default=LR_GRID)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    args = p.parse_args()
    if args.select:
        select()
    elif args.mode:
        train_mode(args)
    else:
        p.error("give --mode {color,silhouette} or --select")


if __name__ == "__main__":
    main()
