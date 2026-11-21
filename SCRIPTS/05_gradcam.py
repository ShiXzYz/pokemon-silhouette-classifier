"""
05_gradcam.py - Grad-CAM heatmaps for the color and silhouette CNNs
Project: DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors: Brian Ryu, Keegan Jewell, Nathan Nguyen

Purpose: Show where each CNN looks. Grad-CAM on ResNet-18 layer4 for 12 HOME
         Pokemon (seed 4002: 4 both right, 3 color-only, 3 silhouette-only,
         2 both wrong), plus the share of heat on the Pokemon vs background.
Inputs:  MODEL/cnn_*_best.pt (03); OUTPUT/tables/test_predictions.csv (04)
Outputs: OUTPUT/figures/gradcam/ grids + per-Pokemon figures
         (color versions show artwork -> gitignored);
         OUTPUT/tables/gradcam_selection.csv
Run:     python SCRIPTS/05_gradcam.py
Runtime: ~5 s
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cnn_utils import (FIGURES, MODELS, ROOT, SEED, TABLES, get_device,
                       load_checkpoint, load_images, load_manifest,
                       to_model_input)

OUT = FIGURES / "gradcam"
# Pokemon drawn per outcome group (12 total)
GROUPS = [("both correct", 4), ("color only correct", 3),
          ("silhouette only correct", 3), ("both wrong", 2)]
INK, INK_2 = "#0b0b0b", "#52514e"
OK_COLOR, BAD_COLOR = "#1baf7a", "#c2361f"
plt.rcParams.update({"font.size": 7, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})


class GradCAM:
    """Grad-CAM: weight layer4 maps by their average gradient, ReLU, normalize."""

    def __init__(self, model):
        self.model = model
        self.acts = self.grads = None
        # Capture layer4 output on each forward pass
        model.layer4.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inp, out):
        self.acts = out                                   # (1, 512, 7, 7)
        # Capture its gradient on backward
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        """Return (224x224 heatmap, class index) for one image."""
        self.model.zero_grad()
        logits = self.model(x)                    # forward
        # Default: explain the predicted class
        target = int(logits.argmax(1)) if target is None else target
        logits[0, target].backward()              # backward
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # (1, 512, 1, 1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)
        # Upsample to image size
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        # Scale to [0, 1]
        return cam / cam.max() if cam.max() > 0 else cam, target


def choose_pokemon(preds, test):
    """Seeded pick of 12 Pokemon across the four outcome groups."""
    # Correct? per Pokemon per model
    wide = preds.pivot(index="pokedex_id", columns="model", values="correct_top1")
    c, s = wide["cnn_color"], wide["cnn_silhouette"]
    # Four outcome groups
    pools = {"both correct": wide.index[c & s], "color only correct": wide.index[c & ~s],
             "silhouette only correct": wide.index[~c & s], "both wrong": wide.index[~c & ~s]}
    rng = np.random.default_rng(SEED)
    chosen, short = [], 0
    for group, k in GROUPS:
        pool = np.array(sorted(pools[group]))     # sorted for a stable draw
        take = min(k, len(pool))
        short += k - take
        chosen += [(int(p), group) for p in rng.choice(pool, take, replace=False)]
    if short:  # fill any shortfall from "both correct"
        rest = np.array(sorted(set(pools["both correct"]) - {p for p, _ in chosen}))
        chosen += [(int(p), "both correct") for p in rng.choice(rest, short, replace=False)]
    # Sort by group, then Pokedex #
    order = {g: i for i, (g, _) in enumerate(GROUPS)}
    return sorted(chosen, key=lambda t: (order[t[1]], t[0]))


def draw(ax_img, ax_cam, img, cam, title, ok):
    """Input image (left) and heatmap overlay (right)."""
    show = img.permute(1, 2, 0).numpy()                   # (C, H, W) -> (H, W, C)
    show = show.squeeze(-1) if show.shape[-1] == 1 else show   # silhouettes: 2-D
    ax_img.imshow(show, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_cam.imshow(show, cmap="gray", vmin=0, vmax=255, alpha=0.35, interpolation="nearest")
    # Heatmap: bright = most influential
    ax_cam.imshow(cam, cmap="inferno", alpha=0.6, vmin=0, vmax=1)
    for ax in (ax_img, ax_cam):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#e4e3df")
    ax_img.set_title(title, loc="left", fontsize=6.5, color=OK_COLOR if ok else BAD_COLOR,
                     fontweight="bold")


def main():
    device = get_device()
    # Labels, test rows, and 04's predictions
    df = load_manifest()
    names = dict(zip(df["pokedex_id"], df["name"]))
    test = df[df["split"] == "test"].set_index("pokedex_id")
    preds = pd.read_csv(TABLES / "test_predictions.csv")
    chosen = choose_pokemon(preds, test)
    sub = test.loc[[p for p, _ in chosen]].reset_index()   # the 12 picks

    rows = {p: {"pokedex_id": p, "name": names[p], "group": g} for p, g in chosen}
    # Same 12 Pokemon for both models
    for mode in ["color", "silhouette"]:
        model, _ = load_checkpoint(MODELS / f"cnn_{mode}_best.pt", device)
        cammer = GradCAM(model)
        images = load_images(sub, mode)
        # Pokemon pixel masks
        masks = load_images(sub, "silhouette")[:, 0].numpy() == 0
        (OUT / mode).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 8, figsize=(9.6, 4.0),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.35})
        for i, (pid, group) in enumerate(chosen):
            x = to_model_input(images[i:i + 1]).to(device)
            cam, pred_idx = cammer(x)
            pred = pred_idx + 1                           # index -> Pokedex #
            ok = pred == pid
            # Share of heat on the Pokemon
            inside = float(cam[masks[i]].sum() / cam.sum()) if cam.sum() > 0 else np.nan
            rows[pid].update({f"{mode}_pred": names[pred], f"{mode}_correct": ok,
                              f"{mode}_cam_share_on_pokemon": round(inside, 4)})
            title = f"#{pid:03d} {names[pid]}" + ("" if ok else f"\n→ {names[pred]}")

            # Single-Pokemon figure
            f1, (a1, a2) = plt.subplots(1, 2, figsize=(3.0, 1.7))
            draw(a1, a2, images[i], cam, title, ok)
            f1.savefig(OUT / mode / f"gradcam_{mode}_{pid:03d}_{names[pid]}.png")
            plt.close(f1)
            # Panel in the grid (4 per row)
            r, c = divmod(i, 4)
            draw(axes[r, 2 * c], axes[r, 2 * c + 1], images[i], cam, title, ok)
        fig.suptitle(f"Grad-CAM (layer4) for the {mode} CNN on HOME test images, "
                     f"predicted class. Green = correct, red = wrong (→ prediction)",
                     x=0.12, ha="left", fontsize=8, fontweight="bold")
        fig.savefig(OUT / f"gradcam_grid_{mode}.png")
        plt.close(fig)

    # Save and summarize
    sel = pd.DataFrame(rows.values())
    sel.to_csv(TABLES / "gradcam_selection.csv", index=False)
    print(sel.to_string(index=False))
    print("\nMean share of Grad-CAM mass on the Pokemon: "
          f"color {sel['color_cam_share_on_pokemon'].mean():.3f}, "
          f"silhouette {sel['silhouette_cam_share_on_pokemon'].mean():.3f}")
    print(f"Saved figures to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
