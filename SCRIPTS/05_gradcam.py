"""
=============================================================================
05_gradcam.py - Grad-CAM heatmaps for the color and silhouette CNNs
=============================================================================
Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen

PURPOSE
    Show WHERE each CNN looks when it classifies a HOME test image, using
    Grad-CAM (Selvaraju et al., 2017) on ResNet-18's last convolutional block
    (layer4, a 7x7 feature map). For 12 test Pokemon chosen with seed 4002 to
    mix outcomes (4 both CNNs right, 3 only color right, 3 only silhouette
    right, 2 both wrong), we compute the heatmap for each model's predicted
    class and overlay it on the input.

    As a simple numeric summary we also report, per image, the share of
    Grad-CAM mass that falls inside the Pokemon's outline (vs. background).

    This script only explains predictions that 04_evaluate.py already scored;
    it does not compute any accuracy numbers.

INPUTS
    - models/cnn_color_best.pt, models/cnn_silhouette_best.pt   (from 03)
    - OUTPUT/tables/test_predictions.csv                          (from 04)
    - DATA/manifest.csv and HOME test images                      (from 00)

OUTPUTS
    - OUTPUT/figures/gradcam/silhouette/gradcam_silhouette_<dex>_<name>.png
    - OUTPUT/figures/gradcam/gradcam_grid_silhouette.png   all 12 in one figure
    - OUTPUT/figures/gradcam/color/gradcam_color_<dex>_<name>.png      (gitignored:
    - OUTPUT/figures/gradcam/gradcam_grid_color.png                     shows artwork)
    - OUTPUT/tables/gradcam_selection.csv  chosen Pokemon, predictions, and
                                           share of Grad-CAM mass on the Pokemon

HOW TO RUN (from the repository root)
    python SCRIPTS/05_gradcam.py

EXPECTED RUNTIME
    ~5 s.
=============================================================================
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
# How many Pokemon to draw from each outcome group (12 total).
GROUPS = [("both correct", 4), ("color only correct", 3),
          ("silhouette only correct", 3), ("both wrong", 2)]
INK, INK_2 = "#0b0b0b", "#52514e"
OK_COLOR, BAD_COLOR = "#1baf7a", "#c2361f"
plt.rcParams.update({"font.size": 7, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "savefig.facecolor": "white"})


class GradCAM:
    """Minimal Grad-CAM: hooks capture layer4's activations A and the gradient
    of the target class score w.r.t. A. Channel weights = gradients averaged
    over space; CAM = ReLU(sum_k w_k * A_k), normalized to [0, 1]."""

    def __init__(self, model):
        self.model = model
        self.acts = self.grads = None
        model.layer4.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x, target=None):
        """x: one normalized (1, 3, 224, 224) image. Returns (cam 224x224, class idx)."""
        self.model.zero_grad()
        logits = self.model(x)
        target = int(logits.argmax(1)) if target is None else target
        logits[0, target].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # (1, 512, 1, 1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)
        # Upsample the 7x7 map to the input size so it can be overlaid.
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        return cam / cam.max() if cam.max() > 0 else cam, target


def choose_pokemon(preds, test):
    """Pick 12 test Pokemon spanning the four correctness patterns (seed 4002).
    If a group has too few members, the shortfall is filled from 'both correct'."""
    wide = preds.pivot(index="pokedex_id", columns="model", values="correct_top1")
    c, s = wide["cnn_color"], wide["cnn_silhouette"]
    pools = {"both correct": wide.index[c & s], "color only correct": wide.index[c & ~s],
             "silhouette only correct": wide.index[~c & s], "both wrong": wide.index[~c & ~s]}
    rng = np.random.default_rng(SEED)
    chosen, short = [], 0
    for group, k in GROUPS:
        pool = np.array(sorted(pools[group]))
        take = min(k, len(pool))
        short += k - take
        chosen += [(int(p), group) for p in rng.choice(pool, take, replace=False)]
    if short:  # top up from the remaining "both correct" Pokemon
        rest = np.array(sorted(set(pools["both correct"]) - {p for p, _ in chosen}))
        chosen += [(int(p), "both correct") for p in rng.choice(rest, short, replace=False)]
    order = {g: i for i, (g, _) in enumerate(GROUPS)}
    return sorted(chosen, key=lambda t: (order[t[1]], t[0]))


def draw(ax_img, ax_cam, img, cam, title, ok):
    """Left: the input image. Right: the same image faded, with the heatmap on top."""
    show = img.permute(1, 2, 0).numpy()
    show = show.squeeze(-1) if show.shape[-1] == 1 else show
    ax_img.imshow(show, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_cam.imshow(show, cmap="gray", vmin=0, vmax=255, alpha=0.35, interpolation="nearest")
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
    df = load_manifest()
    names = dict(zip(df["pokedex_id"], df["name"]))
    test = df[df["split"] == "test"].set_index("pokedex_id")
    preds = pd.read_csv(TABLES / "test_predictions.csv")
    chosen = choose_pokemon(preds, test)
    sub = test.loc[[p for p, _ in chosen]].reset_index()

    rows = {p: {"pokedex_id": p, "name": names[p], "group": g} for p, g in chosen}
    for mode in ["color", "silhouette"]:
        model, _ = load_checkpoint(MODELS / f"cnn_{mode}_best.pt", device)
        cammer = GradCAM(model)
        images = load_images(sub, mode)
        # Pokemon mask (from the silhouette) to measure CAM mass on the Pokemon.
        masks = load_images(sub, "silhouette")[:, 0].numpy() == 0
        (OUT / mode).mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 8, figsize=(9.6, 4.0),
                                 gridspec_kw={"wspace": 0.05, "hspace": 0.35})
        for i, (pid, group) in enumerate(chosen):
            x = to_model_input(images[i:i + 1]).to(device)
            cam, pred_idx = cammer(x)
            pred = pred_idx + 1
            ok = pred == pid
            inside = float(cam[masks[i]].sum() / cam.sum()) if cam.sum() > 0 else np.nan
            rows[pid].update({f"{mode}_pred": names[pred], f"{mode}_correct": ok,
                              f"{mode}_cam_share_on_pokemon": round(inside, 4)})
            title = f"#{pid:03d} {names[pid]}" + ("" if ok else f"\n→ {names[pred]}")

            # One standalone figure per Pokemon...
            f1, (a1, a2) = plt.subplots(1, 2, figsize=(3.0, 1.7))
            draw(a1, a2, images[i], cam, title, ok)
            f1.savefig(OUT / mode / f"gradcam_{mode}_{pid:03d}_{names[pid]}.png")
            plt.close(f1)
            # ...and one panel in the overview grid (4 Pokemon per row).
            r, c = divmod(i, 4)
            draw(axes[r, 2 * c], axes[r, 2 * c + 1], images[i], cam, title, ok)
        fig.suptitle(f"Grad-CAM (layer4) for the {mode} CNN on HOME test images, "
                     f"predicted class. Green = correct, red = wrong (→ prediction)",
                     x=0.12, ha="left", fontsize=8, fontweight="bold")
        fig.savefig(OUT / f"gradcam_grid_{mode}.png")
        plt.close(fig)

    sel = pd.DataFrame(rows.values())
    sel.to_csv(TABLES / "gradcam_selection.csv", index=False)
    print(sel.to_string(index=False))
    print("\nMean share of Grad-CAM mass on the Pokemon: "
          f"color {sel['color_cam_share_on_pokemon'].mean():.3f}, "
          f"silhouette {sel['silhouette_cam_share_on_pokemon'].mean():.3f}")
    print(f"Saved figures to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
