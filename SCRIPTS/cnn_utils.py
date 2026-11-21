"""
=============================================================================
cnn_utils.py - Shared helpers for the CNN scripts (03, 04, 05)
=============================================================================
Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen

PURPOSE
    Helper module imported by 03_train_cnn.py, 04_evaluate.py and
    05_gradcam.py so that all three use exactly the same paths, seed, model
    architecture, image preprocessing and device selection. It is NOT meant to
    be run by itself.

    Keeping preprocessing in one place matters: if training and evaluation
    normalized images differently, test accuracy would silently drop.
=============================================================================
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import v2

# ---------------------------------------------------------------------------
# Paths (all relative to the repository root = parent of SCRIPTS/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "DATA" / "manifest.csv"
MODELS = ROOT / "models"                 # gitignored; weights are ~45 MB each
TABLES = ROOT / "OUTPUT" / "tables"
FIGURES = ROOT / "OUTPUT" / "figures"

SEED = 4002                              # same project seed as Project 1
N_CLASSES = 151                          # Gen I Pokedex #1-151
GEN1_STYLES = ["red-blue", "red-green-japan", "yellow"]   # dropped by --exclude-gen1

# ImageNet channel statistics. The ResNet-18 weights were trained on images
# normalized this way, so both color images and (3-channel) silhouettes use
# them; per the MI2 plan the two modes get identical preprocessing.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    """Seed every random number generator we use (Python, NumPy, PyTorch on
    CPU and all GPUs) so a run is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds CUDA and MPS generators


def get_device():
    """Use an NVIDIA GPU if present, else Apple-silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    """Human-readable device name for logs and the README."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple-silicon GPU)"
    return "cpu"


def build_model(pretrained=True):
    """ImageNet-pretrained ResNet-18 with its 1000-way head replaced by a
    151-way linear layer (one output per Gen I Pokemon). All layers are
    fine-tuned during training."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def load_manifest():
    """Read the manifest and add a 0-based class index (Pokedex # minus 1)."""
    df = pd.read_csv(MANIFEST)
    df["label"] = df["pokedex_id"] - 1
    return df


def load_images(df, mode):
    """Load every image in `df` into one uint8 tensor of shape (N, C, 224, 224).

    Color images keep 3 RGB channels; silhouettes are loaded as 1 grayscale
    channel and only expanded to 3 channels at the very end of preprocessing.
    The whole dataset is ~270 MB as uint8, so preloading it is faster than
    reading PNGs from disk every epoch.
    """
    col, pil_mode = ("color_path", "RGB") if mode == "color" else ("silhouette_path", "L")
    arrs = [np.asarray(Image.open(ROOT / p).convert(pil_mode)) for p in df[col]]
    x = torch.from_numpy(np.stack(arrs))
    return x.permute(0, 3, 1, 2).contiguous() if x.ndim == 4 else x.unsqueeze(1)


# Training-only augmentation from the MI2 plan: rotation +/-15 degrees, scale
# 90-110 %, translation up to 5 % of the image size. No flips, no color jitter.
# fill=255 pads exposed corners with white, matching the dataset background.
# Nearest-neighbor interpolation keeps silhouettes strictly black/white and
# pixel-art sprites crisp; the same settings are used for both modes.
TRAIN_AUGMENT = v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1),
                                interpolation=v2.InterpolationMode.NEAREST, fill=255)
_NORMALIZE = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def to_model_input(x):
    """uint8 (N, C, H, W) -> normalized float (N, 3, H, W).

    Silhouettes (C = 1) are replicated to 3 channels so they fit the
    ImageNet-pretrained first convolution, then both modes are scaled to
    [0, 1] and normalized with the ImageNet mean/std.
    """
    x = x.float() / 255.0
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return _NORMALIZE(x)


class ImageTensorDataset(torch.utils.data.Dataset):
    """Wraps preloaded uint8 images + labels; optionally applies augmentation.
    Augmentation runs per image (each gets its own random transform)."""

    def __init__(self, images, labels, augment=False):
        self.images, self.labels, self.augment = images, labels, augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augment:
            img = TRAIN_AUGMENT(img)
        return img, self.labels[i]


@torch.no_grad()
def predict_logits(model, images, device, batch_size=64):
    """Run the model over uint8 images (no augmentation) and return CPU logits."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        xb = to_model_input(images[i:i + batch_size]).to(device)
        out.append(model(xb).float().cpu())
    return torch.cat(out)


def load_checkpoint(path, device):
    """Load a checkpoint saved by 03_train_cnn.py; returns (model, metadata)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}
