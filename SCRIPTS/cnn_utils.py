"""
cnn_utils.py - Shared helpers for 03_train_cnn.py, 04_evaluate.py, 05_gradcam.py
Project: DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors: Brian Ryu, Keegan Jewell, Nathan Nguyen

Purpose: one place for paths, seed, model, preprocessing and device so all
         three scripts stay consistent. Imported only; not run directly.
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

# Paths, relative to the repo root
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "DATA" / "manifest.csv"
MODELS = ROOT / "MODEL"                  # trained weights
TABLES = ROOT / "OUTPUT" / "tables"
FIGURES = ROOT / "OUTPUT" / "figures"

SEED = 4002
N_CLASSES = 151                          # Gen I Pokedex #1-151
GEN1_STYLES = ["red-blue", "red-green-japan", "yellow"]   # dropped by --exclude-gen1

# ImageNet normalization, used for both color and silhouette inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds CUDA/MPS


def get_device():
    """CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    """Readable device name for logs."""
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "mps (Apple-silicon GPU)"
    return "cpu"


def build_model(pretrained=True):
    """ResNet-18 (ImageNet weights) with a new 151-class output layer."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def load_manifest():
    """Read the manifest; add a 0-based class label."""
    df = pd.read_csv(MANIFEST)
    df["label"] = df["pokedex_id"] - 1
    return df


def load_images(df, mode):
    """Load images into a uint8 tensor (N, C, 224, 224); C = 3 color, 1 silhouette."""
    col, pil_mode = ("color_path", "RGB") if mode == "color" else ("silhouette_path", "L")
    arrs = [np.asarray(Image.open(ROOT / p).convert(pil_mode)) for p in df[col]]
    x = torch.from_numpy(np.stack(arrs))
    return x.permute(0, 3, 1, 2).contiguous() if x.ndim == 4 else x.unsqueeze(1)


# Training augmentation: rotate +/-15 deg, scale 90-110%, shift <=5%, white fill.
# No flips or color jitter. Nearest-neighbor keeps silhouettes binary.
TRAIN_AUGMENT = v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1),
                                interpolation=v2.InterpolationMode.NEAREST, fill=255)
_NORMALIZE = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def to_model_input(x):
    """uint8 images -> normalized float, silhouettes copied to 3 channels."""
    x = x.float() / 255.0
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return _NORMALIZE(x)


class ImageTensorDataset(torch.utils.data.Dataset):
    """Preloaded images + labels, with optional per-image augmentation."""

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
    """Model scores for images (no augmentation), returned on CPU."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        xb = to_model_input(images[i:i + batch_size]).to(device)
        out.append(model(xb).float().cpu())
    return torch.cat(out)


def load_checkpoint(path, device):
    """Load a 03_train_cnn.py checkpoint; returns (model, metadata)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}
