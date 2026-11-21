"""
00_build_dataset.py - Build the Gen I color + silhouette image dataset
Project: DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors: Brian Ryu, Keegan Jewell, Nathan Nguyen

Purpose: Download Gen I sprites (#1-151) in 15 art styles, crop/pad/resize to
         224x224, save color + silhouette versions, assign splits by style
         (train = Gen I-IV, val = Black/White, test = HOME), write the manifest.
Inputs:  PokeAPI sprites repo (pinned commit); DATA/pokeapi_cache.json
Outputs: DATA/raw/, DATA/images/ (gitignored); DATA/manifest.csv
Run:     python SCRIPTS/00_build_dataset.py
Runtime: ~30 s (longer on a slow connection)
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

# Paths, relative to the repo root
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "DATA"
REPO = DATA / "raw" / "pokeapi-sprites"
CACHE = DATA / "pokeapi_cache.json"
MANIFEST = DATA / "manifest.csv"
OUT_COLOR = DATA / "images" / "color"
OUT_SIL = DATA / "images" / "silhouette"

# Pinned sprites commit so rebuilds are identical
SPRITES_URL = "https://github.com/PokeAPI/sprites.git"
SPRITES_COMMIT = "1b3b7c0a538dd2226ae08d0d2f8955125075bbd4"

N_POKEMON = 151          # Gen I only
SIZE = 224               # ResNet-18 input size
PIXEL_ART_MAX_SIDE = 128 # sprites this small are upscaled with nearest-neighbor
CROP_MARGIN = 0.05       # margin around the cropped Pokemon

V = "sprites/pokemon/versions"
# (style, generation, split, folder). Gen I/II use transparent/ folders (the
# defaults have no alpha). Official artwork is excluded (too close to HOME).
SOURCES = [
    ("official-artwork", None, "excluded", "sprites/pokemon/other/official-artwork"),
    ("red-blue", 1, "train", f"{V}/generation-i/red-blue/transparent"),
    ("red-green-japan", 1, "train", f"{V}/generation-i/red-green-japan"),
    ("yellow", 1, "train", f"{V}/generation-i/yellow/transparent"),
    ("gold", 2, "train", f"{V}/generation-ii/gold/transparent"),
    ("silver", 2, "train", f"{V}/generation-ii/silver/transparent"),
    ("crystal", 2, "train", f"{V}/generation-ii/crystal/transparent"),
    ("ruby-sapphire", 3, "train", f"{V}/generation-iii/ruby-sapphire"),
    ("emerald", 3, "train", f"{V}/generation-iii/emerald"),
    ("firered-leafgreen", 3, "train", f"{V}/generation-iii/firered-leafgreen"),
    ("diamond-pearl", 4, "train", f"{V}/generation-iv/diamond-pearl"),
    ("platinum", 4, "train", f"{V}/generation-iv/platinum"),
    ("heartgold-soulsilver", 4, "train", f"{V}/generation-iv/heartgold-soulsilver"),
    ("black-white", 5, "val", f"{V}/generation-v/black-white"),
    ("home", None, "test", "sprites/pokemon/other/home"),
]


def git(*args, **kw):
    """Run a git command in the sprites repo."""
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, **kw)


def ensure_repo():
    """Sparse-clone only our 15 folders at the pinned commit (skip if present)."""
    if REPO.exists():
        head = git("rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
        if head != SPRITES_COMMIT:
            print(f"WARNING: {REPO.relative_to(ROOT)} is at {head[:10]}, not the pinned "
                  f"commit {SPRITES_COMMIT[:10]}. Delete DATA/raw/ to rebuild exactly.")
        return
    REPO.parent.mkdir(parents=True, exist_ok=True)
    # Clone without file contents or checkout (full repo is several GB)
    subprocess.run(
        ["git", "clone", "--no-checkout", "--depth", "1", "--filter=blob:none",
         "--sparse", SPRITES_URL, str(REPO)],
        check=True,
    )
    # Limit checkout to PNGs in our folders
    patterns = "\n".join(f"/{folder}/*.png" for *_, folder in SOURCES) + "\n"
    git("sparse-checkout", "set", "--no-cone", "--stdin", input=patterns, text=True)
    # Fetch and check out the pinned commit
    git("fetch", "--depth", "1", "--filter=blob:none", "origin", SPRITES_COMMIT)
    git("checkout", "--quiet", SPRITES_COMMIT)


def load_metadata():
    """Names and primary types from the cache; query PokeAPI only for missing IDs."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [i for i in range(1, N_POKEMON + 1) if str(i) not in cache]
    for i in missing:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}", timeout=30)
        r.raise_for_status()
        d = r.json()
        types = sorted(d["types"], key=lambda t: t["slot"])   # primary = slot 1
        cache[str(i)] = {"name": d["name"], "primary_type": types[0]["type"]["name"]}
        time.sleep(0.1)                                       # rate limit
    if missing:
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def has_alpha(im):
    """True if the image has transparency."""
    return im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info


def crop_and_pad(rgba):
    """Crop to the Pokemon, then pad to a square (keeps aspect ratio)."""
    rgba = rgba.crop(rgba.getchannel("A").point(lambda a: 255 if a > 0 else 0).getbbox())
    w, h = rgba.size
    side = max(w, h) + 2 * round(CROP_MARGIN * max(w, h))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba, ((side - w) // 2, (side - h) // 2))
    return canvas


def process(path):
    """Return (color, silhouette, info), or None if the image has no alpha."""
    im = Image.open(path)
    if not has_alpha(im):
        return None
    w, h = im.size
    rgba = im.convert("RGBA")
    if rgba.getchannel("A").getextrema()[1] == 0:   # fully transparent
        return None
    rgba = crop_and_pad(rgba)
    # Nearest for pixel sprites (stay sharp); BOX for large art (no edge specks)
    resample = Image.NEAREST if max(w, h) <= PIXEL_ART_MAX_SIDE else Image.BOX
    rgba = rgba.resize((SIZE, SIZE), resample)

    arr = np.asarray(rgba)
    alpha = arr[..., 3]
    mask = alpha > 0          # Pokemon pixels
    if not mask.any():
        return None

    # Color: composite onto white
    a = alpha[..., None].astype(np.float32) / 255.0
    color = (arr[..., :3] * a + 255.0 * (1 - a)).round().astype(np.uint8)
    # Silhouette: Pokemon black, background white
    silhouette = np.where(mask, 0, 255).astype(np.uint8)

    info = {"orig_width": w, "orig_height": h,
            "fill_ratio": round(float(mask.mean()), 4)}  # share of frame covered
    return color, silhouette, info


def main():
    ensure_repo()
    meta = load_metadata()

    rows, dropped = [], []
    for style, gen, split, folder in SOURCES:
        for pid in range(1, N_POKEMON + 1):
            src = REPO / folder / f"{pid}.png"
            if not src.exists():
                continue
            result = process(src)
            if result is None:
                dropped.append(str(src.relative_to(REPO)))
                continue
            color, sil, info = result
            # Save to DATA/images/<color|silhouette>/<split>/<dex>/<style>_<dex>.png
            image_id = f"{style}_{pid:03d}"
            rel = Path(split) / f"{pid:03d}" / f"{image_id}.png"
            for out_dir, arr in ((OUT_COLOR, color), (OUT_SIL, sil)):
                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(out_dir / rel)
            rows.append({
                "image_id": image_id, "pokedex_id": pid,
                "name": meta[str(pid)]["name"],
                "primary_type": meta[str(pid)]["primary_type"],
                "source_style": style,
                "generation": gen if gen is not None else pd.NA,
                "split": split, **info,
                "color_path": str((OUT_COLOR / rel).relative_to(ROOT)),
                "silhouette_path": str((OUT_SIL / rel).relative_to(ROOT)),
            })

    df = pd.DataFrame(rows)
    df["generation"] = df["generation"].astype("Int64")   # NA for HOME/art
    df.to_csv(MANIFEST, index=False)
    print(f"Wrote {MANIFEST.relative_to(ROOT)} ({len(df)} rows)")
    print(f"Dropped for no alpha channel: {len(dropped)}")
    for p in dropped:
        print("  ", p)
    print_stats(df)


def print_stats(df):
    """Print dataset summary stats."""
    print("\n=== Dataset stats ===")
    print(f"Total images: {len(df)}")
    print("\nImages per split:")
    print(df["split"].value_counts().reindex(["train", "val", "test", "excluded"]).to_string())

    used = df[df["split"] != "excluded"]
    per = used.groupby("pokedex_id").size()
    print(f"\nImages per Pokemon (train+val+test): min={per.min()}  median={per.median():g}  "
          f"max={per.max()}")

    order = [s for s, *_ in SOURCES]
    print("\nCount by source_style:")
    print(df["source_style"].value_counts().reindex(order).to_string())

    print("\nPokemon missing per split:")
    all_ids = set(range(1, N_POKEMON + 1))
    for split in ["train", "val", "test"]:
        miss = sorted(all_ids - set(df.loc[df["split"] == split, "pokedex_id"]))
        print(f"  {split}: {miss if miss else 'none'}")

    print("\nMean fill_ratio by source_style:")
    print(df.groupby("source_style")["fill_ratio"].mean().reindex(order).round(3).to_string())


if __name__ == "__main__":
    main()
