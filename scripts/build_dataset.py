"""Build the Gen 1 color/silhouette dataset from the PokeAPI sprites repo.

Usage: python scripts/build_dataset.py

Steps:
  1. Sparse-clone https://github.com/PokeAPI/sprites into data/raw/ (front default
     sprites only) if it is not already there.
  2. Fetch name + primary type for Pokedex IDs 1-151 (cached in data/pokeapi_cache.json).
  3. Crop each image to its alpha bounding box (plus a 5% margin), pad to square,
     resize to 224x224, save a color version on white and a black-on-white silhouette.
  4. Write data/manifest.csv and print summary stats.
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPO = DATA / "raw" / "pokeapi-sprites"
CACHE = DATA / "pokeapi_cache.json"
MANIFEST = DATA / "manifest.csv"
OUT_COLOR = DATA / "images" / "color"
OUT_SIL = DATA / "images" / "silhouette"

N_POKEMON = 151
SIZE = 224
# Sources whose longest side is at most this are pixel sprites -> nearest-neighbor.
PIXEL_ART_MAX_SIDE = 128
# Margin kept around the alpha bounding box, as a fraction of its longer side.
CROP_MARGIN = 0.05

V = "sprites/pokemon/versions"
# (source_style, generation, split, folder). One folder per game, front default only.
# The default Gen I/II folders (red-blue, yellow, gold, silver, crystal) are opaque
# on a white background with no alpha, so for those games we use the transparent/
# copy instead. That is still one image per game per Pokemon, not a duplicate.
# Official artwork is kept in the manifest but excluded from all splits.
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


def ensure_repo():
    if REPO.exists():
        return
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "https://github.com/PokeAPI/sprites.git", str(REPO)],
        check=True,
    )
    patterns = "\n".join(f"/{folder}/*.png" for *_, folder in SOURCES) + "\n"
    subprocess.run(
        ["git", "-C", str(REPO), "sparse-checkout", "set", "--no-cone", "--stdin"],
        input=patterns, text=True, check=True,
    )


def load_metadata():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [i for i in range(1, N_POKEMON + 1) if str(i) not in cache]
    for i in missing:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}", timeout=30)
        r.raise_for_status()
        d = r.json()
        types = sorted(d["types"], key=lambda t: t["slot"])
        cache[str(i)] = {"name": d["name"], "primary_type": types[0]["type"]["name"]}
        time.sleep(0.1)
    if missing:
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def has_alpha(im):
    return im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info


def crop_and_pad(rgba):
    """Crop to the alpha>0 bounding box, then center on a transparent square canvas
    with a CROP_MARGIN margin around the longer side."""
    rgba = rgba.crop(rgba.getchannel("A").point(lambda a: 255 if a > 0 else 0).getbbox())
    w, h = rgba.size
    side = max(w, h) + 2 * round(CROP_MARGIN * max(w, h))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba, ((side - w) // 2, (side - h) // 2))
    return canvas


def process(path):
    """Return (color RGB array, silhouette L array, info dict) or None if no alpha."""
    im = Image.open(path)
    if not has_alpha(im):
        return None
    w, h = im.size
    rgba = im.convert("RGBA")
    if rgba.getchannel("A").getextrema()[1] == 0:
        return None
    rgba = crop_and_pad(rgba)
    # Area averaging (BOX) for large art: Lanczos ringing leaves faint alpha>0
    # specks around the outline that would show up as noise in the silhouette.
    resample = Image.NEAREST if max(w, h) <= PIXEL_ART_MAX_SIDE else Image.BOX
    rgba = rgba.resize((SIZE, SIZE), resample)

    arr = np.asarray(rgba)
    alpha = arr[..., 3]
    mask = alpha > 0
    if not mask.any():
        return None

    a = alpha[..., None].astype(np.float32) / 255.0
    color = (arr[..., :3] * a + 255.0 * (1 - a)).round().astype(np.uint8)
    silhouette = np.where(mask, 0, 255).astype(np.uint8)

    info = {"orig_width": w, "orig_height": h,
            "fill_ratio": round(float(mask.mean()), 4)}
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
    df["generation"] = df["generation"].astype("Int64")
    df.to_csv(MANIFEST, index=False)
    print(f"Wrote {MANIFEST.relative_to(ROOT)} ({len(df)} rows)")
    print(f"Dropped for no alpha channel: {len(dropped)}")
    for p in dropped:
        print("  ", p)
    print_stats(df)


def print_stats(df):
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
