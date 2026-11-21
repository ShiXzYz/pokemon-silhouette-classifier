"""
=============================================================================
00_build_dataset.py - Build the Gen I color + silhouette image dataset
=============================================================================
Project : DS 4002 Project 3, "Who's That Pokemon?" (pokemon-silhouette-classifier)
Authors : Brian Ryu, Keegan Jewell, Nathan Nguyen

PURPOSE
    Download Generation I (Pokedex #1-151) front-default sprites from 15 art
    styles in the PokeAPI sprites repository, crop/pad/resize every image to
    224x224, and save two versions of each: a full-color image composited on
    white and a black-on-white silhouette. Each art style is assigned to one
    split (train = 12 Gen I-IV game styles, val = Black/White, test = HOME,
    excluded = official artwork). Writes the manifest that every later script
    reads.

INPUTS
    - https://github.com/PokeAPI/sprites, pinned to commit SPRITES_COMMIT
      (sparse, blobless clone into DATA/raw/pokeapi-sprites/, ~800 MB)
    - https://pokeapi.co/api/v2/pokemon/{id} for names + primary types, only for
      IDs missing from DATA/pokeapi_cache.json (the committed cache has all
      151, so normally no API calls are made)

OUTPUTS
    - DATA/raw/pokeapi-sprites/              raw sprite repo   (gitignored)
    - DATA/images/color/<split>/<id>/*.png      color images   (gitignored)
    - DATA/images/silhouette/<split>/<id>/*.png silhouettes    (gitignored)
    - DATA/manifest.csv     one row per image: labels, split, paths (committed)
    - DATA/pokeapi_cache.json  name/type lookup cache            (committed)

HOW TO RUN (from the repository root)
    python SCRIPTS/00_build_dataset.py

EXPECTED RUNTIME
    ~30 s on our machine for a first run including the sparse clone (can take
    a few minutes on a slow connection); ~10 s once DATA/raw/ exists.
=============================================================================
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Paths. Everything is resolved relative to the repository root (the parent of
# SCRIPTS/), so the script works no matter which directory it is launched from.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "DATA"
REPO = DATA / "raw" / "pokeapi-sprites"
CACHE = DATA / "pokeapi_cache.json"
MANIFEST = DATA / "manifest.csv"
OUT_COLOR = DATA / "images" / "color"
OUT_SIL = DATA / "images" / "silhouette"

# The sprites repo is updated over time, so we pin the exact commit our dataset
# was built from (committed 2026-11-03). Any later rebuild then produces a
# byte-identical DATA/manifest.csv.
SPRITES_URL = "https://github.com/PokeAPI/sprites.git"
SPRITES_COMMIT = "1b3b7c0a538dd2226ae08d0d2f8955125075bbd4"

N_POKEMON = 151          # Generation I only (Bulbasaur #1 ... Mew #151)
SIZE = 224               # ResNet-18's native ImageNet input resolution
# Sources whose longest side is at most this many pixels are low-res pixel art
# sprites; they are upscaled with nearest-neighbor so pixels stay crisp instead
# of being blurred.
PIXEL_ART_MAX_SIDE = 128
# Margin kept around the alpha bounding box, as a fraction of its longer side,
# so the Pokemon never touches the image border.
CROP_MARGIN = 0.05

V = "sprites/pokemon/versions"
# (source_style, generation, split, folder in the sprites repo).
# One folder per game, front default sprite only.
# The default Gen I/II folders (red-blue, yellow, gold, silver, crystal) are
# opaque on a white background with no alpha channel, so for those games we use
# the transparent/ copy instead. That is still one image per game per Pokemon,
# not a duplicate. Official artwork is kept in the manifest for reference but is
# excluded from all splits (it is too close in style to HOME, the test set).
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
    """Run a git command inside the sprites repo and fail loudly on error."""
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, **kw)


def ensure_repo():
    """Sparse-clone the PokeAPI sprites repo at the pinned commit.

    The full repo is several GB, so we (1) clone without blobs or a checkout,
    (2) restrict the checkout to the 15 folders we use, and (3) fetch and check
    out exactly SPRITES_COMMIT. If the repo already exists we only verify the
    commit, so re-running the script is cheap.
    """
    if REPO.exists():
        head = git("rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
        if head != SPRITES_COMMIT:
            print(f"WARNING: {REPO.relative_to(ROOT)} is at {head[:10]}, not the pinned "
                  f"commit {SPRITES_COMMIT[:10]}. Delete DATA/raw/ to rebuild exactly.")
        return
    REPO.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", "--depth", "1", "--filter=blob:none",
         "--sparse", SPRITES_URL, str(REPO)],
        check=True,
    )
    # Only PNGs directly inside our 15 folders (non-cone patterns).
    patterns = "\n".join(f"/{folder}/*.png" for *_, folder in SOURCES) + "\n"
    git("sparse-checkout", "set", "--no-cone", "--stdin", input=patterns, text=True)
    # GitHub allows fetching a specific commit by SHA; depth 1 keeps it small.
    git("fetch", "--depth", "1", "--filter=blob:none", "origin", SPRITES_COMMIT)
    git("checkout", "--quiet", SPRITES_COMMIT)


def load_metadata():
    """Return {pokedex_id: {name, primary_type}} for IDs 1-151.

    Uses DATA/pokeapi_cache.json and only calls the PokeAPI REST API for IDs that
    are missing from it (politely rate-limited to ~10 requests/s).
    """
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    missing = [i for i in range(1, N_POKEMON + 1) if str(i) not in cache]
    for i in missing:
        r = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}", timeout=30)
        r.raise_for_status()
        d = r.json()
        # A Pokemon's "primary" type is the one in slot 1 (e.g. Bulbasaur = grass).
        types = sorted(d["types"], key=lambda t: t["slot"])
        cache[str(i)] = {"name": d["name"], "primary_type": types[0]["type"]["name"]}
        time.sleep(0.1)
    if missing:
        CACHE.write_text(json.dumps(cache, indent=1))
    return cache


def has_alpha(im):
    """True if the image carries transparency we can use to separate the Pokemon
    from its background."""
    return im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info


def crop_and_pad(rgba):
    """Crop to the alpha>0 bounding box, then center on a transparent square
    canvas with a CROP_MARGIN margin around the longer side.

    Cropping removes the very different amounts of empty space each game leaves
    around its sprites, so object scale is comparable across art styles.
    Padding to a square (rather than stretching) preserves the aspect ratio,
    which matters for silhouettes, where outline shape is the only signal.
    """
    rgba = rgba.crop(rgba.getchannel("A").point(lambda a: 255 if a > 0 else 0).getbbox())
    w, h = rgba.size
    side = max(w, h) + 2 * round(CROP_MARGIN * max(w, h))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba, ((side - w) // 2, (side - h) // 2))
    return canvas


def process(path):
    """Return (color RGB array, silhouette L array, info dict), or None if the
    image has no usable alpha channel."""
    im = Image.open(path)
    if not has_alpha(im):
        return None
    w, h = im.size
    rgba = im.convert("RGBA")
    if rgba.getchannel("A").getextrema()[1] == 0:   # fully transparent image
        return None
    rgba = crop_and_pad(rgba)
    # Nearest-neighbor keeps small pixel sprites sharp. Area averaging (BOX) is
    # used for large art: Lanczos ringing leaves faint alpha>0 specks around the
    # outline that would show up as noise in the silhouette.
    resample = Image.NEAREST if max(w, h) <= PIXEL_ART_MAX_SIDE else Image.BOX
    rgba = rgba.resize((SIZE, SIZE), resample)

    arr = np.asarray(rgba)
    alpha = arr[..., 3]
    mask = alpha > 0          # any non-transparent pixel belongs to the Pokemon
    if not mask.any():
        return None

    # Color version: alpha-composite onto a white background so every style
    # shares the same background (no style-specific background leakage).
    a = alpha[..., None].astype(np.float32) / 255.0
    color = (arr[..., :3] * a + 255.0 * (1 - a)).round().astype(np.uint8)
    # Silhouette version: Pokemon pixels black (0), background white (255).
    silhouette = np.where(mask, 0, 255).astype(np.uint8)

    info = {"orig_width": w, "orig_height": h,
            "fill_ratio": round(float(mask.mean()), 4)}  # share of pixels = Pokemon
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
            # File layout: DATA/images/<color|silhouette>/<split>/<dex #>/<style>_<dex #>.png
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
                # Paths are stored relative to the repo root (e.g. DATA/images/...).
                "color_path": str((OUT_COLOR / rel).relative_to(ROOT)),
                "silhouette_path": str((OUT_SIL / rel).relative_to(ROOT)),
            })

    df = pd.DataFrame(rows)
    df["generation"] = df["generation"].astype("Int64")   # nullable int (NA for HOME/art)
    df.to_csv(MANIFEST, index=False)
    print(f"Wrote {MANIFEST.relative_to(ROOT)} ({len(df)} rows)")
    print(f"Dropped for no alpha channel: {len(dropped)}")
    for p in dropped:
        print("  ", p)
    print_stats(df)


def print_stats(df):
    """Print the summary statistics reported in DATA/README.md."""
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
