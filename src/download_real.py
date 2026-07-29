"""
Downloads real images from LAION-style metadata parquet files (URL, WIDTH, HEIGHT,
punsafe, pwatermark, and an aesthetic-score column), filtering and cropping them to
match the paper's Sec. 7.1 recipe: keep rows whose smaller side is >=512px and whose
total pixel count is <=768^2 with an aesthetic score >=6.5, then center-crop to exactly
512x512 (no resizing, to avoid interpolation blur skewing the LPIPS reconstruction
error). Saved as PNG to match the lossless format already used for data/generated/,
avoiding a JPEG-vs-PNG storage bias between the real and generated sets.

Each row is identified by its `hash` column value (not positional index), since this
script can pull from multiple source parquet files (e.g. the original
real_metadata.parquet plus extra LAION-Aesthetics shards) that don't share an index
space but do share globally-unique row hashes.

Resumable: successes are inferred from files already present in data/real/, and known-dead
URLs are skipped via data/real/_download_manifest.csv, so reruns just top up the shortfall.
"""

import argparse
import csv
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from config import DATA_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

SCORE_COLUMNS = ("AESTHETIC_SCORE", "aesthetic")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and crop real images from a LAION-style metadata parquet."
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to a parquet file with URL/WIDTH/HEIGHT/hash/punsafe/pwatermark columns "
        "(default: data/real/real_metadata.parquet).",
    )
    parser.add_argument("--target", type=int, default=1000, help="Number of successful downloads to reach.")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent download threads.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for candidate order.")
    parser.add_argument("--min-side", type=int, default=512, help="Minimum of width/height required.")
    parser.add_argument("--max-pixels", type=int, default=768 * 768, help="Maximum width*height allowed.")
    parser.add_argument(
        "--min-aesthetic", type=float, default=6.5,
        help="Minimum aesthetic score required (paper's LAION-Aesthetics cutoff).",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--max-punsafe", type=float, default=0.5,
        help="Drop rows with punsafe above this (set to 1.0 to disable).",
    )
    parser.add_argument(
        "--max-pwatermark", type=float, default=0.5,
        help="Drop rows with pwatermark above this (set to 1.0 to disable).",
    )
    return parser.parse_args()


def _score_column(df):
    for col in SCORE_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(f"No aesthetic score column found (expected one of {SCORE_COLUMNS}).")


def _center_crop(img, size):
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _download_one(h, url, real_dir, min_side, max_pixels, timeout):
    """Fetches, validates, crops, and saves one candidate. Returns (hash, url, status)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return h, url, "failed"

        img = Image.open(io.BytesIO(resp.content))
        img.load()

        w, hgt = img.size
        if min(w, hgt) < min_side or w * hgt > max_pixels:
            return h, url, "failed"

        img = img.convert("RGB")
        cropped = _center_crop(img, min_side)
        cropped.save(real_dir / f"real_{h}.png", "PNG")
        return h, url, "success"
    except Exception:
        return h, url, "failed"


def _load_manifest(manifest_path):
    success, failed = set(), set()
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = int(row["hash"])
                (success if row["status"] == "success" else failed).add(h)
    return success, failed


def _existing_downloads(real_dir):
    hashes = set()
    for p in real_dir.glob("real_*.png"):
        try:
            hashes.add(int(p.stem.split("_", 1)[1]))
        except ValueError:
            continue
    return hashes


def main():
    args = parse_args()

    real_dir = Path(DATA_DIR) / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source) if args.source else real_dir / "real_metadata.parquet"
    manifest_path = real_dir / "_download_manifest.csv"

    df = pd.read_parquet(source_path)
    print(f"Loaded {len(df):,} rows from {source_path}")
    score_col = _score_column(df)

    mask = (
        (df["WIDTH"] >= args.min_side)
        & (df["HEIGHT"] >= args.min_side)
        & (df["WIDTH"] * df["HEIGHT"] <= args.max_pixels)
        & (df[score_col] >= args.min_aesthetic)
    )
    if args.max_punsafe < 1.0:
        mask &= df["punsafe"].fillna(0) <= args.max_punsafe
    if args.max_pwatermark < 1.0:
        mask &= df["pwatermark"].fillna(0) <= args.max_pwatermark
    candidates = df[mask]
    print(
        f"{len(candidates):,} rows pass filters (min_side>={args.min_side}, "
        f"max_pixels<={args.max_pixels}, {score_col}>={args.min_aesthetic}, "
        f"punsafe<={args.max_punsafe}, pwatermark<={args.max_pwatermark})"
    )
    candidates = candidates.sample(frac=1.0, random_state=args.seed)
    candidates = candidates.drop_duplicates(subset="hash").set_index("hash", drop=False)

    manifest_success, manifest_failed = _load_manifest(manifest_path)
    already_success = manifest_success | _existing_downloads(real_dir)
    skip = already_success | manifest_failed

    pool = [h for h in candidates.index if h not in skip]
    n_have = len(already_success)
    print(
        f"Already have {n_have:,} real images; {len(manifest_failed):,} known-dead rows "
        f"skipped; {len(pool):,} fresh candidates available."
    )

    if n_have >= args.target:
        print(f"Target of {args.target} already met. Nothing to do.")
        return

    manifest_is_new = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        if manifest_is_new:
            writer.writerow(["hash", "url", "status"])

        success_count = n_have
        attempted = 0
        pool_iter = iter(pool)

        with ThreadPoolExecutor(max_workers=args.workers) as executor, tqdm(
            total=args.target, initial=n_have, desc="Downloading real images"
        ) as pbar:

            def submit_next():
                h = next(pool_iter, None)
                if h is None:
                    return None
                url = candidates.loc[h, "URL"]
                fut = executor.submit(
                    _download_one, h, url, real_dir, args.min_side, args.max_pixels, args.timeout
                )
                futures[fut] = h
                return fut

            futures = {}
            for _ in range(args.workers):
                if submit_next() is None:
                    break

            # Keep draining outstanding futures until none remain, but only submit
            # replacements while under target. This way every submitted download
            # (and file it may have already saved to disk) gets logged to the
            # manifest, instead of being abandoned in-flight once the target is hit.
            while futures:
                done_fut = next(as_completed(futures))
                futures.pop(done_fut)
                h, url, status = done_fut.result()
                writer.writerow([h, url, status])
                manifest_file.flush()
                attempted += 1
                if status == "success":
                    success_count += 1
                    pbar.update(1)
                if success_count < args.target:
                    submit_next()

    print(f"\nDone. {success_count}/{args.target} real images now in {real_dir} ({attempted} attempts this run).")
    if success_count < args.target:
        print(
            f"WARNING: fell short of target by {args.target - success_count}; filtered "
            f"candidate pool exhausted. Loosen --min-side/--max-pixels/--max-punsafe/"
            f"--max-pwatermark, or pass --source with more candidates."
        )


if __name__ == "__main__":
    main()
