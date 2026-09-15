"""从候选瓦片中筛选少量清晰的 Sentinel-2 RGB 影像并导出 JPG。"""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from make_frequency_best_result_figures import make_rgb
from test_manual_tiles import collect_valid_sample_ids, load_sample


TILE_RE = re.compile(r"_r(\d+)_c(\d+)_")


def quality_score(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(gray.std())
    clipped = float(np.mean((gray < 5) | (gray > 250)))
    return sharpness + 2.0 * contrast - 200.0 * clipped


def coordinates(sample):
    match = TILE_RE.search(sample)
    return (int(match.group(1)), int(match.group(2))) if match else None


def select_diverse(candidates, count):
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        row_col = candidate["coordinates"]
        if row_col is not None and any(
            other["coordinates"] is not None
            and abs(row_col[0] - other["coordinates"][0]) <= 2
            and abs(row_col[1] - other["coordinates"][1]) <= 2
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output-dir", default="representative_rgb_jpg")
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--size", type=int, default=768)
    args = parser.parse_args()

    region_dir = Path(args.data_root) / args.region
    split_map = {}
    if args.split_file:
        payload = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        for part in ("train", "val", "test"):
            for item in payload.get(part, []):
                if item["region"] == args.region:
                    split_map[item["sample"]] = part

    samples = collect_valid_sample_ids(region_dir)
    if split_map:
        samples = [sample for sample in samples if sample in split_map]

    candidates = []
    for sample in samples:
        image, _ = load_sample(region_dir, sample, ignore_nodata=True)
        rgb = make_rgb(image)
        candidates.append(
            {
                "sample": sample,
                "split": split_map.get(sample, "all"),
                "rgb": rgb,
                "score": quality_score(rgb),
                "coordinates": coordinates(sample),
            }
        )

    output = Path(args.output_dir) / args.region
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, candidate in enumerate(select_diverse(candidates, args.count), start=1):
        stem = Path(candidate["sample"]).stem
        path = output / f"{rank:02d}_{stem}_rgb.jpg"
        image = Image.fromarray(candidate["rgb"]).resize(
            (args.size, args.size), Image.Resampling.LANCZOS
        )
        image.save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=True,
            dpi=(300, 300),
        )
        rows.append(
            {
                "rank": rank,
                "region": args.region,
                "sample": candidate["sample"],
                "split": candidate["split"],
                "quality_score": candidate["score"],
                "jpg": str(path.resolve()),
            }
        )

    manifest = Path(args.output_dir) / f"{args.region}_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Created {len(rows)} RGB JPG files: {output.resolve()}")


if __name__ == "__main__":
    main()
