"""Visualize RGB, ground-truth instances, semantic components and predicted instances."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from make_frequency_best_result_figures import make_rgb, resize
from test_manual_tiles import load_sample


def instance_colors(labels):
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value in np.unique(labels):
        if value == 0:
            continue
        rng = np.random.default_rng(int(value) * 104729 % (2**32 - 1))
        color = rng.integers(55, 240, size=3, dtype=np.uint8)
        output[labels == value] = color
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\s2_output\qingdao_selected150_2025_sentinel")
    parser.add_argument("--split-file", default="configs/qingdao_2025_spatial_split.json")
    parser.add_argument("--region", default="青岛夏玉米0925")
    parser.add_argument("--semantic-root", default="qingdao_parcel_instance_outputs")
    parser.add_argument("--instance-root", default="qingdao_parcel_instance_test_f1_best")
    parser.add_argument("--output-dir", default="qingdao_parcel_instance_test_figures")
    parser.add_argument("--panel-size", type=int, default=768)
    args = parser.parse_args()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [item for item in split["test"] if item["region"] == args.region]
    rows = []
    for item in tqdm(items, desc="Instance figures"):
        sample = item["sample"]
        stem = Path(sample).stem
        image, _ = load_sample(Path(args.data_root) / args.region, sample, ignore_nodata=True)
        with rasterio.open(Path(args.data_root) / args.region / "instance_mask" / sample) as src:
            gt = src.read(1)
        with rasterio.open(
            Path(args.semantic_root) / args.region / "semantic" / f"{stem}_semantic.tif"
        ) as src:
            semantic = src.read(1) >= 0.60
        _, semantic_components = cv2.connectedComponents(semantic.astype(np.uint8), 8)
        with rasterio.open(
            Path(args.instance_root) / args.region / "instance" / f"{stem}_instance.tif"
        ) as src:
            prediction = src.read(1)
        panels = [
            resize(make_rgb(image), args.panel_size),
            resize(instance_colors(gt), args.panel_size, True),
            resize(instance_colors(semantic_components), args.panel_size, True),
            resize(instance_colors(prediction), args.panel_size, True),
        ]
        gap = 14
        canvas = Image.new("RGB", (args.panel_size * 4 + gap * 3, args.panel_size), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (index * (args.panel_size + gap), 0))
        path = Path(args.output_dir) / args.region / f"{stem}_instance_compare.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, dpi=(300, 300), compress_level=4)
        rows.append({"sample": sample, "figure": str(path.resolve())})
    with (Path(args.output_dir) / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Created {len(rows)} figures: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
