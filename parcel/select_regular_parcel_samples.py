"""Select tiles dominated by parcels resolvable at Sentinel-2 spatial resolution."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio


def parcel_stats(instances):
    height, width = instances.shape
    total = regular = 0
    areas, widths, elongations = [], [], []
    for value in np.unique(instances):
        if value == 0:
            continue
        mask = instances == value
        ys, xs = np.nonzero(mask)
        area = len(xs)
        if area < 20:
            continue
        if np.any(xs == 0) or np.any(ys == 0) or np.any(xs == width - 1) or np.any(ys == height - 1):
            continue
        total += 1
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        effective_width = float(distance.max() * 2.0)
        points = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(points)
        side_a, side_b = rect[1]
        elongation = max(side_a, side_b) / max(min(side_a, side_b), 1.0)
        is_regular = area >= 50 and effective_width >= 4.0 and elongation <= 6.0
        regular += int(is_regular)
        areas.append(area)
        widths.append(effective_width)
        elongations.append(elongation)
    fraction = regular / max(total, 1)
    # 奖励规则地块占比与数量，同时抑制每瓦片数百个细碎实例的拥挤样本。
    score = fraction + 0.20 * min(regular / 25.0, 1.0) - 0.15 * min(total / 250.0, 1.0)
    return {
        "eligible_parcels": total,
        "regular_parcels": regular,
        "regular_fraction": fraction,
        "median_area_pixels": float(np.median(areas)) if areas else 0.0,
        "median_width_pixels": float(np.median(widths)) if widths else 0.0,
        "median_elongation": float(np.median(elongations)) if elongations else 0.0,
        "selection_score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default=r"D:\s2_output\qingdao_selected150_2025_sentinel"
    )
    parser.add_argument("--region", default="青岛夏玉米0925")
    parser.add_argument("--split-file", default="configs/qingdao_2025_spatial_split.json")
    parser.add_argument("--output", default="configs/qingdao_regular_parcel_split.json")
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--test-count", type=int, default=10)
    args = parser.parse_args()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    rows = []
    selected = {}
    requested = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    for part in ("train", "val", "test"):
        part_rows = []
        for item in split[part]:
            if item["region"] != args.region:
                continue
            path = Path(args.data_root) / args.region / "instance_mask" / item["sample"]
            with rasterio.open(path) as src:
                instances = src.read(1)
            row = {"split": part, "sample": item["sample"], **parcel_stats(instances)}
            rows.append(row)
            part_rows.append(row)
        candidates = [row for row in part_rows if row["regular_parcels"] >= 8]
        candidates.sort(key=lambda row: row["selection_score"], reverse=True)
        selected[part] = [
            {"region": args.region, "sample": row["sample"]}
            for row in candidates[: requested[part]]
        ]

    payload = {
        "criteria": {
            "minimum_area_pixels": 50,
            "minimum_area_hectares": 0.5,
            "minimum_effective_width_pixels": 4.0,
            "maximum_elongation": 6.0,
            "minimum_regular_parcels_per_tile": 8,
        },
        **selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["split"], -row["selection_score"])))
    print(json.dumps({part: len(selected[part]) for part in selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
