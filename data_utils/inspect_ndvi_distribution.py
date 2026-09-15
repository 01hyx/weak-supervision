import argparse
import csv
from pathlib import Path

import numpy as np

from test_manual_tiles import DEFAULT_DATA_ROOT, BANDS_PER_STEP, TIME_STEPS, collect_valid_sample_ids, load_sample


def compute_ndvi_features(image_chw, red_band, nir_band):
    h, w = image_chw.shape[-2:]
    data_t = image_chw.reshape(TIME_STEPS, BANDS_PER_STEP, h, w)
    red = data_t[:, red_band].astype(np.float32)
    nir = data_t[:, nir_band].astype(np.float32)
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    return {
        "max": np.nanmax(ndvi, axis=0),
        "mean": np.nanmean(ndvi, axis=0),
        "amp": np.nanmax(ndvi, axis=0) - np.nanmin(ndvi, axis=0),
    }


def quantiles(values):
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    if values.size == 0:
        return {f"q{q}": np.nan for q in qs}
    return {f"q{q}": float(np.percentile(values, q)) for q in qs}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect NDVI feature distributions by GT class.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--regions", nargs="+", default=["滨城区3镇", "阳信县4镇"])
    parser.add_argument("--red-band", type=int, default=2)
    parser.add_argument("--nir-band", type=int, default=3)
    parser.add_argument("--samples-per-region", type=int, default=0)
    parser.add_argument("--output-csv", default="ndvi_distribution.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for region in args.regions:
        region_dir = Path(args.data_root) / region
        sample_ids = collect_valid_sample_ids(region_dir)
        if args.samples_per_region > 0:
            sample_ids = sample_ids[:args.samples_per_region]
        buckets = {
            ("background", "max"): [],
            ("background", "mean"): [],
            ("background", "amp"): [],
            ("target", "max"): [],
            ("target", "mean"): [],
            ("target", "amp"): [],
        }
        for sample_name in sample_ids:
            image, gt = load_sample(region_dir, sample_name, ignore_nodata=True)
            features = compute_ndvi_features(image, args.red_band, args.nir_band)
            valid_bg = gt == 0
            valid_target = gt == 1
            for name, arr in features.items():
                buckets[("background", name)].append(arr[valid_bg])
                buckets[("target", name)].append(arr[valid_target])

        for class_name in ("background", "target"):
            for feature in ("max", "mean", "amp"):
                values = np.concatenate(buckets[(class_name, feature)])
                row = {
                    "region": region,
                    "class": class_name,
                    "feature": feature,
                    "count": int(values.size),
                    **quantiles(values),
                }
                rows.append(row)
                print(row)

    out = Path(args.output_csv)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved: {out}")


if __name__ == "__main__":
    main()
