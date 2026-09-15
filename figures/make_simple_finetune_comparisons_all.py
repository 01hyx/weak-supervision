import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_sample


def percentile_stretch(channel, low=2, high=98):
    values = channel[np.isfinite(channel) & (channel > 0)]
    if values.size == 0:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(values, [low, high])
    if hi <= lo:
        hi = lo + 1e-6
    return (np.clip((channel - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def make_rgb(image, time_index=4):
    start = time_index * 6
    return np.stack(
        [
            percentile_stretch(image[start + 2]),
            percentile_stretch(image[start + 1]),
            percentile_stretch(image[start]),
        ],
        axis=-1,
    )


def read_prob(path):
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
    if prob.max(initial=0) > 1:
        prob /= 255.0
    return prob


def mask_image(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = 255
    out[~valid] = 90
    return out


def make_four_panel(rgb, before, after, gt, valid, panel_size):
    panels = [
        Image.fromarray(rgb).resize(
            (panel_size, panel_size), Image.Resampling.LANCZOS
        ),
        Image.fromarray(mask_image(gt, valid)).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        ),
        Image.fromarray(mask_image(before, valid)).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        ),
        Image.fromarray(mask_image(after, valid)).resize(
            (panel_size, panel_size), Image.Resampling.NEAREST
        ),
    ]
    gap = max(12, panel_size // 64)
    canvas = Image.new(
        "RGB",
        (panel_size * 4 + gap * 3, panel_size),
        "white",
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * (panel_size + gap), 0))
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Create simple before/after/GT mask comparisons."
    )
    parser.add_argument(
        "--sample-csv",
        default="manual_tiles_filtered_problem_removed/remaining_samples.csv",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--before-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument(
        "--after-prob-root", default="manual_tile_prediction_tifs_finetune_head"
    )
    parser.add_argument(
        "--output-dir", default="manual_tile_simple_finetune_comparisons_all"
    )
    parser.add_argument("--before-threshold", type=float, default=0.12)
    parser.add_argument("--after-threshold", type=float, default=0.65)
    parser.add_argument("--panel-size", type=int, default=768)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with Path(args.sample_csv).open("r", encoding="utf-8-sig", newline="") as file:
        samples = list(csv.DictReader(file))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for item in tqdm(samples, desc="simple comparisons"):
        region = item["region"]
        sample = item["sample"]
        stem = Path(sample).stem
        image, gt_label = load_sample(
            Path(args.data_root) / region, sample, ignore_nodata=True
        )
        valid = gt_label != IGNORE_LABEL
        gt = gt_label == 1
        before_prob = read_prob(
            Path(args.before_prob_root) / region / "prob" / f"{stem}_prob.tif"
        )
        after_prob = read_prob(
            Path(args.after_prob_root) / region / "prob" / f"{stem}_prob.tif"
        )
        before = (before_prob >= args.before_threshold) & valid
        after = (after_prob >= args.after_threshold) & valid

        figure = make_four_panel(
            make_rgb(image), before, after, gt, valid, args.panel_size
        )
        output_dir = output_root / region
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{stem}_before_after_gt.png"
        figure.save(output_path, dpi=(args.dpi, args.dpi), compress_level=4)
        manifest.append(
            {
                "region": region,
                "sample": sample,
                "figure": str(output_path.resolve()),
            }
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[INFO] Created {len(manifest)} figures: {output_root.resolve()}")


if __name__ == "__main__":
    main()
