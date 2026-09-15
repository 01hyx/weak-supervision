"""生成六期 Sentinel-2 高频响应对比图，用于观察时相与边界/纹理频域差异。"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm

from eval_wavelet_result_postprocess import haar_detail, normalize_robust
from make_six_period_sentinel_figures import make_figure
from test_manual_tiles import DEFAULT_DATA_ROOT, collect_valid_sample_ids, discover_regions


def read_period(region_dir, sample_name, period):
    with rasterio.open(region_dir / f"w{period}" / sample_name) as src:
        return src.read().astype(np.float32)


def high_frequency_score(data):
    # Per-period bands: B2, B3, B4, B8, B11, B12.
    red = data[2]
    nir = data[3]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    brightness = np.mean(data[:4], axis=0)
    ndwi_like = (data[1] - data[4]) / np.maximum(data[1] + data[4], 1e-6)

    scores = [
        0.45 * normalize_robust(haar_detail(ndvi)),
        0.35 * normalize_robust(haar_detail(brightness)),
        0.20 * normalize_robust(haar_detail(ndwi_like)),
    ]
    gradient_scores = []
    for source in (normalize_robust(ndvi), normalize_robust(brightness)):
        for sigma in (0.8, 1.6):
            smooth = cv2.GaussianBlur(source.astype(np.float32), (0, 0), sigma)
            gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
            gradient_scores.append(normalize_robust(cv2.magnitude(gx, gy)))
    score = normalize_robust(sum(scores) + 0.35 * np.max(np.stack(gradient_scores), axis=0))
    return cv2.GaussianBlur(score.astype(np.float32), (3, 3), 0)


def score_to_panel(score, color_mode="gray"):
    values = score[np.isfinite(score)]
    if values.size:
        lo, hi = np.percentile(values, [2, 99])
    else:
        lo, hi = 0.0, 1.0
    gray = np.clip((score - lo) / max(hi - lo, 1e-6), 0, 1)
    if color_mode == "gray":
        u8 = (gray * 255).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    # 蓝-白-红伪彩色：蓝色为弱高频，白色为中等响应，红色为强边界/纹理响应。
    blue = np.array([29, 78, 216], dtype=np.float32)
    white = np.array([245, 247, 250], dtype=np.float32)
    red = np.array([220, 38, 38], dtype=np.float32)
    panel = np.zeros((*gray.shape, 3), dtype=np.float32)
    low = gray <= 0.5
    t_low = (gray[low] / 0.5).reshape(-1, 1)
    t_high = ((gray[~low] - 0.5) / 0.5).reshape(-1, 1)
    panel[low] = blue * (1.0 - t_low) + white * t_low
    panel[~low] = white * (1.0 - t_high) + red * t_high
    return np.clip(panel, 0, 255).astype(np.uint8)


def make_high_frequency_figure(region_dir, sample_name, panel_size, show_labels, color_mode):
    panels = []
    mean_scores = []
    p90_scores = []
    for period in range(1, 7):
        score = high_frequency_score(read_period(region_dir, sample_name, period))
        panels.append(score_to_panel(score, color_mode=color_mode))
        mean_scores.append(float(np.mean(score[np.isfinite(score)])))
        p90_scores.append(float(np.percentile(score[np.isfinite(score)], 90)))
    return make_figure(panels, panel_size, show_labels), mean_scores, p90_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--output-dir", default="manual_tile_six_period_high_frequency")
    parser.add_argument("--panel-size", type=int, default=768)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--color-mode", choices=["gray", "redblue"], default="gray")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for region_dir in discover_regions(args.data_root, args.regions):
        samples = collect_valid_sample_ids(region_dir)
        if args.sample:
            samples = [sample for sample in samples if Path(sample).stem == Path(args.sample).stem or sample == args.sample]
        for sample_name in tqdm(samples, desc=region_dir.name):
            figure, mean_scores, p90_scores = make_high_frequency_figure(
                region_dir, sample_name, args.panel_size, args.show_labels, args.color_mode
            )
            stem = Path(sample_name).stem
            out_dir = output_root / region_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_six_period_high_frequency.png"
            figure.save(out_path, dpi=(args.dpi, args.dpi), compress_level=4)
            row = {"region": region_dir.name, "sample": sample_name, "figure": str(out_path.resolve())}
            for idx, value in enumerate(mean_scores, start=1):
                row[f"w{idx}_mean"] = value
            for idx, value in enumerate(p90_scores, start=1):
                row[f"w{idx}_p90"] = value
            rows.append(row)

    with (output_root / "six_period_high_frequency_manifest.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Created {len(rows)} high-frequency figures: {output_root.resolve()}")


if __name__ == "__main__":
    main()
