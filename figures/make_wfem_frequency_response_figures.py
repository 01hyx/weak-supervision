import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from make_frequency_best_result_figures import make_rgb, resize
from test_manual_tiles import DEFAULT_DATA_ROOT, collect_valid_sample_ids, discover_regions, load_sample


def normalize_robust(array, low=2, high=98):
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(valid, [low, high])
    return np.clip((array - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def haar_components(array):
    """Haar 小波分解，并将 LL/LH/HL/HH 上采样回原尺寸用于论文可视化。"""
    h, w = array.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        array = cv2.copyMakeBorder(array, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    x00 = array[0::2, 0::2]
    x01 = array[0::2, 1::2]
    x10 = array[1::2, 0::2]
    x11 = array[1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.25
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5

    target_size = (w, h)
    return tuple(
        cv2.resize(component, target_size, interpolation=cv2.INTER_LINEAR)
        for component in (ll, lh, hl, hh)
    )


def period_source(data, time_index):
    # 每期 6 波段顺序：[B2, B3, B4, B8, B11, B12]。
    start = time_index * 6
    blue, green, red, nir, swir1, _ = data[start:start + 6]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    ndwi_like = (green - swir1) / np.maximum(green + swir1, 1e-6)
    brightness = np.mean([blue, green, red, nir], axis=0)

    # 中文注释：用植被指数、亮度和水分相关指数共同构造 WFEM 可解释输入，
    # 避免单一 NDVI 把道路、沟渠等边界信息弱化。
    return (
        0.50 * normalize_robust(ndvi)
        + 0.30 * normalize_robust(brightness)
        + 0.20 * normalize_robust(ndwi_like)
    ).astype(np.float32)


def wfem_response(data, periods):
    ll_list = []
    high_list = []
    directional_list = []
    for period in periods:
        source = period_source(data, period - 1)
        ll, lh, hl, hh = haar_components(source)
        high = np.sqrt(lh * lh + hl * hl + hh * hh)
        ll_list.append(normalize_robust(ll))
        high_list.append(normalize_robust(high))
        directional_list.append(
            np.stack(
                [
                    normalize_robust(np.abs(lh)),
                    normalize_robust(np.abs(hl)),
                    normalize_robust(np.abs(hh)),
                ],
                axis=-1,
            )
        )

    ll_response = cv2.GaussianBlur(np.mean(np.stack(ll_list), axis=0), (5, 5), 0)
    high_response = cv2.GaussianBlur(np.max(np.stack(high_list), axis=0), (3, 3), 0)
    directional = np.max(np.stack(directional_list), axis=0)
    return normalize_robust(ll_response), normalize_robust(high_response), directional


def apply_colormap(score, cmap):
    gray = (normalize_robust(score) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cmap), cv2.COLOR_BGR2RGB)


def high_rgb_panel(high_response, directional):
    # R/G/B 分别对应 LH/HL/HH，并用合成高频强度控制亮度。
    panel = normalize_robust(directional) * np.expand_dims(normalize_robust(high_response), axis=-1)
    panel = np.power(panel, 0.75)
    return np.clip(panel * 255, 0, 255).astype(np.uint8)


def boundary_overlay(rgb, high_response, quantile):
    out = rgb.astype(np.float32)
    valid = np.any(rgb > 0, axis=-1)
    threshold = np.percentile(high_response[valid], quantile) if np.any(valid) else np.percentile(high_response, quantile)
    edges = (high_response >= threshold) & valid
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) > 0
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), dtype=np.uint8)) > 0

    color = np.array([255, 212, 0], dtype=np.float32)
    out[edges] = 0.25 * out[edges] + 0.75 * color
    out[~valid] = 65
    return np.clip(out, 0, 255).astype(np.uint8), edges


def make_figure(rgb, ll_response, high_response, directional, panel_size, edge_quantile):
    overlay, edges = boundary_overlay(rgb, high_response, edge_quantile)
    panels = [
        resize(rgb, panel_size),
        resize(apply_colormap(ll_response, cv2.COLORMAP_VIRIDIS), panel_size),
        resize(high_rgb_panel(high_response, directional), panel_size),
        resize(overlay, panel_size),
    ]

    gap = max(16, panel_size // 48)
    outer = max(12, panel_size // 64)
    canvas = Image.new("RGB", (outer * 2 + panel_size * 4 + gap * 3, outer * 2 + panel_size), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (outer + index * (panel_size + gap), outer))
    return canvas, int(edges.sum()), float(np.mean(high_response)), float(np.percentile(high_response, 95))


def iter_samples(args):
    if args.split:
        split = json.loads(Path(args.split).read_text(encoding="utf-8"))
        for split_part in args.split_parts:
            for item in split[split_part]:
                yield split_part, item["region"], item["sample"]
        return
    for region_dir in discover_regions(args.data_root, args.regions):
        for sample_name in collect_valid_sample_ids(region_dir):
            yield "all", region_dir.name, sample_name


def parse_args():
    parser = argparse.ArgumentParser(description="Create WFEM low/high frequency response figures.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="paper_figures/fig5_2_wfem_frequency_response_all")
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--split-parts", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--periods", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--rgb-time", type=int, default=4, help="0-based period; default 4 means W5.")
    parser.add_argument("--edge-quantile", type=float, default=88.0)
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for split_part, region, sample_name in tqdm(list(iter_samples(args)), desc="WFEM response figures"):
        image, _ = load_sample(Path(args.data_root) / region, sample_name, ignore_nodata=True)
        rgb = make_rgb(image, args.rgb_time)
        ll_response, high_response, directional = wfem_response(image, args.periods)
        figure, edge_pixels, high_mean, high_p95 = make_figure(
            rgb, ll_response, high_response, directional, args.panel_size, args.edge_quantile
        )

        stem = Path(sample_name).stem
        output_dir = output_root / split_part / region
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{stem}_fig5_2_wfem_response.png"
        figure.save(output_path, dpi=(args.dpi, args.dpi), compress_level=4)
        rows.append(
            {
                "split": split_part,
                "region": region,
                "sample": sample_name,
                "figure": str(output_path.resolve()),
                "periods": ",".join(str(p) for p in args.periods),
                "edge_pixels": edge_pixels,
                "high_mean": high_mean,
                "high_p95": high_p95,
            }
        )

    rows.sort(key=lambda row: row["edge_pixels"], reverse=True)
    manifest = output_root / "fig5_2_wfem_response_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Created {len(rows)} figures: {output_root.resolve()}")
    print(f"[INFO] Manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
