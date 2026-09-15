import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_multitemporal_image
from tif_binary_postprocess import read_single_band


def stretch_rgb(image):
    data = image.reshape(6, 6, image.shape[-2], image.shape[-1])
    rgb = np.stack([data[4, 2], data[4, 1], data[4, 0]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        vals = rgb[..., channel]
        valid = vals > 0
        if np.any(valid):
            lo, hi = np.percentile(vals[valid], [2, 98])
        else:
            lo, hi = 0.0, 1.0
        out[..., channel] = np.clip((vals - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def boundary_mask(mask, radius=2):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    return cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)


def overlay(base, mask, color, alpha=0.65):
    out = base.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 1)


def read_rows(csv_path, top_fn, top_boundary, top_score):
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["fn"] = int(row["fn"])
        row["boundary_error_rate"] = float(row["boundary_error_rate"])
        row["score_for_sort"] = float(row.get("reliable_problem_score", 0.0))
    selected = []
    seen = set()
    if top_score > 0:
        for row in sorted(rows, key=lambda item: item["score_for_sort"], reverse=True)[:top_score]:
            key = (row["region"], row["sample"])
            selected.append((row, "可信问题"))
            seen.add(key)
    for row in sorted(rows, key=lambda item: item["fn"], reverse=True)[:top_fn]:
        key = (row["region"], row["sample"])
        if key in seen:
            continue
        selected.append((row, "漏检严重"))
        seen.add(key)
    for row in sorted(rows, key=lambda item: item["boundary_error_rate"], reverse=True)[:top_boundary]:
        key = (row["region"], row["sample"])
        if key not in seen:
            selected.append((row, "边界问题"))
            seen.add(key)
    return selected


def make_one(data_root, pred_root, output_dir, row, tag, threshold):
    region = row["region"]
    sample = row["sample"]
    stem = Path(sample).stem
    region_dir = Path(data_root) / region
    image, valid = load_multitemporal_image(region_dir, sample)
    gt, _ = read_single_band(region_dir / "mask" / sample)
    prob, _ = read_single_band(Path(pred_root) / region / "prob" / f"{stem}_prob.tif")

    gt = (gt > 0)
    prob = prob.astype(np.float32)
    if prob.max() > 1.0:
        prob = prob / 255.0
    pred = prob >= threshold
    gt[~valid] = False
    pred[~valid] = False

    rgb = stretch_rgb(image)
    fn = gt & ~pred
    fp = ~gt & pred & valid
    tp = gt & pred
    boundary = (boundary_mask(gt, 2) | boundary_mask(pred, 2)) & (gt != pred) & valid

    pred_vis = np.zeros((*gt.shape, 3), dtype=np.float32)
    pred_vis[pred] = [0.0, 0.8, 1.0]
    gt_vis = np.zeros((*gt.shape, 3), dtype=np.float32)
    gt_vis[gt] = [1.0, 1.0, 1.0]
    err_vis = np.zeros((*gt.shape, 3), dtype=np.float32)
    err_vis[tp] = [0.0, 0.8, 0.25]
    err_vis[fp] = [0.1, 0.35, 1.0]
    err_vis[fn] = [1.0, 0.15, 0.12]

    fn_overlay = overlay(rgb, fn, [1.0, 0.0, 0.0], 0.72)
    boundary_overlay = overlay(rgb, boundary, [1.0, 0.9, 0.0], 0.85)

    panels = [
        ("RGB reference", rgb),
        ("GT mask white=target", gt_vis),
        ("Prediction cyan=target", pred_vis),
        ("Error green=TP blue=FP red=FN", err_vis),
        ("Missed target overlay red", fn_overlay),
        ("Boundary error overlay yellow", boundary_overlay),
    ]

    tile_w, tile_h = gt.shape[1], gt.shape[0]
    title_h = 24
    header_h = 36
    canvas = Image.new("RGB", (tile_w * 3, header_h + (tile_h + title_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    header = (
        f"{tag}: {region}/{stem}  FN={row['fn']}  "
        f"FN rate={float(row['fn_rate']):.3f}  boundary={float(row['boundary_error_rate']):.3f}"
    )
    draw.text((6, 8), header, fill=(0, 0, 0))
    for idx, (title, img) in enumerate(panels):
        col = idx % 3
        row_idx = idx // 3
        x = col * tile_w
        y = header_h + row_idx * (tile_h + title_h)
        draw.text((x + 6, y + 5), title, fill=(0, 0, 0))
        img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(img_u8), (x, y + title_h))

    out_dir = Path(output_dir) / region
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_error_focus.png"
    canvas.save(out_path)
    print(f"[INFO] Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Make focused error maps for hard tiles.")
    parser.add_argument("--ranking-csv", default="manual_tile_error_rankings/finetune_head_error_ranking.csv")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pred-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--output-dir", default="manual_tile_error_focus_maps")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--top-fn", type=int, default=5)
    parser.add_argument("--top-boundary", type=int, default=5)
    parser.add_argument("--top-score", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    for row, tag in read_rows(args.ranking_csv, args.top_fn, args.top_boundary, args.top_score):
        make_one(args.data_root, args.pred_root, args.output_dir, row, tag, args.threshold)


if __name__ == "__main__":
    main()
