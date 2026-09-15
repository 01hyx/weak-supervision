import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from make_fig4_1_transfer_finetune_best import make_rgb, read_prob, resize_panel
from regularize_parcel_boundaries import regularize_mask
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, load_sample


def binary_panel(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = 255
    out[~valid] = 90
    return out


def contour(mask, width=2):
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, 255, width)
    return canvas > 0


def regularized_visual_mask(pred, regularized, valid):
    """论文展示用：保留预测主体，只用规则化边界增强视觉直线感。"""
    out = binary_panel(pred, valid)
    edge = contour(regularized & valid, width=2)
    # 用浅青色边界压在黑白 mask 上，既能看到结果主体，也能看到规则化线条。
    out[edge] = (0, 180, 200)
    return out


def regularized_overlay(rgb, gt, pred, regularized, valid):
    out = rgb.astype(np.float32)
    fill = (pred & valid)
    fill_color = np.asarray([0, 114, 178], dtype=np.float32)
    out[fill] = 0.68 * out[fill] + 0.32 * fill_color

    pred_edge = contour(pred & valid, width=1)
    reg_edge = contour(regularized & valid, width=2)
    gt_edge = contour(gt & valid, width=2)
    # 色盲友好：预测原边界为蓝，规则化边界为青，人工样本边界为黄。
    out[pred_edge] = np.asarray([0, 114, 178], dtype=np.float32)
    out[reg_edge] = np.asarray([0, 190, 210], dtype=np.float32)
    out[gt_edge] = np.asarray([240, 228, 66], dtype=np.float32)
    out[~valid] = 70
    return np.clip(out, 0, 255).astype(np.uint8)


def load_font(size, bold=False):
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_label(draw, x, y, label, font):
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rounded_rectangle(
        (x, y, x + bbox[2] - bbox[0] + 24, y + bbox[3] - bbox[1] + 16),
        radius=4,
        fill=(255, 255, 255),
        outline=(25, 25, 25),
        width=2,
    )
    draw.text((x + 12, y + 7), label, fill=(0, 0, 0), font=font)


def make_figure(rgb, gt, base, pred, regularized, valid, panel_size, mode):
    if mode == "mask":
        d_panel = regularized_visual_mask(pred, regularized, valid)
    elif mode == "overlay":
        d_panel = regularized_overlay(rgb, gt, pred, regularized, valid)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    panels = [
        resize_panel(rgb, panel_size),
        resize_panel(binary_panel(gt, valid), panel_size, nearest=True),
        resize_panel(binary_panel(base, valid), panel_size, nearest=True),
        resize_panel(d_panel, panel_size, nearest=(mode == "mask")),
    ]
    gap = max(18, panel_size // 42)
    outer = max(14, panel_size // 64)
    width = outer * 2 + panel_size * 4 + gap * 3
    height = outer * 2 + panel_size
    canvas = Image.new("RGB", (width, height), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (outer + i * (panel_size + gap), outer))

    draw = ImageDraw.Draw(canvas)
    font = load_font(max(28, panel_size // 24), bold=True)
    for i, label in enumerate(["(A)", "(B)", "(C)", "(D)"]):
        draw_label(draw, outer + i * (panel_size + gap) + 10, outer + 10, label, font)
    return canvas


def selected_items(split, manifest, top_n):
    if not manifest:
        return split["train"] + split["val"]
    rows = []
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            base_miou = float(row["base_miou"])
            fine_miou = float(row["finetune_miou"])
            fine_f1 = float(row["finetune_f1"])
            score = fine_miou + 0.55 * (fine_miou - base_miou) + 0.15 * fine_f1
            rows.append((score, row["region"], row["sample"]))
    rows.sort(reverse=True)
    return [{"region": region, "sample": sample} for _, region, sample in rows[:top_n]]


def main():
    parser = argparse.ArgumentParser(description="Create visually regularized Figure 4-1 candidates.")
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument("--finetune-prob-root", default="frequency_boundary_prediction_tifs_all")
    parser.add_argument("--output-dir", default="paper_figures/fig4_1_regularized_visual_candidates")
    parser.add_argument("--candidate-manifest", default="paper_figures/fig4_1_best_transfer_finetune_all/fig4_1_manifest.csv")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--mode", choices=["mask", "overlay"], default="mask")
    parser.add_argument("--base-threshold", type=float, default=0.12)
    parser.add_argument("--finetune-threshold", type=float, default=0.60)
    parser.add_argument("--rgb-time", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-hole-area", type=int, default=20)
    parser.add_argument("--close-size", type=int, default=1)
    parser.add_argument("--epsilon-ratio", type=float, default=0.0015)
    parser.add_argument("--epsilon-pixels", type=float, default=0.75)
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    items = selected_items(split, args.candidate_manifest, args.top_n)
    out_root = Path(args.output_dir) / args.mode
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for item in tqdm(items, desc=f"regularized visual ({args.mode})"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        region_dir = Path(args.data_root) / region
        image, label = load_sample(region_dir, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        gt = label == 1
        base_prob = read_prob(Path(args.base_prob_root) / region / "prob" / f"{stem}_prob.tif")
        fine_prob = read_prob(Path(args.finetune_prob_root) / region / "prob" / f"{stem}_prob.tif")
        base = (base_prob >= args.base_threshold) & valid
        pred = (fine_prob >= args.finetune_threshold) & valid
        regularized = regularize_mask(
            pred,
            valid,
            min_area=args.min_area,
            max_hole_area=args.max_hole_area,
            close_size=args.close_size,
            epsilon_ratio=args.epsilon_ratio,
            epsilon_pixels=args.epsilon_pixels,
        )
        rgb = make_rgb(image, args.rgb_time)
        fig = make_figure(rgb, gt, base, pred, regularized, valid, args.panel_size, args.mode)
        out_dir = out_root / region
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_fig4_1_regularized_{args.mode}.png"
        fig.save(out_path, dpi=(args.dpi, args.dpi), compress_level=4)
        rows.append({"region": region, "sample": sample, "figure": str(out_path.resolve())})

    manifest = out_root / f"fig4_1_regularized_{args.mode}_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["region", "sample", "figure"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Created {len(rows)} figures in: {out_root.resolve()}")
    print(f"[INFO] Manifest: {manifest.resolve()}")


if __name__ == "__main__":
    main()
