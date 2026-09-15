import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from make_fig4_1_transfer_finetune_best import make_rgb, read_prob, resize_panel
from test_manual_tiles import DEFAULT_DATA_ROOT, IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm
from tif_binary_postprocess import fill_small_holes, remove_small_components, write_single_band


def regularize_mask(mask, valid, min_area=24, max_hole_area=80, close_size=3, epsilon_ratio=0.006, epsilon_pixels=1.5):
    """对模型二值结果进行轻量地块边界规则化：清理噪声、填小孔、多边形简化。"""
    work = mask.astype(bool) & valid
    work = remove_small_components(work, min_area)
    work = fill_small_holes(work, max_hole_area)

    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        work = cv2.morphologyEx(work.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        work = work & valid

    # 先按 4 邻域拆开地块，再分别规则化，避免整块轮廓简化时吞掉窄道路/沟渠。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(work.astype(np.uint8), 4)
    out = np.zeros_like(work, dtype=np.uint8)
    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        pad = 2
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(work.shape[1], x + width + pad), min(work.shape[0], y + height + pad)
        crop = (labels[y0:y1, x0:x1] == label_id).astype(np.uint8)
        contours, hierarchy = cv2.findContours(crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]
        crop_out = np.zeros_like(crop, dtype=np.uint8)
        for idx, contour in enumerate(contours):
            contour_area = abs(cv2.contourArea(contour))
            if contour_area < min_area:
                continue
            epsilon = max(float(epsilon_pixels), float(epsilon_ratio) * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            parent = hierarchy[idx][3]
            color = 1 if parent == -1 else 0
            if parent != -1 and contour_area <= max_hole_area:
                continue
            cv2.drawContours(crop_out, [approx], -1, color, thickness=-1)
        out[y0:y1, x0:x1] |= crop_out

    out = (out > 0) & valid
    out = remove_small_components(out, min_area)
    return out


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


def overlay(rgb, gt, before, after, valid):
    out = rgb.astype(np.float32)
    # 色盲友好：人工边界黄，规则化前品红，规则化后青色。
    colors = [
        (contour(before & valid, 2), np.asarray([213, 94, 0], dtype=np.float32), 0.95),
        (contour(after & valid, 2), np.asarray([0, 158, 180], dtype=np.float32), 0.95),
        (contour(gt & valid, 2), np.asarray([240, 228, 66], dtype=np.float32), 0.95),
    ]
    for mask, color, alpha in colors:
        out[mask] = (1 - alpha) * out[mask] + alpha * color
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


def draw_label(draw, x, y, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.rounded_rectangle(
        (x, y, x + bbox[2] - bbox[0] + 24, y + bbox[3] - bbox[1] + 16),
        radius=4,
        fill=(255, 255, 255),
        outline=(25, 25, 25),
        width=2,
    )
    draw.text((x + 12, y + 7), text, fill=(0, 0, 0), font=font)


def make_regularization_figure(rgb, gt, before, after, valid, panel_size=1024):
    panels = [
        resize_panel(rgb, panel_size),
        resize_panel(binary_panel(gt, valid), panel_size, nearest=True),
        resize_panel(binary_panel(before, valid), panel_size, nearest=True),
        resize_panel(binary_panel(after, valid), panel_size, nearest=True),
        resize_panel(overlay(rgb, gt, before, after, valid), panel_size),
    ]
    labels = ["(A)", "(B)", "(C)", "(D)", "(E)"]
    gap = max(18, panel_size // 42)
    outer = max(14, panel_size // 64)
    width = outer * 2 + panel_size * len(panels) + gap * (len(panels) - 1)
    height = outer * 2 + panel_size
    canvas = Image.new("RGB", (width, height), "white")
    for i, panel in enumerate(panels):
        canvas.paste(panel, (outer + i * (panel_size + gap), outer))
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(28, panel_size // 24), bold=True)
    for i, label in enumerate(labels):
        draw_label(draw, outer + i * (panel_size + gap) + 10, outer + 10, label, font)
    return canvas


def precision_recall_f1(cm):
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def metric_row(cm):
    metrics = metrics_from_cm(cm)
    precision, recall, f1 = precision_recall_f1(cm)
    return {
        "oa_percent": metrics["oa"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_iou": metrics["iou"][1],
        "miou": metrics["miou"],
    }


def main():
    parser = argparse.ArgumentParser(description="Regularize parcel boundaries from probability masks.")
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prob-root", default="frequency_boundary_prediction_tifs_all")
    parser.add_argument("--output-dir", default="parcel_boundary_regularized_best_all")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rgb-time", type=int, default=4)
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--min-area", type=int, default=24)
    parser.add_argument("--max-hole-area", type=int, default=80)
    parser.add_argument("--close-size", type=int, default=3)
    parser.add_argument("--epsilon-ratio", type=float, default=0.006)
    parser.add_argument("--epsilon-pixels", type=float, default=1.5)
    parser.add_argument("--region", default=None)
    parser.add_argument("--sample", default=None, help="Optional sample file name or stem for a single-tile demo.")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    items = split["train"] + split["val"]
    if args.region:
        items = [item for item in items if item["region"] == args.region]
    if args.sample:
        wanted = Path(args.sample).stem
        items = [item for item in items if Path(item["sample"]).stem == wanted]

    output = Path(args.output_dir)
    pred_root = output / "pred"
    fig_root = output / "figures"
    rows = []
    before_total = np.zeros((2, 2), dtype=np.int64)
    after_total = np.zeros((2, 2), dtype=np.int64)
    changed_pixels = 0

    params = {
        "threshold": args.threshold,
        "min_area": args.min_area,
        "max_hole_area": args.max_hole_area,
        "close_size": args.close_size,
        "epsilon_ratio": args.epsilon_ratio,
        "epsilon_pixels": args.epsilon_pixels,
    }

    for item in tqdm(items, desc="regularizing parcel boundaries"):
        region = item["region"]
        sample = item["sample"]
        stem = Path(sample).stem
        region_dir = Path(args.data_root) / region
        image, label = load_sample(region_dir, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        gt = label == 1
        prob = read_prob(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif")
        before = (prob >= args.threshold) & valid
        after = regularize_mask(
            before,
            valid,
            min_area=args.min_area,
            max_hole_area=args.max_hole_area,
            close_size=args.close_size,
            epsilon_ratio=args.epsilon_ratio,
            epsilon_pixels=args.epsilon_pixels,
        )

        before_cm = confusion_matrix(before.astype(np.uint8), label)
        after_cm = confusion_matrix(after.astype(np.uint8), label)
        before_total += before_cm
        after_total += after_cm
        changed = int(np.count_nonzero((before != after) & valid))
        changed_pixels += changed

        with rasterio.open(region_dir / "mask" / sample) as src:
            profile = src.profile.copy()
        out_arr = np.full(label.shape, IGNORE_LABEL, dtype=np.uint8)
        out_arr[valid] = after[valid].astype(np.uint8)
        write_single_band(pred_root / region / f"{stem}_regularized_pred.tif", out_arr, profile)

        rgb = make_rgb(image, args.rgb_time)
        fig = make_regularization_figure(rgb, gt, before, after, valid, panel_size=args.panel_size)
        fig_path = fig_root / region / f"{stem}_regularized_compare.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.save(fig_path, dpi=(args.dpi, args.dpi), compress_level=4)

        row_before = metric_row(before_cm)
        row_after = metric_row(after_cm)
        rows.append(
            {
                "region": region,
                "sample": sample,
                "figure": str(fig_path.resolve()),
                "regularized_tif": str((pred_root / region / f"{stem}_regularized_pred.tif").resolve()),
                "changed_pixels": changed,
                **{f"before_{k}": v for k, v in row_before.items()},
                **{f"after_{k}": v for k, v in row_after.items()},
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "regularized_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "samples": len(rows),
        "prob_root": args.prob_root,
        "params": params,
        "changed_pixels": changed_pixels,
        "before_confusion_matrix": before_total.tolist(),
        "after_confusion_matrix": after_total.tolist(),
        "before_metrics": metric_row(before_total),
        "after_metrics": metric_row(after_total),
        "panel_order": [
            "(A) Sentinel-2 imagery",
            "(B) manually interpreted reference samples",
            "(C) fine-tuned prediction mask",
            "(D) regularized prediction mask",
            "(E) boundary overlay: yellow=reference, orange=before, cyan=regularized",
        ],
    }
    summary_path = output / "regularized_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Created {len(rows)} regularized masks and figures in: {output.resolve()}")
    print(f"[INFO] Manifest: {manifest.resolve()}")
    print(f"[INFO] Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
