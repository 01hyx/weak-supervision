import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from make_frequency_best_result_figures import make_rgb, read_prob, resize
from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    IGNORE_LABEL,
    collect_valid_sample_ids,
    confusion_matrix,
    discover_regions,
    load_sample,
    metrics_from_cm,
)


def mask_panel(mask, valid):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask & valid] = 255
    out[~valid] = 70
    return out


def overlay_error(rgb, gt, pred, valid):
    """生成误差叠加图：蓝色为正确识别玉米，橙色为误检，紫红色为漏检。"""
    out = rgb.astype(np.float32)
    tp = gt & pred & valid
    fp = (~gt) & pred & valid
    fn = gt & (~pred) & valid

    colors = {
        "tp": np.array([0, 114, 178], dtype=np.float32),
        "fp": np.array([230, 159, 0], dtype=np.float32),
        "fn": np.array([204, 0, 121], dtype=np.float32),
    }
    alpha = {"tp": 0.42, "fp": 0.72, "fn": 0.78}
    for key, mask in (("tp", tp), ("fp", fp), ("fn", fn)):
        out[mask] = (1.0 - alpha[key]) * out[mask] + alpha[key] * colors[key]

    # 用浅色边界辅助辨认人工标注和预测边缘，避免误差块与底图混在一起。
    kernel = np.ones((3, 3), dtype=np.uint8)
    gt_edge = cv2.dilate(gt.astype(np.uint8), kernel) != cv2.erode(gt.astype(np.uint8), kernel)
    pred_edge = cv2.dilate(pred.astype(np.uint8), kernel) != cv2.erode(pred.astype(np.uint8), kernel)
    out[gt_edge & valid] = (255, 255, 255)
    out[pred_edge & valid] = 0.65 * out[pred_edge & valid] + 0.35 * np.array([255, 220, 0], dtype=np.float32)
    out[~valid] = 65
    return np.clip(out, 0, 255).astype(np.uint8)


def make_figure(rgb, gt, pred, valid, panel_size):
    panels = [
        resize(rgb, panel_size),
        resize(mask_panel(gt, valid), panel_size, True),
        resize(mask_panel(pred, valid), panel_size, True),
        resize(overlay_error(rgb, gt, pred, valid), panel_size),
    ]
    gap = max(16, panel_size // 48)
    outer = max(12, panel_size // 64)
    canvas = Image.new(
        "RGB",
        (outer * 2 + panel_size * 4 + gap * 3, outer * 2 + panel_size),
        "white",
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (outer + index * (panel_size + gap), outer))
    return canvas


def precision_recall(cm):
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def iter_samples(args):
    if args.split and str(args.split).lower() != "none":
        split = json.loads(Path(args.split).read_text(encoding="utf-8"))
        for split_part in args.split_parts:
            for item in split[split_part]:
                yield split_part, item["region"], item["sample"]
        return

    for region_dir in discover_regions(args.data_root, args.regions):
        for sample_name in collect_valid_sample_ids(region_dir):
            yield "all", region_dir.name, sample_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create four-panel figures for direct-transfer errors of the base RS3Mamba model."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-prob-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument("--output-dir", default="paper_figures/fig5_1_base_transfer_error_all")
    parser.add_argument("--figure-suffix", default="fig5_1_base_error")
    parser.add_argument("--manifest-name", default="fig5_1_base_error_manifest.csv")
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--split-parts", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--rgb-time", type=int, default=4, help="0-based period; default 4 means W5.")
    parser.add_argument("--panel-size", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []

    samples = list(iter_samples(args))
    for split_part, region, sample_name in tqdm(samples, desc="base transfer error figures"):
        stem = Path(sample_name).stem
        prob_path = Path(args.base_prob_root) / region / "prob" / f"{stem}_prob.tif"
        if not prob_path.exists():
            missing.append(str(prob_path))
            continue

        image, label = load_sample(Path(args.data_root) / region, sample_name, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        gt = label == 1
        pred = (read_prob(prob_path) >= args.threshold) & valid
        rgb = make_rgb(image, args.rgb_time)

        figure_dir = output_root / split_part / region
        figure_dir.mkdir(parents=True, exist_ok=True)
        output_path = figure_dir / f"{stem}_{args.figure_suffix}.png"
        make_figure(rgb, gt, pred, valid, args.panel_size).save(
            output_path, dpi=(args.dpi, args.dpi), compress_level=4
        )

        cm = confusion_matrix(pred.astype(np.uint8), label)
        metrics = metrics_from_cm(cm)
        precision, recall = precision_recall(cm)
        tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
        rows.append(
            {
                "split": split_part,
                "region": region,
                "sample": sample_name,
                "figure": str(output_path.resolve()),
                "threshold": args.threshold,
                "oa_percent": metrics["oa"],
                "precision": precision,
                "recall": recall,
                "f1": metrics["f1"][1],
                "target_iou": metrics["iou"][1],
                "miou": metrics["miou"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "error_pixels": fp + fn,
            }
        )

    rows.sort(key=lambda row: row["error_pixels"], reverse=True)
    if rows:
        manifest = output_root / args.manifest_name
        with manifest.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Created {len(rows)} figures: {output_root.resolve()}")
        print(f"[INFO] Manifest: {manifest.resolve()}")

    if missing:
        missing_path = output_root / "missing_base_probability_tifs.txt"
        missing_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"[WARN] Missing {len(missing)} probability files: {missing_path.resolve()}")


if __name__ == "__main__":
    main()
