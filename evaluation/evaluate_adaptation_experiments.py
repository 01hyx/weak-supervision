"""统一评价基础迁移、普通区域微调和频域增强区域微调。"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model.FreqTuneRS3Mamba import FreqTuneRS3Mamba
from test_manual_tiles import (
    BANDS_PER_STEP,
    IGNORE_LABEL,
    IN_CHANNELS,
    N_CLASSES,
    TIME_STEPS,
    WINDOW_SIZE,
    build_model,
    confusion_matrix,
    load_checkpoint,
    load_sample,
    metrics_from_cm,
)
from threshold_sweep_manual_tiles import infer_one_image_target_prob


def build_experiment_model(config, device):
    if config.get("use_frequency_enhance", False):
        model = FreqTuneRS3Mamba(
            num_classes=N_CLASSES,
            in_channels=IN_CHANNELS,
            pretrained=False,
            use_phenology_fusion=False,
            time_steps=TIME_STEPS,
            bands_per_step=BANDS_PER_STEP,
            phenology_prior_mode="data",
            use_frequency_enhance=True,
            frequency_module=config.get("frequency_module", "wavelet"),
        ).to(device)
    else:
        model = build_model(device)
    load_checkpoint(model, config["checkpoint"], device)
    model.eval()
    return model


def precision_recall(cm):
    tp = int(cm[1, 1])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def object_stats(mask, min_small_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([], dtype=np.int64)
    return {
        "patch_count": int(areas.size),
        "small_component_count": int(np.count_nonzero(areas < min_small_area)),
        "mean_patch_area": float(areas.mean()) if areas.size else 0.0,
    }


def stretch(channel):
    valid = channel[np.isfinite(channel) & (channel > 0)]
    if not valid.size:
        return np.zeros_like(channel, dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    return (np.clip((channel - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def rgb_image(image):
    start = 4 * 6
    return np.stack([stretch(image[start + 2]), stretch(image[start + 1]), stretch(image[start])], axis=-1)


def save_boundary_overlay(path, image, gt, pred, valid):
    rgb = rgb_image(image).copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    gt_edge = cv2.dilate(gt.astype(np.uint8), kernel) != cv2.erode(gt.astype(np.uint8), kernel)
    pred_edge = cv2.dilate(pred.astype(np.uint8), kernel) != cv2.erode(pred.astype(np.uint8), kernel)
    rgb[gt_edge & valid] = (255, 215, 0)
    rgb[pred_edge & valid] = (0, 180, 255)
    Image.fromarray(rgb).save(path)


def evaluate_experiment(config, samples, args, device, output_dir):
    model = build_experiment_model(config, device)
    threshold = float(config["threshold"])
    cm_total = np.zeros((2, 2), dtype=np.int64)
    patch_count = 0
    small_count = 0
    patch_areas = []
    boundary_dir = output_dir / "boundary_overlays" / config["experiment_name"]
    if args.save_boundary_overlays:
        boundary_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(tqdm(samples, desc=config["experiment_name"])):
        region_dir = Path(args.data_root) / item["region"]
        image, gt_label = load_sample(region_dir, item["sample"], ignore_nodata=True)
        prob = infer_one_image_target_prob(
            model, image, device, args.stride, args.batch_size, WINDOW_SIZE
        )
        valid = gt_label != IGNORE_LABEL
        pred = (prob >= threshold) & valid
        gt = gt_label == 1
        cm_total += confusion_matrix(pred.astype(np.uint8), gt_label)
        stats = object_stats(pred, args.min_small_area)
        patch_count += stats["patch_count"]
        small_count += stats["small_component_count"]
        if stats["patch_count"]:
            patch_areas.extend([stats["mean_patch_area"]] * stats["patch_count"])

        if args.save_boundary_overlays and (
            args.max_boundary_overlays <= 0 or index < args.max_boundary_overlays
        ):
            region_out = boundary_dir / item["region"]
            region_out.mkdir(parents=True, exist_ok=True)
            save_boundary_overlay(
                region_out / f"{Path(item['sample']).stem}_boundary.png",
                image,
                gt,
                pred,
                valid,
            )

    metrics = metrics_from_cm(cm_total)
    precision, recall = precision_recall(cm_total)
    return {
        "experiment": config["experiment_name"],
        "checkpoint": config["checkpoint"],
        "threshold": threshold,
        "samples": len(samples),
        "oa": metrics["oa"],
        "precision": precision,
        "recall": recall,
        "f1": metrics["f1"][1],
        "target_iou": metrics["iou"][1],
        "miou": metrics["miou"],
        "patch_count": patch_count,
        "small_component_count": small_count,
        "mean_patch_area": float(np.mean(patch_areas)) if patch_areas else 0.0,
        "confusion_matrix": cm_total.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-configs",
        nargs="+",
        default=[
            "configs/experiment_base_direct.json",
            "configs/experiment_adaptation_no_frequency.json",
            "configs/experiment_frequency_wavelet.json",
        ],
    )
    parser.add_argument(
        "--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json"
    )
    parser.add_argument("--split-part", choices=["train", "val"], default="val")
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--output-dir", default="adaptation_experiment_comparison")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--min-small-area", type=int, default=50)
    parser.add_argument("--save-boundary-overlays", action="store_true")
    parser.add_argument("--max-boundary-overlays", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = json.loads(Path(args.split).read_text(encoding="utf-8"))
    samples = payload[args.split_part]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = []
    for path in args.experiment_configs:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        if not Path(config["checkpoint"]).exists():
            print(f"[WARN] 跳过，权重不存在: {config['checkpoint']}")
            continue
        results.append(evaluate_experiment(config, samples, args, device, output_dir))

    json_path = output_dir / "experiment_metrics.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if results:
        csv_path = output_dir / "experiment_metrics.csv"
        fields = [key for key in results[0] if key != "confusion_matrix"]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in results:
                writer.writerow({key: row[key] for key in fields})
    print(f"[INFO] 评价完成: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
