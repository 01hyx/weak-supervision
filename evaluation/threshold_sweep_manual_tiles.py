import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    DEFAULT_STRIDE,
    IGNORE_LABEL,
    LABELS,
    N_CLASSES,
    WINDOW_SIZE,
    build_model,
    collect_valid_sample_ids,
    compute_time_quality,
    count_sliding_window,
    discover_regions,
    grouper,
    load_checkpoint,
    load_sample,
    metrics_from_cm,
    sliding_window,
)


DEFAULT_CKPT = Path(__file__).resolve().parent / "results_shixun" / "RS3Mamba_epoch25_miou0.8634.pth"
DEFAULT_OUTPUT_DIR = "threshold_sweep_manual_tiles"


def make_thresholds(start, end, step):
    count = int(round((end - start) / step)) + 1
    thresholds = np.asarray([start + i * step for i in range(count)], dtype=np.float32)
    thresholds = np.clip(thresholds, 0.0, 1.0)
    return np.unique(np.round(thresholds, 6))


def infer_one_image_target_prob(net, image_chw, device, stride, batch_size, window_size):
    image_hwc = image_chw.transpose((1, 2, 0))
    h, w = image_hwc.shape[:2]
    prob_sum = np.zeros((h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.float32)

    total_windows = count_sliding_window(image_hwc, step=stride, window_size=window_size)
    total_batches = (total_windows + batch_size - 1) // batch_size

    for coords in tqdm(
        grouper(batch_size, sliding_window(image_hwc, step=stride, window_size=window_size)),
        total=total_batches,
        leave=False,
    ):
        patches = []
        positions = []
        quality = []
        pad_mask = []
        for x, y, win_h, win_w in coords:
            patch = np.copy(image_hwc[x:x + win_h, y:y + win_w]).transpose((2, 0, 1))
            pos, qua, pad = compute_time_quality(patch)
            patches.append(patch)
            positions.append(pos)
            quality.append(qua)
            pad_mask.append(pad)

        patches = torch.from_numpy(np.asarray(patches, dtype=np.float32)).to(device)
        positions = torch.from_numpy(np.asarray(positions, dtype=np.float32)).to(device)
        quality = torch.from_numpy(np.asarray(quality, dtype=np.float32)).to(device)
        pad_mask = torch.from_numpy(np.asarray(pad_mask, dtype=np.bool_)).to(device)

        logits = net(
            patches,
            batch_positions=positions,
            quality_score=quality,
            pad_mask=pad_mask,
        )
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        for prob, (x, y, win_h, win_w) in zip(probs, coords):
            prob_sum[x:x + win_h, y:y + win_w] += prob
            pred_count[x:x + win_h, y:y + win_w] += 1.0

    return prob_sum / np.maximum(pred_count, 1e-6)


def add_confusion_for_thresholds(cms, target_prob, gt, thresholds):
    valid = gt != IGNORE_LABEL
    gt_valid = gt[valid].astype(np.uint8)
    prob_valid = target_prob[valid]

    gt_zero = gt_valid == 0
    gt_one = gt_valid == 1
    for index, threshold in enumerate(thresholds):
        pred_one = prob_valid >= threshold
        pred_zero = ~pred_one
        cms[index, 0, 0] += np.count_nonzero(gt_zero & pred_zero)
        cms[index, 0, 1] += np.count_nonzero(gt_zero & pred_one)
        cms[index, 1, 0] += np.count_nonzero(gt_one & pred_zero)
        cms[index, 1, 1] += np.count_nonzero(gt_one & pred_one)


def summarize_thresholds(thresholds, cms):
    rows = []
    for threshold, cm in zip(thresholds, cms):
        metrics = metrics_from_cm(cm)
        rows.append({
            "threshold": float(threshold),
            "confusion_matrix": cm.tolist(),
            **metrics,
        })
    return rows


def best_row(rows, key):
    return max(rows, key=lambda item: item[key])


def print_best(title, row):
    print(f"\n===== Best threshold by MIoU: {title} =====")
    print(f"threshold: {row['threshold']:.4f}")
    print(f"OA       : {row['oa']:.2f}")
    print(f"target R : {row['class_acc'][1]:.4f}")
    print(f"target F1: {row['f1'][1]:.4f}")
    print(f"target IoU: {row['iou'][1]:.4f}")
    print(f"MIoU     : {row['miou']:.4f}")
    print(np.asarray(row["confusion_matrix"]))


def evaluate_region_thresholds(net, region_dir, device, stride, batch_size, window_size, thresholds):
    sample_ids = collect_valid_sample_ids(region_dir)
    cms = np.zeros((len(thresholds), N_CLASSES, N_CLASSES), dtype=np.int64)
    skipped = []

    print(f"\n[INFO] Region: {region_dir.name}, samples: {len(sample_ids)}")
    with torch.no_grad():
        for sample_name in tqdm(sample_ids, desc=region_dir.name):
            try:
                image, gt = load_sample(region_dir, sample_name, ignore_nodata=True)
                target_prob = infer_one_image_target_prob(
                    net, image, device, stride, batch_size, window_size
                )
                add_confusion_for_thresholds(cms, target_prob, gt, thresholds)
            except (RasterioIOError, ValueError, RuntimeError) as exc:
                skipped.append({"sample": sample_name, "error": str(exc)})
                print(f"[WARN] Skip sample: {sample_name} ({exc})")

    rows = summarize_thresholds(thresholds, cms)
    best = best_row(rows, "miou")
    print_best(region_dir.name, best)
    return {
        "region": region_dir.name,
        "samples": len(sample_ids),
        "skipped": skipped,
        "thresholds": rows,
        "best_miou": best,
    }, cms


def save_results(region_results, overall_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_best = best_row(overall_rows, "miou")
    payload = {
        "labels": LABELS,
        "regions": region_results,
        "overall": {
            "thresholds": overall_rows,
            "best_miou": overall_best,
        },
    }

    json_path = output_dir / "threshold_sweep_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "threshold_sweep_overall.csv"
    fieldnames = [
        "threshold", "pixels", "oa", "background_acc", "target_acc",
        "background_f1", "target_f1", "mean_f1", "kappa",
        "background_iou", "target_iou", "miou",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in overall_rows:
            writer.writerow({
                "threshold": item["threshold"],
                "pixels": item["pixels"],
                "oa": item["oa"],
                "background_acc": item["class_acc"][0],
                "target_acc": item["class_acc"][1],
                "background_f1": item["f1"][0],
                "target_f1": item["f1"][1],
                "mean_f1": item["mean_f1"],
                "kappa": item["kappa"],
                "background_iou": item["iou"][0],
                "target_iou": item["iou"][1],
                "miou": item["miou"],
            })

    print(f"\n[INFO] Saved JSON: {json_path}")
    print(f"[INFO] Saved CSV : {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Find the best target probability threshold.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--regions", nargs="*", default=None, help="Region folder names. Default: all regions.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)

    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Checkpoint: {args.ckpt}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Thresholds: {thresholds[0]:.4f} .. {thresholds[-1]:.4f} ({len(thresholds)} values)")

    regions = discover_regions(args.data_root, args.regions)
    net = build_model(device)
    load_checkpoint(net, args.ckpt, device)
    net.eval()

    region_results = []
    overall_cms = np.zeros((len(thresholds), N_CLASSES, N_CLASSES), dtype=np.int64)
    for region in regions:
        result, cms = evaluate_region_thresholds(
            net, region, device, args.stride, args.batch_size, WINDOW_SIZE, thresholds
        )
        region_results.append(result)
        overall_cms += cms

    overall_rows = summarize_thresholds(thresholds, overall_cms)
    print_best("overall", best_row(overall_rows, "miou"))
    save_results(region_results, overall_rows, args.output_dir)


if __name__ == "__main__":
    main()
