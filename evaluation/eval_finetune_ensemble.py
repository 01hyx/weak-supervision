import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from finetune_manual_tiles import load_split_file
from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    DEFAULT_STRIDE,
    IGNORE_LABEL,
    WINDOW_SIZE,
    build_model,
    load_checkpoint,
    load_sample,
)
from threshold_sweep_manual_tiles import infer_one_image_target_prob, make_thresholds, summarize_thresholds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate probability ensemble of two fine-tuned checkpoints.")
    parser.add_argument("--split-file", default="manual_tiles_finetune_head_trial3/split.json")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--head-ckpt", default="manual_tiles_finetune_head_trial3/best.pth")
    parser.add_argument("--hard-ckpt", default="manual_tiles_finetune_hard_trial1/best.pth")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0", help="Weight for head checkpoint probability.")
    parser.add_argument("--threshold-start", type=float, default=0.50)
    parser.add_argument("--threshold-end", type=float, default=0.85)
    parser.add_argument("--threshold-step", type=float, default=0.025)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_confusion(cm, prob, gt, threshold):
    valid = gt != IGNORE_LABEL
    gt_valid = gt[valid].astype(np.uint8)
    pred = prob[valid] >= threshold
    gt_zero = gt_valid == 0
    gt_one = gt_valid == 1
    cm[0, 0] += np.count_nonzero(gt_zero & ~pred)
    cm[0, 1] += np.count_nonzero(gt_zero & pred)
    cm[1, 0] += np.count_nonzero(gt_one & ~pred)
    cm[1, 1] += np.count_nonzero(gt_one & pred)


def main():
    args = parse_args()
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    device = torch.device(args.device)
    _, val_samples = load_split_file(args.split_file, args.data_root)

    head = build_model(device)
    load_checkpoint(head, args.head_ckpt, device)
    head.eval()

    hard = build_model(device)
    load_checkpoint(hard, args.hard_ckpt, device)
    hard.eval()

    cms = np.zeros((len(alphas), len(thresholds), 2, 2), dtype=np.int64)
    with torch.no_grad():
        for region_dir, sample_name in tqdm(val_samples, desc="ensemble-val"):
            image, gt = load_sample(region_dir, sample_name, ignore_nodata=True)
            head_prob = infer_one_image_target_prob(head, image, device, args.stride, args.batch_size, WINDOW_SIZE)
            hard_prob = infer_one_image_target_prob(hard, image, device, args.stride, args.batch_size, WINDOW_SIZE)
            for alpha_index, alpha in enumerate(alphas):
                prob = alpha * head_prob + (1.0 - alpha) * hard_prob
                for threshold_index, threshold in enumerate(thresholds):
                    add_confusion(cms[alpha_index, threshold_index], prob, gt, threshold)

    overall_best = None
    for alpha_index, alpha in enumerate(alphas):
        rows = summarize_thresholds(thresholds, cms[alpha_index])
        best = max(rows, key=lambda item: item["miou"])
        print(
            f"alpha_head={alpha:.3f} threshold={best['threshold']:.3f} "
            f"miou={best['miou']:.4f} target_iou={best['iou'][1]:.4f} "
            f"oa={best['oa']:.2f} cm={best['confusion_matrix']}"
        )
        if overall_best is None or best["miou"] > overall_best["best"]["miou"]:
            overall_best = {"alpha_head": alpha, "best": best}

    print(f"OVERALL_BEST={overall_best}")


if __name__ == "__main__":
    main()
