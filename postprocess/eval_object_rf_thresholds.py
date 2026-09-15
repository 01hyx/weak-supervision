import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from test_manual_tiles import metrics_from_cm
from train_object_rf_postprocess import collect_samples, evaluate_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved object RF postprocessor at several probability thresholds.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-root", default="object_rf_threshold_eval")
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = joblib.load(args.model_path)
    clf = bundle["model"]
    model_args = argparse.Namespace(**bundle["args"])
    region_to_id = {region: idx for idx, region in enumerate(model_args.regions)}
    train_samples, val_samples = collect_samples(
        model_args.pred_root, model_args.regions, model_args.seed, model_args.train_ratio
    )
    samples = train_samples if args.split == "train" else val_samples

    rows = []
    for threshold in [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        model_args.rf_threshold = threshold
        model_args.out_root = str(Path(args.out_root) / f"thr_{threshold:.2f}")
        _, before_cm, after_cm = evaluate_split(model_args, clf, samples, region_to_id, args.split)
        before = metrics_from_cm(before_cm)
        after = metrics_from_cm(after_cm)
        for stage, metrics in (("before", before), ("rf_after", after)):
            tn, fp = (after_cm if stage == "rf_after" else before_cm)[0]
            fn, tp = (after_cm if stage == "rf_after" else before_cm)[1]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            rows.append({
                "threshold": threshold,
                "split": args.split,
                "stage": stage,
                "oa": metrics["oa"],
                "precision": precision,
                "recall": metrics["class_acc"][1],
                "target_f1": metrics["f1"][1],
                "target_iou": metrics["iou"][1],
                "miou": metrics["miou"],
            })
        print(
            f"thr={threshold:.2f}: MIoU {before['miou']:.4f}->{after['miou']:.4f}, "
            f"targetIoU {before['iou'][1]:.4f}->{after['iou'][1]:.4f}, "
            f"F1 {before['f1'][1]:.4f}->{after['f1'][1]:.4f}"
        )

    out = Path(args.out_root) / f"{args.split}_threshold_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved: {out}")


if __name__ == "__main__":
    main()
