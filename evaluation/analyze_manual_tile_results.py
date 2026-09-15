import csv
import json
from pathlib import Path

import numpy as np

from test_manual_tiles import metrics_from_cm


RESULT_JSON = Path("threshold_sweep_manual_tiles") / "threshold_sweep_metrics.json"
OUTPUT_CSV = Path("threshold_sweep_manual_tiles") / "threshold_sweep_region_best.csv"


def row_from_metrics(region, best):
    return {
        "region": region,
        "threshold": best["threshold"],
        "pixels": best["pixels"],
        "oa": best["oa"],
        "background_acc": best["class_acc"][0],
        "target_recall": best["class_acc"][1],
        "background_f1": best["f1"][0],
        "target_f1": best["f1"][1],
        "mean_f1": best["mean_f1"],
        "kappa": best["kappa"],
        "background_iou": best["iou"][0],
        "target_iou": best["iou"][1],
        "miou": best["miou"],
        "cm": best["confusion_matrix"],
    }


def main():
    data = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    rows = []
    rows.append(row_from_metrics("overall", data["overall"]["best_miou"]))
    for region in data["regions"]:
        rows.append(row_from_metrics(region["region"], region["best_miou"]))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in rows[0].keys() if key != "cm"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    for row in rows:
        cm = np.asarray(row["cm"])
        tn, fp = cm[0]
        fn, tp = cm[1]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print(
            f"{row['region']}: t={row['threshold']:.2f}, "
            f"MIoU={row['miou']:.4f}, target IoU={row['target_iou']:.4f}, "
            f"target recall={row['target_recall']:.4f}, target precision={precision:.4f}, "
            f"FP={fp}, FN={fn}"
        )
    print(f"saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
