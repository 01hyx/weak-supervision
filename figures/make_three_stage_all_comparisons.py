"""生成原始权重、区域微调和频域微调的全部样本五列对比图及指标。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from make_frequency_best_result_figures import make_rgb, mask_panel, read_prob, resize
from test_manual_tiles import IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm


def precision_recall(cm):
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def make_figure(rgb, gt, base, adapted, frequency, valid, panel_size):
    panels = [
        resize(rgb, panel_size),
        resize(mask_panel(gt, valid), panel_size, True),
        resize(mask_panel(base, valid), panel_size, True),
        resize(mask_panel(adapted, valid), panel_size, True),
        resize(mask_panel(frequency, valid), panel_size, True),
    ]
    gap = max(12, panel_size // 48)
    outer = max(10, panel_size // 64)
    canvas = Image.new("RGB", (outer * 2 + panel_size * 5 + gap * 4, outer * 2 + panel_size), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (outer + index * (panel_size + gap), outer))
    return canvas


def summarize(cm):
    metrics = metrics_from_cm(cm)
    precision, recall = precision_recall(cm)
    return {
        "oa_percent": metrics["oa"],
        "precision": precision,
        "recall": recall,
        "f1": metrics["f1"][1],
        "target_iou": metrics["iou"][1],
        "miou": metrics["miou"],
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--base-root", default="manual_tile_prediction_tifs_all")
    parser.add_argument("--adapted-root", default="manual_tile_prediction_tifs_finetune_head")
    parser.add_argument("--frequency-root", default="frequency_prediction_tifs_all")
    parser.add_argument("--output-dir", default="three_stage_all_comparison_figures")
    parser.add_argument("--panel-size", type=int, default=1024)
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    cms = {
        part: {model: np.zeros((2, 2), dtype=np.int64) for model in ("base", "adapted", "frequency")}
        for part in ("train", "val", "all")
    }
    rows = []
    thresholds = {"base": 0.12, "adapted": 0.65, "frequency": 0.60}
    roots = {"base": args.base_root, "adapted": args.adapted_root, "frequency": args.frequency_root}

    for part in ("train", "val"):
        for item in tqdm(split[part], desc=f"{part} figures"):
            region, sample = item["region"], item["sample"]
            stem = Path(sample).stem
            image, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
            valid = label != IGNORE_LABEL
            gt = label == 1
            predictions = {}
            sample_metrics = {}
            sample_cms = {}
            for model in ("base", "adapted", "frequency"):
                prob_path = Path(roots[model]) / region / "prob" / f"{stem}_prob.tif"
                predictions[model] = (read_prob(prob_path) >= thresholds[model]) & valid
                cm = confusion_matrix(predictions[model].astype(np.uint8), label)
                sample_cms[model] = cm
                cms[part][model] += cm
                cms["all"][model] += cm
                sample_metrics[model] = metrics_from_cm(cm)

            path = output / part / region / f"{stem}_three_stage_compare.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            make_figure(
                make_rgb(image), gt, predictions["base"], predictions["adapted"],
                predictions["frequency"], valid, args.panel_size
            ).save(path, dpi=(300, 300), compress_level=4)
            row = {
                "split": part, "region": region, "sample": sample, "figure": str(path.resolve()),
                "adapted_vs_base": sample_metrics["adapted"]["miou"] - sample_metrics["base"]["miou"],
                "frequency_vs_adapted": sample_metrics["frequency"]["miou"] - sample_metrics["adapted"]["miou"],
            }
            for model in ("base", "adapted", "frequency"):
                precision, recall = precision_recall(sample_cms[model])
                row.update({
                    f"{model}_oa_percent": sample_metrics[model]["oa"],
                    f"{model}_precision": precision,
                    f"{model}_recall": recall,
                    f"{model}_f1": sample_metrics[model]["f1"][1],
                    f"{model}_target_iou": sample_metrics[model]["iou"][1],
                    f"{model}_miou": sample_metrics[model]["miou"],
                })
            rows.append(row)

    summary = {
        part: {model: summarize(cm) for model, cm in model_cms.items()}
        for part, model_cms in cms.items()
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "all_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "all_metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["split", "model", "samples", "oa_percent", "precision", "recall", "f1", "target_iou", "miou"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for part in ("train", "val", "all"):
            for model in ("base", "adapted", "frequency"):
                writer.writerow({
                    "split": part, "model": model,
                    "samples": len(split[part]) if part != "all" else len(split["train"]) + len(split["val"]),
                    **{key: summary[part][model][key] for key in fields[3:]},
                })
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Created {len(rows)} figures: {output.resolve()}")


if __name__ == "__main__":
    main()
