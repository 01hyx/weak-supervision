"""生成频域微调模型加入边界监督前后的验证集对比图。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from make_frequency_best_result_figures import (
    boundary_comparison,
    make_contact_sheet,
    make_rgb,
    mask_panel,
    read_prob,
    resize,
)
from test_manual_tiles import IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm


def make_figure(rgb, gt, before, after, valid, panel_size):
    panels = [
        resize(rgb, panel_size),
        resize(mask_panel(gt, valid), panel_size, True),
        resize(mask_panel(before, valid), panel_size, True),
        resize(mask_panel(after, valid), panel_size, True),
        resize(boundary_comparison(rgb, gt, before, after, valid), panel_size),
    ]
    gap = max(12, panel_size // 48)
    outer = max(10, panel_size // 64)
    canvas = Image.new("RGB", (outer * 2 + panel_size * 5 + gap * 4, outer * 2 + panel_size), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (outer + index * (panel_size + gap), outer))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--before-root", default="frequency_best_prediction_tifs_val")
    parser.add_argument("--after-root", default="frequency_boundary_prediction_tifs_val")
    parser.add_argument("--output-dir", default="frequency_boundary_supervision_figures")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--panel-size", type=int, default=1024)
    args = parser.parse_args()

    samples = json.loads(Path(args.split).read_text(encoding="utf-8"))["val"]
    output = Path(args.output_dir)
    rows = []
    for item in tqdm(samples, desc="boundary comparisons"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        image, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        gt = label == 1
        before = (read_prob(Path(args.before_root) / region / "prob" / f"{stem}_prob.tif") >= args.threshold) & valid
        after = (read_prob(Path(args.after_root) / region / "prob" / f"{stem}_prob.tif") >= args.threshold) & valid
        before_metrics = metrics_from_cm(confusion_matrix(before.astype(np.uint8), label))
        after_metrics = metrics_from_cm(confusion_matrix(after.astype(np.uint8), label))
        path = output / region / f"{stem}_boundary_supervision_compare.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        make_figure(make_rgb(image), gt, before, after, valid, args.panel_size).save(
            path, dpi=(300, 300), compress_level=4
        )
        rows.append({
            "region": region, "sample": sample, "figure": str(path.resolve()),
            "before_miou": before_metrics["miou"], "after_miou": after_metrics["miou"],
            "miou_change": after_metrics["miou"] - before_metrics["miou"],
            "before_target_iou": before_metrics["iou"][1], "after_target_iou": after_metrics["iou"][1],
        })
    rows.sort(key=lambda row: row["miou_change"], reverse=True)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet([Path(row["figure"]) for row in rows[:6]], output / "top_6_improvements.png")
    make_contact_sheet([Path(row["figure"]) for row in rows[-6:]], output / "bottom_6_changes.png")
    print(f"[INFO] Created {len(rows)} figures: {output.resolve()}")


if __name__ == "__main__":
    main()
