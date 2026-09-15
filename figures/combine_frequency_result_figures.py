"""合并训练集与验证集的最佳频域微调结果图。"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def read_rows(path, split_name):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["split"] = split_name
    return rows


def make_contact_sheet(paths, output_path, width=1920):
    images = [Image.open(path).convert("RGB") for path in paths]
    resized = []
    for image in images:
        height = round(width * image.height / image.width)
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    gap = 16
    canvas = Image.new(
        "RGB",
        (width, sum(image.height for image in resized) + gap * (len(resized) - 1)),
        "white",
    )
    y = 0
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height + gap
    canvas.save(output_path, dpi=(300, 300), compress_level=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="frequency_best_train_result_figures")
    parser.add_argument("--val-root", default="frequency_best_result_figures")
    parser.add_argument("--output-dir", default="frequency_best_all_result_figures")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    val_root = Path(args.val_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_rows(train_root / "frequency_best_result_manifest.csv", "train")
    rows += read_rows(val_root / "frequency_best_result_manifest.csv", "val")

    for row in rows:
        source = Path(row["figure"])
        destination = output_root / row["region"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        row["figure"] = str(destination.resolve())

    rows.sort(key=lambda row: float(row["miou_improvement"]), reverse=True)
    manifest_path = output_root / "frequency_best_all_result_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def mean(key, selected):
        return sum(float(row[key]) for row in selected) / max(len(selected), 1)

    summary = {}
    for split_name in ("all", "train", "val"):
        selected = rows if split_name == "all" else [row for row in rows if row["split"] == split_name]
        summary[split_name] = {
            "samples": len(selected),
            "mean_base_miou": mean("base_miou", selected),
            "mean_frequency_miou": mean("frequency_miou", selected),
            "mean_miou_improvement": mean("miou_improvement", selected),
            "mean_base_target_iou": mean("base_target_iou", selected),
            "mean_frequency_target_iou": mean("frequency_target_iou", selected),
            "mean_target_iou_improvement": mean("target_iou_improvement", selected),
            "improved_samples": sum(float(row["miou_improvement"]) > 0 for row in selected),
            "worse_samples": sum(float(row["miou_improvement"]) < 0 for row in selected),
        }
    (output_root / "frequency_best_all_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_contact_sheet(
        [Path(row["figure"]) for row in rows[:12]],
        output_root / "top_12_improvements_contact_sheet.png",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
