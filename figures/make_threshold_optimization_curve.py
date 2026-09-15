import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from test_manual_tiles import IGNORE_LABEL, confusion_matrix, load_sample, metrics_from_cm


def load_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\Times New Roman.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def read_prob(path):
    with rasterio.open(path) as src:
        prob = src.read(1).astype(np.float32)
    if prob.max(initial=0) > 1.0:
        prob /= 255.0
    return prob


def precision_recall(cm):
    tp, fp, fn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return precision, recall


def summarize_cm(cm):
    metrics = metrics_from_cm(cm)
    precision, recall = precision_recall(cm)
    return {
        "precision": precision,
        "recall": recall,
        "f1": metrics["f1"][1],
        "miou": metrics["miou"],
        "target_iou": metrics["iou"][1],
        "oa": metrics["oa"],
    }


def evaluate_thresholds(args):
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    samples = split[args.split_part]
    thresholds = np.array(args.thresholds, dtype=np.float32)
    cms = np.zeros((len(thresholds), 2, 2), dtype=np.int64)
    per_sample = {metric: [[] for _ in thresholds] for metric in ("precision", "recall", "f1", "miou")}

    for item in tqdm(samples, desc=f"threshold sweep ({args.split_part})"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        _, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        prob = read_prob(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif")

        for idx, threshold in enumerate(thresholds):
            pred = (prob >= threshold) & valid
            cm = confusion_matrix(pred.astype(np.uint8), label)
            cms[idx] += cm
            row = summarize_cm(cm)
            for metric in per_sample:
                per_sample[metric][idx].append(row[metric])

    rows = []
    for idx, threshold in enumerate(thresholds):
        row = {"threshold": float(threshold), **summarize_cm(cms[idx])}
        for metric in per_sample:
            values = np.asarray(per_sample[metric][idx], dtype=np.float64)
            row[f"{metric}_p25"] = float(np.percentile(values, 25))
            row[f"{metric}_p75"] = float(np.percentile(values, 75))
            row[f"{metric}_mean_sample"] = float(np.mean(values))
        rows.append(row)
    return rows


def save_rows(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fig5_3_threshold_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def font(size, bold=False):
    path = load_chinese_font()
    bold_candidates = [
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\Times New Roman Bold.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    ]
    if bold:
        for candidate in bold_candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_dashed_line(draw, xy0, xy1, fill, width=3, dash=18, gap=14):
    x0, y0 = xy0
    x1, y1 = xy1
    length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if length == 0:
        return
    dx = (x1 - x0) / length
    dy = (y1 - y0) / length
    pos = 0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x0 + dx * pos, y0 + dy * pos), (x0 + dx * end, y0 + dy * end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def line_points(x_values, y_values, chart, y_min, y_max):
    left, top, right, bottom = chart
    points = []
    for x, y in zip(x_values, y_values):
        px = left + (x - 0.1) / 0.8 * (right - left)
        py = bottom - (y - y_min) / max(y_max - y_min, 1e-6) * (bottom - top)
        points.append((float(px), float(py)))
    return points


def plot_curve(rows, args, output_dir):
    x = np.array([row["threshold"] for row in rows], dtype=np.float64)
    metrics = [
        ("Precision", "precision", (217, 144, 34), (248, 217, 167)),
        ("Recall", "recall", (31, 157, 58), (194, 232, 202)),
        ("F1", "f1", (31, 74, 168), (197, 207, 245)),
        ("MIoU", "miou", (199, 53, 61), (244, 198, 201)),
    ]
    best_row = max(rows, key=lambda item: item[args.best_metric])
    best_threshold = best_row["threshold"]

    scale = 2
    width, height = 1900 * scale, 1250 * scale
    margin_l, margin_r = 185 * scale, 120 * scale
    margin_t, margin_b = 165 * scale, 165 * scale
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    axis_font = font(38 * scale, True)
    tick_font = font(31 * scale)
    label_font = font(38 * scale, True)
    legend_font = font(34 * scale, True)

    chart_left, chart_right = margin_l, margin_l + chart_w
    chart_top, chart_bottom = margin_t, margin_t + chart_h

    all_values = []
    for _, key, _, _ in metrics:
        all_values.extend([row[key] for row in rows])
        all_values.extend([row[f"{key}_p25"] for row in rows])
        all_values.extend([row[f"{key}_p75"] for row in rows])
    ymin = max(0, float(np.min(all_values)) - 0.04)
    ymax = min(1, float(np.max(all_values)) + 0.04)
    chart = (chart_left, chart_top, chart_right, chart_bottom)

    shade_l = chart_left + (max(best_threshold - 0.05, 0.1) - 0.1) / 0.8 * chart_w
    shade_r = chart_left + (min(best_threshold + 0.05, 0.9) - 0.1) / 0.8 * chart_w
    draw.rectangle((shade_l, chart_top, shade_r, chart_bottom), fill=(235, 235, 235, 220))

    for t in np.linspace(ymin, ymax, 6):
        py = chart_bottom - (t - ymin) / max(ymax - ymin, 1e-6) * chart_h
        draw.line([(chart_left, py), (chart_right, py)], fill=(235, 235, 235, 150), width=1 * scale)
        draw.line([(chart_left - 18 * scale, py), (chart_left, py)], fill=(0, 0, 0), width=3 * scale)
        tick = f"{t:.2f}"
        bbox = draw.textbbox((0, 0), tick, font=tick_font)
        draw.text((chart_left - 28 * scale - (bbox[2] - bbox[0]), py - (bbox[3] - bbox[1]) / 2), tick, fill=(0, 0, 0), font=tick_font)

    best_x = chart_left + (best_threshold - 0.1) / 0.8 * chart_w
    draw_dashed_line(draw, (best_x, chart_top), (best_x, chart_bottom), (0, 0, 0, 150), width=2 * scale)

    for idx, (label, key, color, band_color) in enumerate(metrics):
        y = np.array([row[key] for row in rows], dtype=np.float64)
        y25 = np.array([row[f"{key}_p25"] for row in rows], dtype=np.float64)
        y75 = np.array([row[f"{key}_p75"] for row in rows], dtype=np.float64)
        best_value = best_row[key]

        lower = line_points(x, y25, chart, ymin, ymax)
        upper = line_points(x, y75, chart, ymin, ymax)
        draw.polygon(upper + lower[::-1], fill=(*band_color, 70))
        pts = line_points(x, y, chart, ymin, ymax)
        draw.line(pts, fill=color, width=5 * scale, joint="curve")

        best_y = chart_bottom - (best_value - ymin) / max(ymax - ymin, 1e-6) * chart_h
        draw_dashed_line(draw, (chart_left, best_y), (chart_right, best_y), (*color, 120), width=2 * scale, dash=12 * scale, gap=12 * scale)
        draw.ellipse((best_x - 9 * scale, best_y - 9 * scale, best_x + 9 * scale, best_y + 9 * scale), fill=color, outline=(255, 255, 255), width=3 * scale)

        lx = chart_left + 70 * scale + idx * 360 * scale
        ly = chart_top - 100 * scale
        draw.line([(lx, ly + 13 * scale), (lx + 58 * scale, ly + 13 * scale)], fill=color, width=6 * scale)
        draw.ellipse((lx + 22 * scale, ly + 4 * scale, lx + 40 * scale, ly + 22 * scale), fill=color, outline=(255, 255, 255), width=2 * scale)
        draw.text((lx + 72 * scale, ly - 4 * scale), label, fill=color, font=legend_font)

    draw.line([(chart_left, chart_top), (chart_left, chart_bottom), (chart_right, chart_bottom)], fill=(0, 0, 0), width=4 * scale)
    # 论文图只显示 0.1 间隔主刻度，避免 0.65/0.675 等补充阈值标签重叠。
    for value in np.arange(0.1, 1.0, 0.1):
        px = chart_left + (value - 0.1) / 0.8 * chart_w
        draw.line([(px, chart_bottom), (px, chart_bottom + 24 * scale)], fill=(0, 0, 0), width=4 * scale)
        tick = f"{value:.1f}"
        bbox = draw.textbbox((0, 0), tick, font=tick_font)
        draw.text((px - (bbox[2] - bbox[0]) / 2, chart_bottom + 33 * scale), tick, fill=(0, 0, 0), font=tick_font)

    y_label = "Metric value"
    bbox = draw.textbbox((0, 0), y_label, font=label_font)
    label_img = Image.new("RGBA", (bbox[2] - bbox[0] + 20 * scale, bbox[3] - bbox[1] + 20 * scale), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((10 * scale, 10 * scale), y_label, fill=(0, 0, 0), font=label_font)
    label_img = label_img.rotate(90, expand=True)
    image.paste(label_img, (32 * scale, int(chart_top + chart_h / 2 - label_img.height / 2)), label_img)

    bottom_label = "Threshold"
    bbox = draw.textbbox((0, 0), bottom_label, font=axis_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, height - 82 * scale), bottom_label, fill=(0, 0, 0), font=axis_font)

    png_path = output_dir / "fig5_3_threshold_optimization_curve.png"
    pdf_path = output_dir / "fig5_3_threshold_optimization_curve.pdf"
    image.save(png_path, dpi=(args.dpi, args.dpi), compress_level=4)
    try:
        image.save(pdf_path, "PDF", resolution=args.dpi)
    except Exception:
        pdf_path = None
    return png_path, pdf_path, best_row


def parse_args():
    parser = argparse.ArgumentParser(description="Draw Fig. 5-3 threshold optimization curve.")
    parser.add_argument("--split", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--split-part", default="val", choices=["train", "val"])
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--prob-root", default="frequency_boundary_prediction_tifs_all")
    parser.add_argument("--output-dir", default="paper_figures/fig5_3_threshold_optimization_curve")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--best-metric", default="miou", choices=["precision", "recall", "f1", "miou"])
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = evaluate_thresholds(args)
    csv_path = save_rows(rows, output_dir)
    png_path, pdf_path, best = plot_curve(rows, args, output_dir)
    summary = {
        "split_part": args.split_part,
        "prob_root": args.prob_root,
        "best_metric": args.best_metric,
        "best": best,
        "csv": str(csv_path.resolve()),
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()) if pdf_path else None,
    }
    (output_dir / "fig5_3_threshold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
