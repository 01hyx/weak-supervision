"""Create original/label/before/after figures for parcel split refinement."""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from make_frequency_best_result_figures import make_rgb, mask_panel, read_prob, resize
from test_manual_tiles import IGNORE_LABEL, load_sample
from tif_binary_postprocess import read_single_band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--prob-root", required=True)
    parser.add_argument("--refined-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-part", choices=["train", "val", "test"], required=True)
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--panel-size", type=int, default=768)
    args = parser.parse_args()

    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    items = [item for item in split[args.split_part] if item["region"] in args.regions]
    rows = []
    for item in tqdm(items, desc="Parcel split figures"):
        region, sample = item["region"], item["sample"]
        stem = Path(sample).stem
        image, label = load_sample(Path(args.data_root) / region, sample, ignore_nodata=True)
        valid = label != IGNORE_LABEL
        ground_truth = label == 1
        before = (read_prob(Path(args.prob_root) / region / "prob" / f"{stem}_prob.tif") >= args.threshold) & valid
        refined, _ = read_single_band(Path(args.refined_root) / region / "pred" / f"{stem}_pred.tif")
        after = (refined == 1) & valid
        panels = [
            resize(make_rgb(image), args.panel_size),
            resize(mask_panel(ground_truth, valid), args.panel_size, True),
            resize(mask_panel(before, valid), args.panel_size, True),
            resize(mask_panel(after, valid), args.panel_size, True),
        ]
        gap = max(12, args.panel_size // 48)
        outer = max(10, args.panel_size // 64)
        canvas = Image.new(
            "RGB",
            (outer * 2 + args.panel_size * 4 + gap * 3, outer * 2 + args.panel_size),
            "white",
        )
        for index, panel in enumerate(panels):
            canvas.paste(panel, (outer + index * (args.panel_size + gap), outer))
        path = Path(args.output_dir) / region / f"{stem}_parcel_split_compare.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, dpi=(300, 300), compress_level=4)
        rows.append({"region": region, "sample": sample, "figure": str(path.resolve())})

    with (Path(args.output_dir) / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Created {len(rows)} figures: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
