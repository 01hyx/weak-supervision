import argparse
import csv
import json
from pathlib import Path

from test_manual_tiles import DEFAULT_DATA_ROOT, collect_valid_sample_ids, discover_regions


PROBLEM_COORDS = {
    "滨城区3镇": [
        (4, 14), (5, 13), (5, 14), (6, 13), (9, 11),
        (10, 11), (11, 10), (12, 10), (13, 6), (13, 10),
        (14, 10), (15, 10), (15, 11), (16, 11), (16, 12),
        (18, 14), (20, 12), (21, 12), (22, 11),
    ],
    "阳信县4镇": [
        (4, 2), (4, 3), (4, 4), (4, 6), (5, 5), (5, 6),
        (5, 7), (5, 31), (6, 1), (6, 2), (6, 8), (7, 0),
        (7, 8), (8, 2), (10, 7), (10, 26), (10, 37), (11, 38),
        (12, 3), (12, 4), (13, 4), (14, 4), (14, 5), (14, 6),
    ],
}


def sample_coord(sample_name):
    stem = Path(sample_name).stem
    parts = stem.split("_")
    row = None
    col = None
    for part in parts:
        if part.startswith("r") and part[1:].isdigit():
            row = int(part[1:])
        if part.startswith("c") and part[1:].isdigit():
            col = int(part[1:])
    if row is None or col is None:
        raise ValueError(f"Cannot parse row/col from sample name: {sample_name}")
    return row, col


def parse_args():
    parser = argparse.ArgumentParser(description="Create a split after excluding manually selected problem samples.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default="manual_tiles_filtered_problem_removed")
    parser.add_argument("--regions", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    excluded = []
    unmatched = []
    matched_keys = set()

    problem_sets = {
        region: {tuple(coord) for coord in coords}
        for region, coords in PROBLEM_COORDS.items()
    }

    for region_dir in discover_regions(args.data_root, args.regions):
        region = region_dir.name
        sample_names = collect_valid_sample_ids(region_dir)
        for sample_name in sample_names:
            coord = sample_coord(sample_name)
            item = {"region": region, "sample": sample_name, "row": coord[0], "col": coord[1]}
            if coord in problem_sets.get(region, set()):
                excluded.append(item)
                matched_keys.add((region, coord))
            else:
                all_samples.append(item)

    for region, coords in problem_sets.items():
        for coord in sorted(coords):
            if (region, coord) not in matched_keys:
                unmatched.append({"region": region, "row": coord[0], "col": coord[1]})

    split = {
        "train": [{"region": item["region"], "sample": item["sample"]} for item in all_samples],
        "val": [{"region": item["region"], "sample": item["sample"]} for item in all_samples],
        "excluded": [{"region": item["region"], "sample": item["sample"]} for item in excluded],
        "unmatched": unmatched,
        "note": "Problem samples selected by the user were excluded. Train and val both contain the remaining samples for filtered-set evaluation; create a separate train/val split before model selection.",
    }

    split_path = output_dir / "filtered_split.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    for name, rows in [("remaining_samples.csv", all_samples), ("excluded_problem_samples.csv", excluded)]:
        path = output_dir / name
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["region", "sample", "row", "col"])
            writer.writeheader()
            writer.writerows(rows)

    unmatched_path = output_dir / "unmatched_problem_coords.csv"
    with unmatched_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "row", "col"])
        writer.writeheader()
        writer.writerows(unmatched)

    by_region_remaining = {}
    by_region_excluded = {}
    for item in all_samples:
        by_region_remaining[item["region"]] = by_region_remaining.get(item["region"], 0) + 1
    for item in excluded:
        by_region_excluded[item["region"]] = by_region_excluded.get(item["region"], 0) + 1

    print(f"[INFO] Remaining samples: {len(all_samples)} {by_region_remaining}")
    print(f"[INFO] Excluded samples : {len(excluded)} {by_region_excluded}")
    print(f"[INFO] Unmatched coords  : {len(unmatched)}")
    if unmatched:
        for item in unmatched:
            print(f"[WARN] Unmatched: {item['region']} r{item['row']:04d} c{item['col']:04d}")
    print(f"[INFO] Saved split: {split_path}")
    print(f"[INFO] Saved CSVs : {output_dir}")


if __name__ == "__main__":
    main()
