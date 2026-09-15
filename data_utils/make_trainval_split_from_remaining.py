import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a train/val split from remaining samples CSV/JSON.")
    parser.add_argument("--filtered-split", default="manual_tiles_filtered_problem_removed/filtered_split.json")
    parser.add_argument("--output-json", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.filtered_split).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = list(payload["train"])
    rng = random.Random(args.seed)

    by_region = {}
    for item in samples:
        by_region.setdefault(item["region"], []).append(item)

    train = []
    val = []
    for region, items in sorted(by_region.items()):
        rng.shuffle(items)
        split = max(1, min(len(items) - 1, int(round(len(items) * args.train_ratio))))
        train.extend(items[:split])
        val.extend(items[split:])
        print(f"{region}: train={split}, val={len(items) - split}")

    out = {
        "train": train,
        "val": val,
        "excluded": payload.get("excluded", []),
        "unmatched": payload.get("unmatched", []),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved: {output_json}")


if __name__ == "__main__":
    main()
