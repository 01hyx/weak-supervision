"""两阶段搜索小波频域增强区域适应微调参数。"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


BASE_CONFIG = {
    "use_frequency_enhance": True,
    "frequency_module": "wavelet",
    "freeze_backbone": True,
    "trainable_backbone_prefixes": [],
    "backbone_lr": 1e-6,
    "head_lr": 1e-4,
    "frequency_channels": 64,
    "pretrained_weight": "results_shixun/RS3Mamba_epoch25_miou0.8634.pth",
    "split_file": "manual_tiles_filtered_problem_removed/filtered_trainval_split.json",
    "data_root": r"D:\s2_output\manual_tiles_maize30_stride128",
    "epochs": 3,
    "batch_size": 2,
    "val_batch_size": 4,
    "num_workers": 0,
    "weight_decay": 1e-4,
    "class_weights": [1.0, 1.5],
    "dice_weight": 0.5,
    "focal_gamma": 0.0,
    "tversky_weight": 0.0,
    "stride": 256,
    "threshold_start": 0.3,
    "threshold_end": 0.85,
    "threshold_step": 0.025,
    "max_train_samples": 48,
    "max_val_samples": 12,
    "seed": 42,
    "device": "cuda",
}


CANDIDATES = [
    {
        "name": "frozen_lr3e5_cw15",
        "head_lr": 3e-5,
    },
    {
        "name": "frozen_lr1e4_cw15",
        "head_lr": 1e-4,
    },
    {
        "name": "frozen_lr3e4_cw15",
        "head_lr": 3e-4,
    },
    {
        "name": "frozen_lr2e4_cw15",
        "head_lr": 2e-4,
    },
    {
        "name": "frozen_lr5e4_cw15",
        "head_lr": 5e-4,
    },
    {
        "name": "frozen_lr3e4_cw125",
        "head_lr": 3e-4,
        "class_weights": [1.0, 1.25],
    },
    {
        "name": "frozen_lr3e4_dice025",
        "head_lr": 3e-4,
        "dice_weight": 0.25,
    },
    {
        "name": "layer4_lr1e6_head1e4",
        "trainable_backbone_prefixes": ["layers.3."],
        "backbone_lr": 1e-6,
        "head_lr": 1e-4,
    },
    {
        "name": "layer4_lr3e6_head1e4",
        "trainable_backbone_prefixes": ["layers.3."],
        "backbone_lr": 3e-6,
        "head_lr": 1e-4,
    },
    {
        "name": "layer4_lr1e6_cw20",
        "trainable_backbone_prefixes": ["layers.3."],
        "backbone_lr": 1e-6,
        "head_lr": 1e-4,
        "class_weights": [1.0, 2.0],
    },
]


def run_candidate(root, candidate, force):
    name = candidate["name"]
    output_dir = root / name
    metrics_path = output_dir / "best_metrics.json"
    config_path = root / "configs" / f"{name}.json"
    config = dict(BASE_CONFIG)
    config.update(candidate)
    config.pop("name")
    config["output_dir"] = str(output_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    if force or not metrics_path.exists():
        subprocess.run(
            [sys.executable, "finetune_frequency_adaptation.py", "--config", str(config_path)],
            check=True,
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = payload["best"]
    return {
        "name": name,
        "epoch": payload["epoch"],
        "threshold": best["threshold"],
        "oa": best["oa"],
        "f1": best["f1"][1],
        "target_iou": best["iou"][1],
        "miou": best["miou"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="frequency_parameter_search")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [run_candidate(root, candidate, args.force) for candidate in CANDIDATES]
    rows.sort(key=lambda row: row["miou"], reverse=True)
    with (root / "search_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (root / "search_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
