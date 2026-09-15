"""Train an auxiliary head for separators between adjacent maize parcels."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluate_adaptation_experiments import build_experiment_model
from finetune_manual_tiles import load_split_file, make_train_loader, seed_everything
from test_manual_tiles import IGNORE_LABEL


class InternalBoundaryHead(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, feature, output_size):
        logits = self.layers(feature)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def shift(x, dy, dx):
    output = torch.zeros_like(x)
    y_src_start, y_src_end = max(-dy, 0), x.shape[-2] - max(dy, 0)
    x_src_start, x_src_end = max(-dx, 0), x.shape[-1] - max(dx, 0)
    y_dst_start, y_dst_end = max(dy, 0), x.shape[-2] - max(-dy, 0)
    x_dst_start, x_dst_end = max(dx, 0), x.shape[-1] - max(-dx, 0)
    output[..., y_dst_start:y_dst_end, x_dst_start:x_dst_end] = x[
        ..., y_src_start:y_src_end, x_src_start:x_src_end
    ]
    return output


def internal_separator_target(label, maximum_gap=4):
    valid = label != IGNORE_LABEL
    maize = label == 1
    background = (label == 0) & valid
    separator = torch.zeros_like(maize)
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for distance in range(1, maximum_gap + 1):
        for dy, dx in directions:
            separator |= (
                background
                & shift(maize, dy * distance, dx * distance)
                & shift(maize, -dy * distance, -dx * distance)
            )
    separator = F.max_pool2d(separator.float().unsqueeze(1), 3, stride=1, padding=1) > 0
    valid_core = -F.max_pool2d(-valid.float().unsqueeze(1), 3, stride=1, padding=1) > 0.5
    return separator.float(), valid_core


def separator_loss(logits, target, valid, positive_weight):
    weight = torch.tensor([positive_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits[valid], target[valid], pos_weight=weight
    )
    probability = torch.sigmoid(logits[valid])
    target_valid = target[valid]
    intersection = torch.sum(probability * target_valid)
    dice = 1.0 - (2 * intersection + 1) / (
        torch.sum(probability) + torch.sum(target_valid) + 1
    )
    return bce + dice, bce, dice


def run_epoch(model, head, hook_cache, loader, device, optimizer, positive_weight):
    training = optimizer is not None
    head.train(training)
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "positive": 0, "valid": 0}
    batches = 0
    for batch in tqdm(loader, desc="boundary-head train" if training else "boundary-head val", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        positions = batch["positions"].to(device)
        quality = batch["quality"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        with torch.no_grad():
            model(image, batch_positions=positions, quality_score=quality, pad_mask=pad_mask)
        logits = head(hook_cache["feature"], label.shape[-2:])
        target, valid = internal_separator_target(label)
        loss, bce, dice = separator_loss(logits, target, valid, positive_weight)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["bce"] += float(bce.detach())
        totals["dice"] += float(dice.detach())
        totals["positive"] += int(torch.count_nonzero((target > 0.5) & valid))
        totals["valid"] += int(torch.count_nonzero(valid))
        batches += 1
    for key in ("loss", "bce", "dice"):
        totals[key] /= max(batches, 1)
    return totals


def validation_f1(model, head, hook_cache, loader, device):
    counts = {threshold: [0, 0, 0] for threshold in [x / 20 for x in range(2, 19)]}
    head.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="separator threshold", leave=False):
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            model(
                image,
                batch_positions=batch["positions"].to(device),
                quality_score=batch["quality"].to(device),
                pad_mask=batch["pad_mask"].to(device),
            )
            probability = torch.sigmoid(head(hook_cache["feature"], label.shape[-2:]))
            target, valid = internal_separator_target(label)
            for threshold, values in counts.items():
                pred = probability >= threshold
                values[0] += int(torch.count_nonzero(pred & (target > 0.5) & valid))
                values[1] += int(torch.count_nonzero(pred & (target <= 0.5) & valid))
                values[2] += int(torch.count_nonzero((~pred) & (target > 0.5) & valid))
    rows = []
    for threshold, (tp, fp, fn) in counts.items():
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            }
        )
    return max(rows, key=lambda item: item["f1"]), rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/experiment_frequency_boundary_supervision.json")
    parser.add_argument("--data-root", default=r"D:\s2_output\manual_tiles_maize30_stride128")
    parser.add_argument("--split-file", default="manual_tiles_filtered_problem_removed/filtered_trainval_split.json")
    parser.add_argument("--output-dir", default="internal_boundary_head_binzhou")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    model = build_experiment_model(config, device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    hook_cache = {}
    hook = model.decoder.p1.register_forward_hook(
        lambda module, inputs, output: hook_cache.__setitem__("feature", output.detach())
    )
    head = InternalBoundaryHead(64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    train_samples, val_samples = load_split_file(args.split_file, args.data_root)
    loader_args = argparse.Namespace(batch_size=args.batch_size, num_workers=0)
    train_loader = make_train_loader(train_samples, loader_args, device)
    from finetune_manual_tiles import ManualTileDataset
    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        ManualTileDataset(val_samples, augmentation=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, head, hook_cache, train_loader, device, optimizer, args.positive_weight)
        val_metrics = run_epoch(model, head, hook_cache, val_loader, device, None, args.positive_weight)
        best_threshold, threshold_rows = validation_f1(model, head, hook_cache, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "threshold": best_threshold["threshold"],
            "separator_precision": best_threshold["precision"],
            "separator_recall": best_threshold["recall"],
            "separator_f1": best_threshold["f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        payload = {
            "state_dict": head.state_dict(),
            "epoch": epoch,
            "best_threshold": best_threshold,
            "model_config": config,
        }
        torch.save(payload, output / "latest.pth")
        if best_threshold["f1"] > best_f1:
            best_f1 = best_threshold["f1"]
            torch.save(payload, output / "best.pth")
            (output / "best_metrics.json").write_text(
                json.dumps({"epoch": epoch, "best": best_threshold, "thresholds": threshold_rows}, indent=2),
                encoding="utf-8",
            )
        with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    hook.remove()


if __name__ == "__main__":
    main()
