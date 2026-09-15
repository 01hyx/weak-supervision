import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rasterio.errors import RasterioIOError
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from test_manual_tiles import (
    DEFAULT_DATA_ROOT,
    DEFAULT_STRIDE,
    IGNORE_LABEL,
    N_CLASSES,
    WINDOW_SIZE,
    build_model,
    collect_valid_sample_ids,
    compute_time_quality,
    discover_regions,
    load_checkpoint,
    load_sample,
    metrics_from_cm,
)
from threshold_sweep_manual_tiles import DEFAULT_CKPT, add_confusion_for_thresholds, infer_one_image_target_prob, make_thresholds, summarize_thresholds


DEFAULT_OUTPUT_DIR = "manual_tiles_finetune"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_samples(data_root, region_names, train_ratio, seed):
    rng = random.Random(seed)
    train = []
    val = []
    for region_dir in discover_regions(data_root, region_names):
        sample_ids = collect_valid_sample_ids(region_dir)
        rng.shuffle(sample_ids)
        split = max(1, min(len(sample_ids) - 1, int(round(len(sample_ids) * train_ratio))))
        train.extend((region_dir, name) for name in sample_ids[:split])
        val.extend((region_dir, name) for name in sample_ids[split:])
        print(f"[INFO] {region_dir.name}: train={split}, val={len(sample_ids) - split}")
    return train, val


class ManualTileDataset(Dataset):
    def __init__(self, samples, augmentation=True, window_size=WINDOW_SIZE):
        self.samples = list(samples)
        self.augmentation = augmentation
        self.window_size = window_size

    def __len__(self):
        return len(self.samples)

    def random_crop(self, image, label):
        crop_h, crop_w = self.window_size
        _, h, w = image.shape
        if h <= crop_h:
            x = 0
            crop_h = h
        else:
            x = random.randint(0, h - crop_h)
        if w <= crop_w:
            y = 0
            crop_w = w
        else:
            y = random.randint(0, w - crop_w)
        return image[:, x:x + crop_h, y:y + crop_w], label[x:x + crop_h, y:y + crop_w]

    def augment(self, image, label):
        if random.random() < 0.5:
            image = image[:, ::-1, :]
            label = label[::-1, :]
        if random.random() < 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def __getitem__(self, index):
        region_dir, sample_name = self.samples[index]
        image, label = load_sample(region_dir, sample_name, ignore_nodata=True)
        image, label = self.random_crop(image, label)
        if self.augmentation:
            image, label = self.augment(image, label)
        positions, quality, pad_mask = compute_time_quality(image)
        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "positions": torch.from_numpy(positions.astype(np.float32)),
            "quality": torch.from_numpy(quality.astype(np.float32)),
            "pad_mask": torch.from_numpy(pad_mask.astype(np.bool_)),
        }


def target_dice_loss(logits, target):
    valid = target != IGNORE_LABEL
    if not torch.any(valid):
        return logits.sum() * 0.0
    prob = torch.softmax(logits, dim=1)[:, 1]
    target_one = (target == 1).float()
    prob = prob[valid]
    target_one = target_one[valid]
    inter = torch.sum(prob * target_one)
    denom = torch.sum(prob) + torch.sum(target_one)
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


def target_tversky_loss(logits, target, alpha=0.4, beta=0.6):
    valid = target != IGNORE_LABEL
    if not torch.any(valid):
        return logits.sum() * 0.0
    prob = torch.softmax(logits, dim=1)[:, 1]
    target_one = (target == 1).float()
    prob = prob[valid]
    target_one = target_one[valid]
    true_pos = torch.sum(prob * target_one)
    false_pos = torch.sum(prob * (1.0 - target_one))
    false_neg = torch.sum((1.0 - prob) * target_one)
    return 1.0 - (true_pos + 1.0) / (true_pos + alpha * false_pos + beta * false_neg + 1.0)


def focal_cross_entropy(logits, target, class_weights, gamma):
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE_LABEL, reduction="none")
    valid = target != IGNORE_LABEL
    if not torch.any(valid):
        return logits.sum() * 0.0
    pt = torch.exp(-ce[valid])
    return torch.mean(((1.0 - pt) ** gamma) * ce[valid])


def train_one_epoch(net, loader, optimizer, device, class_weights, dice_weight, focal_gamma, tversky_weight):
    net.train()
    total_loss = 0.0
    total_batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        quality = batch["quality"].to(device, non_blocking=True)
        pad_mask = batch["pad_mask"].to(device, non_blocking=True)

        logits = net(image, batch_positions=positions, quality_score=quality, pad_mask=pad_mask)
        if focal_gamma and focal_gamma > 0:
            ce = focal_cross_entropy(logits, label, class_weights, focal_gamma)
        else:
            ce = F.cross_entropy(logits, label, weight=class_weights, ignore_index=IGNORE_LABEL)
        dice = target_dice_loss(logits, label)
        tversky = target_tversky_loss(logits, label)
        loss = ce + dice_weight * dice + tversky_weight * tversky

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def configure_trainable_parameters(net, trainable_prefixes):
    prefixes = tuple(prefix.strip() for prefix in trainable_prefixes.split(",") if prefix.strip())
    if not prefixes or "all" in prefixes:
        for param in net.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total = sum(p.numel() for p in net.parameters())
        print(f"[INFO] Trainable parameters: {trainable}/{total} (all)")
        return

    for name, param in net.named_parameters():
        param.requires_grad = name.startswith(prefixes)
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"[INFO] Trainable prefixes: {prefixes}")
    print(f"[INFO] Trainable parameters: {trainable}/{total}")


def evaluate_samples(net, samples, device, stride, batch_size, thresholds, max_samples=0):
    net.eval()
    cms = np.zeros((len(thresholds), N_CLASSES, N_CLASSES), dtype=np.int64)
    skipped = []
    eval_samples = samples[:max_samples] if max_samples and max_samples > 0 else samples
    with torch.no_grad():
        for region_dir, sample_name in tqdm(eval_samples, desc="val", leave=False):
            try:
                image, gt = load_sample(region_dir, sample_name, ignore_nodata=True)
                prob = infer_one_image_target_prob(net, image, device, stride, batch_size, WINDOW_SIZE)
                add_confusion_for_thresholds(cms, prob, gt, thresholds)
            except (RasterioIOError, ValueError, RuntimeError) as exc:
                skipped.append({"region": region_dir.name, "sample": sample_name, "error": str(exc)})
                print(f"[WARN] Skip val sample: {region_dir.name}/{sample_name} ({exc})")
    rows = summarize_thresholds(thresholds, cms)
    best = max(rows, key=lambda item: item["miou"])
    return best, rows, skipped


def load_split_file(split_path, data_root):
    data_root = Path(data_root)
    with Path(split_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    def restore(items):
        restored = []
        for item in items:
            restored.append((data_root / item["region"], item["sample"]))
        return restored

    return restore(payload["train"]), restore(payload["val"])


def score_training_errors(net, samples, device, stride, batch_size, threshold, max_samples=0):
    net.eval()
    scored = []
    eval_samples = samples[:max_samples] if max_samples and max_samples > 0 else samples
    with torch.no_grad():
        for region_dir, sample_name in tqdm(eval_samples, desc="hard-score", leave=False):
            try:
                image, gt = load_sample(region_dir, sample_name, ignore_nodata=True)
                prob = infer_one_image_target_prob(net, image, device, stride, batch_size, WINDOW_SIZE)
                valid = gt != IGNORE_LABEL
                pred = prob >= threshold
                gt_one = gt == 1
                target_pixels = max(int(np.count_nonzero(valid & gt_one)), 1)
                background_pixels = max(int(np.count_nonzero(valid & ~gt_one)), 1)
                fn_rate = np.count_nonzero(valid & gt_one & ~pred) / target_pixels
                fp_rate = np.count_nonzero(valid & ~gt_one & pred) / background_pixels
                hard_score = 1.0 + 3.0 * fn_rate + 2.0 * fp_rate
                scored.append({
                    "region": region_dir.name,
                    "sample": sample_name,
                    "fn_rate": float(fn_rate),
                    "fp_rate": float(fp_rate),
                    "hard_score": float(hard_score),
                })
            except (RasterioIOError, ValueError, RuntimeError) as exc:
                scored.append({
                    "region": region_dir.name,
                    "sample": sample_name,
                    "fn_rate": 0.0,
                    "fp_rate": 0.0,
                    "hard_score": 1.0,
                    "error": str(exc),
                })
    return scored


def make_train_loader(samples, args, device, sampler_weights=None):
    sampler = None
    shuffle = True
    if sampler_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler_weights, dtype=torch.double),
            num_samples=len(samples),
            replacement=True,
        )
        shuffle = False
    window_size = getattr(args, "window_size", WINDOW_SIZE)
    return DataLoader(
        ManualTileDataset(samples, augmentation=True, window_size=window_size),
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def save_split(output_dir, train_samples, val_samples):
    payload = {
        "train": [{"region": r.name, "sample": s} for r, s in train_samples],
        "val": [{"region": r.name, "sample": s} for r, s in val_samples],
    }
    with (Path(output_dir) / "split.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune RS3Mamba on manual Bincheng/Yangxin tiles.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--regions", nargs="*", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-file", default=None, help="Reuse an existing split.json instead of making a new split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trainable-prefixes", default="Fuse,decoder", help="Comma-separated parameter prefixes, or 'all'.")
    parser.add_argument("--class-weights", default="1.0,2.0")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--tversky-weight", type=float, default=0.0)
    parser.add_argument("--hard-sample", action="store_true", help="Weight training tiles by starting-checkpoint FP/FN rates.")
    parser.add_argument("--hard-threshold", type=float, default=0.7)
    parser.add_argument("--max-hard-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--threshold-start", type=float, default=0.04)
    parser.add_argument("--threshold-end", type=float, default=0.40)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_file:
        train_samples, val_samples = load_split_file(args.split_file, args.data_root)
        print(f"[INFO] Reused split: {args.split_file}")
    else:
        train_samples, val_samples = split_samples(args.data_root, args.regions, args.train_ratio, args.seed)
    save_split(output_dir, train_samples, val_samples)

    device = torch.device(args.device)
    net = build_model(device)
    load_checkpoint(net, args.ckpt, device)
    configure_trainable_parameters(net, args.trainable_prefixes)
    net.train()

    class_weights = torch.tensor(
        [float(x.strip()) for x in args.class_weights.split(",")],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        [param for param in net.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sampler_weights = None
    if args.hard_sample:
        print(f"[INFO] Scoring hard samples at threshold={args.hard_threshold:.3f}")
        hard_rows = score_training_errors(
            net, train_samples, device, args.stride, args.val_batch_size, args.hard_threshold, args.max_hard_samples
        )
        hard_by_key = {(row["region"], row["sample"]): row for row in hard_rows}
        sampler_weights = [
            hard_by_key.get((region_dir.name, sample_name), {"hard_score": 1.0})["hard_score"]
            for region_dir, sample_name in train_samples
        ]
        with (output_dir / "hard_sample_scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["region", "sample", "fn_rate", "fp_rate", "hard_score", "error"])
            writer.writeheader()
            for row in sorted(hard_rows, key=lambda item: item["hard_score"], reverse=True):
                writer.writerow({
                    "region": row["region"],
                    "sample": row["sample"],
                    "fn_rate": row["fn_rate"],
                    "fp_rate": row["fp_rate"],
                    "hard_score": row["hard_score"],
                    "error": row.get("error", ""),
                })
        top = sorted(hard_rows, key=lambda item: item["hard_score"], reverse=True)[:5]
        for item in top:
            print(
                f"[HARD] {item['region']}/{item['sample']} "
                f"fn={item['fn_rate']:.3f} fp={item['fp_rate']:.3f} score={item['hard_score']:.3f}"
            )
    train_loader = make_train_loader(train_samples, args, device, sampler_weights)
    thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    history = []
    best_miou = -1.0

    print(f"[INFO] Fine-tune checkpoint: {args.ckpt}")
    print(f"[INFO] Train samples: {len(train_samples)}, val samples: {len(val_samples)}")
    print(f"[INFO] Threshold sweep: {thresholds[0]:.3f}..{thresholds[-1]:.3f}, n={len(thresholds)}")

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            net, train_loader, optimizer, device, class_weights, args.dice_weight, args.focal_gamma, args.tversky_weight
        )
        best, _, skipped = evaluate_samples(
            net, val_samples, device, args.stride, args.val_batch_size, thresholds, args.max_val_samples
        )
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "threshold": best["threshold"],
            "pixels": best["pixels"],
            "oa": best["oa"],
            "target_acc": best["class_acc"][1],
            "target_f1": best["f1"][1],
            "target_iou": best["iou"][1],
            "miou": best["miou"],
            "skipped": len(skipped),
        }
        history.append(row)
        print(
            f"[EPOCH {epoch}] loss={loss:.4f} val_miou={best['miou']:.4f} "
            f"target_iou={best['iou'][1]:.4f} target_f1={best['f1'][1]:.4f} "
            f"threshold={best['threshold']:.3f}"
        )
        torch.save(net.state_dict(), output_dir / "latest.pth")
        if best["miou"] > best_miou:
            best_miou = best["miou"]
            torch.save(net.state_dict(), output_dir / "best.pth")
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, "best": best, "skipped": skipped}, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved new best: {output_dir / 'best.pth'}")

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    print(f"[INFO] Finished. Best val MIoU: {best_miou:.4f}")
    print(f"[INFO] Outputs: {output_dir}")


if __name__ == "__main__":
    main()
