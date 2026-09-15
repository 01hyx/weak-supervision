import argparse
import csv
import itertools
import json
import os
import warnings
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.errors import RasterioIOError
from tqdm import tqdm


warnings.filterwarnings(
    "ignore",
    message="Mapping deprecated model name .*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
    category=UserWarning,
)


WINDOW_SIZE = (256, 256)
DEFAULT_STRIDE = 128
IN_CHANNELS = 36
TIME_STEPS = 6
BANDS_PER_STEP = 6
N_CLASSES = 2
LABELS = ["background", "target"]
IGNORE_LABEL = 255
NORMALIZE_DIVISOR = 65535.0

DEFAULT_DATA_ROOT = r"D:\s2_output\manual_tiles_maize30_stride128"
DEFAULT_CKPT = r"D:\时序分类\RS3Mamba\results_shixun\RS3Mamba_epoch25_miou0.8634.pth"
DEFAULT_OUTPUT_DIR = "test_results_manual_tiles"


def list_tif_names(folder):
    names = set()
    for name in os.listdir(folder):
        lower = name.lower()
        if lower.endswith(".tif") or lower.endswith(".tiff"):
            names.add(name)
    return names


def collect_valid_sample_ids(region_dir):
    mask_folder = region_dir / "mask"
    time_folders = [region_dir / f"w{i}" for i in range(1, 7)]

    if not mask_folder.is_dir():
        raise FileNotFoundError(f"Mask folder not found: {mask_folder}")
    for folder in time_folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Time folder not found: {folder}")

    mask_names = list_tif_names(mask_folder)
    valid_names = set(mask_names)
    for folder in time_folders:
        valid_names &= list_tif_names(folder)

    valid_names = sorted(valid_names)
    if not valid_names:
        raise RuntimeError(f"No valid samples found in: {region_dir}")

    return valid_names


def read_raster(path):
    with rasterio.open(path) as src:
        return src.read()


def load_multitemporal_image(region_dir, sample_name):
    phase_arrays = []
    for i in range(1, 7):
        path = region_dir / f"w{i}" / sample_name
        arr = read_raster(path)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if arr.shape[0] != BANDS_PER_STEP:
            raise ValueError(f"Expected 6 bands in {path}, got shape {arr.shape}")
        phase_arrays.append(arr.astype(np.float32))

    data = np.concatenate(phase_arrays, axis=0)
    valid_mask = np.any(data != 0, axis=0)
    data = data / NORMALIZE_DIVISOR
    return data, valid_mask


def load_binary_mask(region_dir, sample_name):
    mask = read_raster(region_dir / "mask" / sample_name)
    if mask.ndim == 3:
        mask = mask[0]
    return (mask > 0).astype(np.int64)


def load_sample(region_dir, sample_name, ignore_nodata=True):
    data, valid_mask = load_multitemporal_image(region_dir, sample_name)
    label = load_binary_mask(region_dir, sample_name)
    if ignore_nodata:
        label = label.copy()
        label[~valid_mask] = IGNORE_LABEL
    return data, label


def compute_time_quality(data_patch):
    h, w = data_patch.shape[-2:]
    data_t = data_patch.reshape(TIME_STEPS, BANDS_PER_STEP, h, w)
    valid = np.any(data_t != 0, axis=1)
    quality = valid.reshape(TIME_STEPS, -1).mean(axis=1).astype(np.float32)
    pad_mask = quality <= 0.0
    positions = np.arange(1, TIME_STEPS + 1, dtype=np.float32)
    return positions, quality, pad_mask


def sliding_window(image_hwc, step, window_size):
    for x in range(0, image_hwc.shape[0], step):
        if x + window_size[0] > image_hwc.shape[0]:
            x = image_hwc.shape[0] - window_size[0]
        for y in range(0, image_hwc.shape[1], step):
            if y + window_size[1] > image_hwc.shape[1]:
                y = image_hwc.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(image_hwc, step, window_size):
    return sum(1 for _ in sliding_window(image_hwc, step, window_size))


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def infer_one_image(net, image_chw, device, stride, batch_size, window_size):
    image_hwc = image_chw.transpose((1, 2, 0))
    h, w = image_hwc.shape[:2]
    pred_sum = np.zeros((h, w, N_CLASSES), dtype=np.float32)
    pred_count = np.zeros((h, w, 1), dtype=np.float32)

    total_windows = count_sliding_window(image_hwc, step=stride, window_size=window_size)
    total_batches = (total_windows + batch_size - 1) // batch_size

    for coords in tqdm(
        grouper(batch_size, sliding_window(image_hwc, step=stride, window_size=window_size)),
        total=total_batches,
        leave=False,
    ):
        patches = []
        positions = []
        quality = []
        pad_mask = []
        for x, y, win_h, win_w in coords:
            patch = np.copy(image_hwc[x:x + win_h, y:y + win_w]).transpose((2, 0, 1))
            pos, qua, pad = compute_time_quality(patch)
            patches.append(patch)
            positions.append(pos)
            quality.append(qua)
            pad_mask.append(pad)

        patches = torch.from_numpy(np.asarray(patches, dtype=np.float32)).to(device)
        positions = torch.from_numpy(np.asarray(positions, dtype=np.float32)).to(device)
        quality = torch.from_numpy(np.asarray(quality, dtype=np.float32)).to(device)
        pad_mask = torch.from_numpy(np.asarray(pad_mask, dtype=np.bool_)).to(device)

        outs = net(
            patches,
            batch_positions=positions,
            quality_score=quality,
            pad_mask=pad_mask,
        ).detach().cpu().numpy()

        for out, (x, y, win_h, win_w) in zip(outs, coords):
            pred_sum[x:x + win_h, y:y + win_w] += out.transpose((1, 2, 0))
            pred_count[x:x + win_h, y:y + win_w] += 1.0

    pred_sum /= np.maximum(pred_count, 1e-6)
    return np.argmax(pred_sum, axis=-1).astype(np.uint8)


def confusion_matrix(predictions, gts):
    predictions = np.asarray(predictions).reshape(-1)
    gts = np.asarray(gts).reshape(-1)
    valid = gts != IGNORE_LABEL
    predictions = predictions[valid]
    gts = gts[valid]

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for gt, pred in zip(gts, predictions):
        if 0 <= gt < N_CLASSES and 0 <= pred < N_CLASSES:
            cm[gt, pred] += 1
    return cm


def metrics_from_cm(cm):
    total = int(np.sum(cm))
    oa = (np.trace(cm) * 100.0 / float(total)) if total > 0 else 0.0
    class_acc = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=cm.sum(axis=1) != 0,
    )

    f1 = np.zeros(N_CLASSES, dtype=np.float64)
    for i in range(N_CLASSES):
        denom = np.sum(cm[i, :]) + np.sum(cm[:, i])
        if denom > 0:
            f1[i] = 2.0 * cm[i, i] / denom

    pa = (np.trace(cm) / float(total)) if total > 0 else 0.0
    pe = (np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)) if total > 0 else 0.0
    kappa = (pa - pe) / (1 - pe) if pe != 1 else 0.0

    denom = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = np.divide(np.diag(cm), denom, out=np.zeros(N_CLASSES, dtype=np.float64), where=denom != 0)

    return {
        "pixels": total,
        "oa": float(oa),
        "class_acc": class_acc.tolist(),
        "f1": f1.tolist(),
        "mean_f1": float(np.nanmean(f1)),
        "kappa": float(kappa),
        "iou": iou.tolist(),
        "miou": float(np.nanmean(iou)),
    }


def print_metrics(title, cm, metrics):
    print(f"\n===== {title} =====")
    print("Confusion matrix:")
    print(cm)
    print(f"{metrics['pixels']} pixels processed")
    print(f"Total accuracy : {metrics['oa']:.2f}")
    for label, score in zip(LABELS, metrics["class_acc"]):
        print(f"{label}: {score:.4f}")
    print("---")
    print("F1Score:")
    for label, score in zip(LABELS, metrics["f1"]):
        print(f"{label}: {score:.4f}")
    print(f"mean F1Score: {metrics['mean_f1']:.4f}")
    print("---")
    print(f"Kappa: {metrics['kappa']:.4f}")
    print(np.asarray(metrics["iou"]))
    print(f"mean MIoU: {metrics['miou']:.4f}")


def build_model(device):
    from model.RS3Mamba import RS3Mamba

    net = RS3Mamba(
        num_classes=N_CLASSES,
        in_channels=IN_CHANNELS,
        pretrained=False,
        use_phenology_fusion=False,
        time_steps=TIME_STEPS,
        bands_per_step=BANDS_PER_STEP,
        phenology_prior_mode="data",
    ).to(device)
    return net


def load_checkpoint(net, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")


def discover_regions(data_root, requested_regions):
    data_root = Path(data_root)
    if requested_regions:
        regions = [data_root / name for name in requested_regions]
    else:
        regions = [
            child for child in data_root.iterdir()
            if child.is_dir() and (child / "mask").is_dir()
        ]
    if not regions:
        raise RuntimeError(f"No region folders found under: {data_root}")
    return regions


def evaluate_region(net, region_dir, device, stride, batch_size, window_size):
    sample_ids = collect_valid_sample_ids(region_dir)
    cm_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    skipped = []

    print(f"\n[INFO] Region: {region_dir.name}, samples: {len(sample_ids)}")
    with torch.no_grad():
        for sample_name in tqdm(sample_ids, desc=region_dir.name):
            try:
                image, gt = load_sample(region_dir, sample_name, ignore_nodata=True)
                pred = infer_one_image(net, image, device, stride, batch_size, window_size)
                cm_total += confusion_matrix(pred, gt)
            except (RasterioIOError, ValueError, RuntimeError) as exc:
                skipped.append({"sample": sample_name, "error": str(exc)})
                print(f"[WARN] Skip sample: {sample_name} ({exc})")

    result = metrics_from_cm(cm_total)
    result["region"] = region_dir.name
    result["samples"] = len(sample_ids)
    result["skipped"] = skipped
    result["confusion_matrix"] = cm_total.tolist()
    print_metrics(region_dir.name, cm_total, result)
    if skipped:
        print(f"[INFO] Skipped samples in {region_dir.name}: {len(skipped)}")
    return result, cm_total


def save_results(results, overall_result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "regions": results,
        "overall": overall_result,
    }
    json_path = output_dir / "manual_tiles_test_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "manual_tiles_test_metrics.csv"
    fieldnames = [
        "region", "samples", "skipped_count", "pixels", "oa",
        "background_acc", "target_acc", "background_f1", "target_f1",
        "mean_f1", "kappa", "background_iou", "target_iou", "miou",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results + [overall_result]:
            writer.writerow({
                "region": item["region"],
                "samples": item.get("samples", ""),
                "skipped_count": len(item.get("skipped", [])),
                "pixels": item["pixels"],
                "oa": item["oa"],
                "background_acc": item["class_acc"][0],
                "target_acc": item["class_acc"][1],
                "background_f1": item["f1"][0],
                "target_f1": item["f1"][1],
                "mean_f1": item["mean_f1"],
                "kappa": item["kappa"],
                "background_iou": item["iou"][0],
                "target_iou": item["iou"][1],
                "miou": item["miou"],
            })

    print(f"\n[INFO] Saved JSON: {json_path}")
    print(f"[INFO] Saved CSV : {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RS3Mamba on manual Bincheng/Yangxin tiles.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--regions", nargs="*", default=None, help="Region folder names. Default: all regions.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    window_size = WINDOW_SIZE

    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Checkpoint: {args.ckpt}")
    print(f"[INFO] Device: {device}")

    regions = discover_regions(args.data_root, args.regions)
    net = build_model(device)
    load_checkpoint(net, args.ckpt, device)
    net.eval()

    all_results = []
    overall_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for region in regions:
        result, cm = evaluate_region(net, region, device, args.stride, args.batch_size, window_size)
        all_results.append(result)
        overall_cm += cm

    overall_result = metrics_from_cm(overall_cm)
    overall_result["region"] = "overall"
    overall_result["samples"] = sum(item["samples"] for item in all_results)
    overall_result["skipped"] = [s for item in all_results for s in item["skipped"]]
    overall_result["confusion_matrix"] = overall_cm.tolist()
    print_metrics("overall", overall_cm, overall_result)
    save_results(all_results, overall_result, args.output_dir)


if __name__ == "__main__":
    main()
