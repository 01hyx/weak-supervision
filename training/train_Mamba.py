import os
import sys
import warnings
import re
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
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

from utils_Mamba import *
from model.UNetFormer import UNetFormer
from model.RS3Mamba import RS3Mamba, load_pretrained_ckpt


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


class LineFilterStream:
    def __init__(self, stream, allow_patterns):
        self.stream = stream
        self.allow_patterns = allow_patterns
        self._buffer = ""

    def _allow(self, line):
        stripped = line.strip()
        if not stripped:
            return False
        if stripped == "---":
            return True
        if stripped.startswith("[") and any(ch.isdigit() for ch in stripped):
            return True
        for pattern in self.allow_patterns:
            if pattern.search(line):
                return True
        return False

    def _write_line(self, line):
        if self._allow(line):
            self.stream.write(line)

    def write(self, data):
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line + "\n")

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        self.stream.flush()


RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = os.path.join('./results_shixun', RUN_TS)


def setup_log_file(log_dir=RUN_DIR):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'train_log_{RUN_TS}.txt')
    log_fp = open(log_path, 'a', buffering=1, encoding='utf-8')

    allow_patterns = [
        re.compile(r"^\[INFO\] Evaluation skipped unreadable samples"),
        re.compile(r"^Confusion matrix:"),
        re.compile(r"pixels processed"),
        re.compile(r"^Total accuracy"),
        re.compile(r"^F1Score:"),
        re.compile(r"^mean F1Score"),
        re.compile(r"^Kappa:"),
        re.compile(r"MIoU"),
        re.compile(r"^Train \(epoch"),
        re.compile(r"^\[WARN\] Skip unreadable sample:"),
        re.compile(r"^background:"),
        re.compile(r"^target:"),
    ]
    log_stream = LineFilterStream(log_fp, allow_patterns)
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)

    print(f"[INFO] Training log file: {log_path}")
    return log_path


LOG_FILE_PATH = setup_log_file()


def _tqdm(*args, **kwargs):
    # Send progress bars to the real console to avoid polluting the log file.
    kwargs.setdefault("file", sys.__stdout__)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


if MODEL == 'UNetformer':
    # UNetFormer in this repo is configured for RGB-like inputs.
    raise ValueError("UNetformer is not configured for 36-channel multitemporal input in this script.")
elif MODEL == 'RS3Mamba':
    # Disable timm online weight download; VSSM weights are loaded from local checkpoint below.
    net = RS3Mamba(
        num_classes=N_CLASSES,
        in_channels=IN_CHANNELS,
        pretrained=False,
        use_phenology_fusion=True,
        time_steps=TIME_STEPS,
        bands_per_step=BANDS_PER_STEP,
        phenology_prior_mode="data",
    ).cuda()
    net = load_pretrained_ckpt(net)
else:
    raise ValueError(f"Unsupported MODEL: {MODEL}")

params = 0
for _, param in net.named_parameters():
    params += param.nelement()
print(params)

print(
    "training : ",
    str(len(train_ids)) + ", testing : " + str(len(test_ids)) + ", Stride_Size : " + str(Stride_Size) + ", BATCH_SIZE : " + str(BATCH_SIZE)
)

train_set = ISPRS_dataset(train_ids, cache=CACHE, augmentation=True)
loader_kwargs = {
    'dataset': train_set,
    'batch_size': BATCH_SIZE,
    'shuffle': True,
    'num_workers': NUM_WORKERS,
    'pin_memory': PIN_MEMORY,
    'drop_last': True,
}
if NUM_WORKERS > 0:
    loader_kwargs['prefetch_factor'] = 2
train_loader = torch.utils.data.DataLoader(**loader_kwargs)

base_lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def infer_one_image(net, image_chw, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    image_hwc = image_chw.transpose((1, 2, 0))
    h, w = image_hwc.shape[:2]

    pred_sum = np.zeros((h, w, N_CLASSES), dtype=np.float32)
    pred_count = np.zeros((h, w, 1), dtype=np.float32)

    total_windows = count_sliding_window(image_hwc, step=stride, window_size=window_size)
    total_batches = (total_windows + batch_size - 1) // batch_size

    for coords in _tqdm(
        grouper(batch_size, sliding_window(image_hwc, step=stride, window_size=window_size)),
        total=total_batches,
        leave=False,
    ):
        image_patches = [
            np.copy(image_hwc[x:x + win_h, y:y + win_w]).transpose((2, 0, 1))
            for x, y, win_h, win_w in coords
        ]
        image_patches = np.asarray(image_patches, dtype=np.float32)
        image_patches = torch.from_numpy(image_patches).cuda(non_blocking=True)

        outs = net(image_patches).detach().cpu().numpy()

        for out, (x, y, win_h, win_w) in zip(outs, coords):
            out = out.transpose((1, 2, 0))
            pred_sum[x:x + win_h, y:y + win_w] += out
            pred_count[x:x + win_h, y:y + win_w] += 1.0

    pred_sum /= np.maximum(pred_count, 1e-6)
    return np.argmax(pred_sum, axis=-1).astype(np.uint8)


def test(net, eval_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    all_preds = []
    all_gts = []
    skipped = 0

    with torch.no_grad():
        for sample_name in _tqdm(eval_ids, total=len(eval_ids), leave=False):
            try:
                image, gt = load_sample(sample_name, ignore_nodata=True)
            except (RasterioIOError, ValueError) as exc:
                skipped += 1
                print(f"[WARN] Skip eval sample: {sample_name} ({exc})")
                continue

            pred = infer_one_image(net, image, stride=stride, batch_size=batch_size, window_size=window_size)

            valid = gt != IGNORE_LABEL
            all_preds.append(pred[valid])
            all_gts.append(gt[valid])

    if len(all_preds) == 0:
        raise RuntimeError("No valid evaluation samples available after filtering unreadable files.")

    if skipped > 0:
        print(f"[INFO] Evaluation skipped unreadable samples: {skipped}")

    miou = metrics(np.concatenate(all_preds), np.concatenate(all_gts))
    if all:
        return miou, all_preds, all_gts
    return miou


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS):
    weights = weights.cuda()

    iter_ = 0
    miou_best = 0.0
    save_dir = RUN_DIR
    os.makedirs(save_dir, exist_ok=True)

    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 2:
                data, target = batch
                batch_positions = None
                quality_score = None
                pad_mask = None
            else:
                data, target, batch_positions, quality_score, pad_mask = batch

            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if batch_positions is not None:
                batch_positions = batch_positions.cuda(non_blocking=True)
            if quality_score is not None:
                quality_score = quality_score.cuda(non_blocking=True)
            if pad_mask is not None:
                pad_mask = pad_mask.cuda(non_blocking=True)

            optimizer.zero_grad()
            output = net(
                data,
                batch_positions=batch_positions,
                quality_score=quality_score,
                pad_mask=pad_mask,
            )
            loss = loss_calc(output, target, weights)

            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                pred = np.argmax(output.detach().cpu().numpy()[0], axis=0)
                gt = target.detach().cpu().numpy()[0]
                print(
                    'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLr: {:.6f}\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                        e,
                        epochs,
                        batch_idx,
                        len(train_loader),
                        100.0 * batch_idx / len(train_loader),
                        optimizer.state_dict()['param_groups'][0]['lr'],
                        float(loss.item()),
                        accuracy(pred, gt),
                    )
                )

            iter_ += 1

            if iter_ % 500 == 0 and len(test_ids) > 0:
                net.eval()
                miou = test(net, test_ids, all=False, stride=Stride_Size, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
                net.train()

                if miou > miou_best:
                    save_path = os.path.join(save_dir, f'{MODEL}_epoch{e}_miou{miou:.4f}.pth')
                    torch.save(net.state_dict(), save_path)
                    print(f"Saved best model to {save_path}")
                    miou_best = miou

        if scheduler is not None:
            scheduler.step()


def main():
    if MODE == 'Train':
        train(net, optimizer, 50, scheduler)
    elif MODE == 'Test':
        test_ckpt = ''
        if not test_ckpt:
            raise ValueError("Please set test_ckpt path before MODE='Test'.")
        net.load_state_dict(torch.load(test_ckpt), strict=False)
        net.eval()
        miou = test(net, test_ids, all=False, stride=Stride_Size, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
        print("MIoU:", miou)
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")


if __name__ == '__main__':
    main()
