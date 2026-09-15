import argparse
from collections import deque
from pathlib import Path

import numpy as np
import rasterio

try:
    import cv2
except ImportError:
    cv2 = None


def read_single_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
    return arr, profile


def write_single_band(path, arr, profile, dtype="uint8", nodata=255):
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(dtype), 1)


def connected_components(binary):
    binary = binary.astype(bool)
    if cv2 is not None:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        components = []
        for idx in range(1, num_labels):
            x, y, w, h, _ = stats[idx]
            crop = labels[y:y + h, x:x + w] == idx
            ys, xs = np.nonzero(crop)
            components.append(np.stack([ys + y, xs + x], axis=1).astype(np.int32))
        return labels.astype(np.int32), components

    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components = []
    current = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for row in range(h):
        for col in range(w):
            if not binary[row, col] or labels[row, col] != 0:
                continue

            current += 1
            queue = deque([(row, col)])
            labels[row, col] = current
            coords = []

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if binary[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = current
                        queue.append((nr, nc))

            components.append(np.asarray(coords, dtype=np.int32))

    return labels, components


def remove_small_components(binary, min_area):
    if min_area <= 1:
        return binary
    labels, components = connected_components(binary)
    out = binary.copy()
    for idx, coords in enumerate(components, start=1):
        if coords.shape[0] < min_area:
            out[labels == idx] = False
    return out


def fill_small_holes(binary, max_hole_area):
    if max_hole_area <= 0:
        return binary
    inverse = ~binary
    labels, components = connected_components(inverse)
    h, w = binary.shape
    out = binary.copy()
    for idx, coords in enumerate(components, start=1):
        touches_border = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_border and coords.shape[0] <= max_hole_area:
            out[labels == idx] = True
    return out


def object_level_refine(pred, prob=None, object_labels=None, overlap_threshold=0.55, prob_threshold=0.12, min_object_area=20):
    refined = pred.astype(bool).copy()

    if object_labels is None:
        object_labels, components = connected_components(refined)
        object_ids = range(1, len(components) + 1)
    else:
        object_ids = [obj_id for obj_id in np.unique(object_labels) if obj_id != 0]

    for obj_id in object_ids:
        obj = object_labels == obj_id
        area = int(np.count_nonzero(obj))
        if area < min_object_area:
            refined[obj] = False
            continue

        overlap = float(np.count_nonzero(pred & obj)) / float(area)
        mean_prob = float(np.mean(prob[obj])) if prob is not None and np.count_nonzero(obj) else overlap

        if overlap >= overlap_threshold or mean_prob >= prob_threshold:
            refined[obj] = True
        elif overlap <= (1.0 - overlap_threshold):
            refined[obj] = False

    return refined


def make_candidate_object_labels(pred, prob, candidate_prob_threshold):
    if prob is None or candidate_prob_threshold is None:
        return None
    candidate = (prob >= candidate_prob_threshold) | pred
    labels, _ = connected_components(candidate)
    return labels


def sessrs_object_refine(pred, prob, object_labels, args):
    if object_labels is None:
        object_labels = make_candidate_object_labels(pred, prob, args.candidate_prob_threshold)

    return object_level_refine(
        pred=pred,
        prob=prob,
        object_labels=object_labels,
        overlap_threshold=args.overlap_threshold,
        prob_threshold=args.prob_threshold,
        min_object_area=args.min_object_area,
    )


def postprocess(pred, prob, object_labels, args):
    refined = sessrs_object_refine(pred, prob, object_labels, args)
    refined = remove_small_components(refined, args.min_component_area)
    refined = fill_small_holes(refined, args.max_hole_area)
    return refined.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description="Binary object-level postprocess for GeoTIFF masks.")
    parser.add_argument("--pred-tif", required=True, help="Binary prediction tif, 0=background, 1=target.")
    parser.add_argument("--out-tif", required=True)
    parser.add_argument("--prob-tif", default=None, help="Optional target probability tif.")
    parser.add_argument("--object-tif", default=None, help="Optional object-id tif. 0 means background/no object.")
    parser.add_argument("--prob-threshold", type=float, default=0.12)
    parser.add_argument("--candidate-prob-threshold", type=float, default=None)
    parser.add_argument("--overlap-threshold", type=float, default=0.55)
    parser.add_argument("--min-object-area", type=int, default=20)
    parser.add_argument("--min-component-area", type=int, default=20)
    parser.add_argument("--max-hole-area", type=int, default=64)
    parser.add_argument("--nodata", type=int, default=255)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_arr, profile = read_single_band(args.pred_tif)
    pred = pred_arr == 1

    prob = None
    if args.prob_tif:
        prob, _ = read_single_band(args.prob_tif)
        prob = prob.astype(np.float32)
        if prob.max() > 1.0:
            prob = prob / 255.0

    object_labels = None
    if args.object_tif:
        object_labels, _ = read_single_band(args.object_tif)
        object_labels = object_labels.astype(np.int32)

    refined = postprocess(pred, prob, object_labels, args)
    valid = pred_arr != args.nodata
    out = np.full(pred_arr.shape, args.nodata, dtype=np.uint8)
    out[valid] = refined[valid]
    write_single_band(args.out_tif, out, profile, dtype="uint8", nodata=args.nodata)
    print(f"[INFO] Saved postprocessed tif: {args.out_tif}")


if __name__ == "__main__":
    main()
