#!/usr/bin/env python3
"""Fit stable continuous billboard quadrilaterals from raw tracking masks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


BIN_WIDTH = 36.0
MIN_COMPONENT_AREA = 35
MIN_ASPECT_RATIO = 2.0
MIN_BAND_THICKNESS = 5.0
MAX_BAND_THICKNESS = 180.0
MAX_MASK_IMAGE_RATIO = 0.30
EXTEND_ALONG_AXIS = 18.0
SMOOTH_WINDOW = 5
SCENE_JUMP_PIXELS = 260.0

COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 255),
    3: (0, 165, 255),
    4: (255, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit PCA billboard bands and apply temporal smoothing."
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--raw-masks",
        type=Path,
        required=True,
        help="Directory containing object_1, object_2, ...",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/geometry")
    )
    parser.add_argument("--frame-pattern", default="*.jpg")
    parser.add_argument("--fps", type=float, default=25.0)
    return parser.parse_args()


def robust_line_fit(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 3:
        raise ValueError("Not enough valid samples")
    for _ in range(7):
        if np.count_nonzero(valid) < 3:
            break
        coefficients = np.polyfit(x[valid], y[valid], 1)
        residual = np.abs(y - np.polyval(coefficients, x))
        active = residual[valid]
        median = np.median(active)
        mad = np.median(np.abs(active - median))
        threshold = max(2.5, median + 3.5 * max(mad, 0.5))
        new_valid = np.isfinite(residual) & (residual <= threshold)
        if np.array_equal(valid, new_valid):
            break
        valid = new_valid
    if np.count_nonzero(valid) < 3:
        raise ValueError("Not enough inliers after robust fitting")
    return np.polyfit(x[valid], y[valid], 1), valid


def select_mask_pixels(mask):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    threshold = max(MIN_COMPONENT_AREA, int(round(largest * 0.012)))
    selected_labels = 1 + np.flatnonzero(areas >= threshold)
    selected = np.isin(labels, selected_labels)
    ys, xs = np.where(selected)
    if len(xs) < 100:
        return None
    return np.column_stack([xs, ys]).astype(np.float64)


def fit_billboard_band(mask):
    height, width = mask.shape
    pixels = select_mask_pixels(mask)
    if pixels is None:
        return None, "too_few_pixels"
    if len(pixels) > height * width * MAX_MASK_IMAGE_RATIO:
        return None, "mask_too_large"

    if len(pixels) > 80000:
        step = int(np.ceil(len(pixels) / 80000))
        pca_pixels = pixels[::step]
    else:
        pca_pixels = pixels

    center = np.median(pca_pixels, axis=0)
    centered = pca_pixels - center
    covariance = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    direction = vectors[:, order[0]]
    aspect_ratio = (
        999.0 if values[1] <= 1e-6 else float(np.sqrt(values[0] / values[1]))
    )
    if aspect_ratio < MIN_ASPECT_RATIO:
        return None, "not_elongated"

    if direction[0] < 0 or (
        abs(direction[0]) < 1e-6 and direction[1] < 0
    ):
        direction = -direction
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    relative = pixels - center
    along = relative @ direction
    across = relative @ normal

    across_low, across_high = np.percentile(across, [1.0, 99.0])
    keep = (across >= across_low) & (across <= across_high)
    along = along[keep]
    across = across[keep]
    if len(along) < 100:
        return None, "too_few_inliers"

    u_min, u_max = np.percentile(along, [0.8, 99.2])
    axis_length = float(u_max - u_min)
    if axis_length < 45:
        return None, "band_too_short"

    number_of_bins = max(6, int(np.ceil(axis_length / BIN_WIDTH)))
    edges = np.linspace(u_min, u_max, number_of_bins + 1)
    sample_u, sample_center, sample_half = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        local = across[(along >= left) & (along < right)]
        if len(local) < 18:
            continue
        lower, upper = np.percentile(local, [4.0, 96.0])
        thickness = float(upper - lower)
        if thickness < MIN_BAND_THICKNESS or thickness > MAX_BAND_THICKNESS:
            continue
        sample_u.append((left + right) * 0.5)
        sample_center.append((lower + upper) * 0.5)
        sample_half.append(thickness * 0.5)

    if len(sample_u) < 4:
        return None, "too_few_edge_bins"
    sample_u = np.asarray(sample_u)
    sample_center = np.asarray(sample_center)
    sample_half = np.asarray(sample_half)
    center_line, center_valid = robust_line_fit(sample_u, sample_center)
    half_line, half_valid = robust_line_fit(sample_u, sample_half)
    valid = center_valid & half_valid
    if np.count_nonzero(valid) < 3:
        return None, "too_few_joint_inliers"

    typical_half = float(np.median(sample_half[valid]))
    if not (
        MIN_BAND_THICKNESS * 0.5
        <= typical_half
        <= MAX_BAND_THICKNESS * 0.5
    ):
        return None, "invalid_thickness"

    def edge_points(u):
        local_center = float(np.polyval(center_line, u))
        local_half = float(np.polyval(half_line, u))
        local_half = float(
            np.clip(
                local_half,
                max(MIN_BAND_THICKNESS * 0.5, typical_half * 0.55),
                min(MAX_BAND_THICKNESS * 0.5, typical_half * 1.65),
            )
        )
        middle = center + direction * u + normal * local_center
        return middle - normal * local_half, middle + normal * local_half

    start_a, start_b = edge_points(float(u_min - EXTEND_ALONG_AXIS))
    end_a, end_b = edge_points(float(u_max + EXTEND_ALONG_AXIS))
    quad = np.vstack([start_a, end_a, end_b, start_b])
    if not np.isfinite(quad).all():
        return None, "non_finite_corners"
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    polygon_area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if polygon_area < 200 or polygon_area > height * width * MAX_MASK_IMAGE_RATIO:
        return None, "invalid_area"

    details = {
        "aspect_ratio": aspect_ratio,
        "axis_length": axis_length,
        "average_thickness": typical_half * 2.0,
        "valid_bins": int(np.count_nonzero(valid)),
    }
    return (quad, details), None


def split_valid_segments(quads):
    valid_indices = [index for index, quad in enumerate(quads) if quad is not None]
    if not valid_indices:
        return []
    segments = []
    current = [valid_indices[0]]
    for index in valid_indices[1:]:
        previous = current[-1]
        gap = index - previous
        displacement = float(
            np.mean(np.linalg.norm(quads[index] - quads[previous], axis=1))
        )
        if gap > 3 or displacement > SCENE_JUMP_PIXELS:
            segments.append(current)
            current = [index]
        else:
            current.append(index)
    segments.append(current)
    return segments


def running_median(values, window):
    radius = window // 2
    output = np.empty_like(values)
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        output[index] = np.median(values[left:right], axis=0)
    return output


def smooth_quads(quads):
    result = [None if quad is None else quad.copy() for quad in quads]
    for segment in split_valid_segments(quads):
        if len(segment) < 3:
            continue
        first, last = segment[0], segment[-1]
        known = np.asarray(segment, dtype=np.float64)
        dense_indices = np.arange(first, last + 1)
        dense = np.empty((len(dense_indices), 4, 2), dtype=np.float64)
        raw_known = np.stack([quads[index] for index in segment])
        for corner in range(4):
            for coordinate in range(2):
                dense[:, corner, coordinate] = np.interp(
                    dense_indices,
                    known,
                    raw_known[:, corner, coordinate],
                )
        median = running_median(dense, SMOOTH_WINDOW)
        deviation = np.mean(np.linalg.norm(dense - median, axis=2), axis=1)
        smoothed = dense * 0.35 + median * 0.65
        smoothed[deviation > 45.0] = median[deviation > 45.0]
        for offset, frame_index in enumerate(dense_indices):
            result[int(frame_index)] = smoothed[offset]
    return result


def create_clean_masks(object_id, frame_files, raw_root, clean_root):
    raw_quads, fit_details = [], []
    failure_reasons = defaultdict(int)
    for frame_index in range(len(frame_files)):
        mask_path = raw_root / f"object_{object_id}" / f"{frame_index + 1:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raw_quads.append(None)
            fit_details.append(None)
            failure_reasons["missing_raw_mask"] += 1
            continue
        fitted, error = fit_billboard_band(mask)
        if fitted is None:
            raw_quads.append(None)
            fit_details.append(None)
            failure_reasons[error] += 1
        else:
            quad, details = fitted
            raw_quads.append(quad)
            fit_details.append(details)

    smooth = smooth_quads(raw_quads)
    output_dir = clean_root / f"object_{object_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = []
    for frame_index, frame_path in enumerate(frame_files):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Could not read {frame_path}")
        height, width = image.shape[:2]
        clean = np.zeros((height, width), dtype=np.uint8)
        quad = smooth[frame_index]
        if quad is not None:
            clipped = quad.copy()
            clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
            clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
            cv2.fillConvexPoly(
                clean, np.rint(clipped).astype(np.int32), 255, cv2.LINE_AA
            )
            quad_json = np.round(clipped, 3).tolist()
        else:
            quad_json = None
        output_path = output_dir / f"{frame_index + 1:06d}.png"
        if not cv2.imwrite(str(output_path), clean):
            raise RuntimeError(f"Could not write {output_path}")
        geometry.append(
            {
                "frame_index": frame_index,
                "image_name": frame_path.name,
                "quad": quad_json,
                "fit": fit_details[frame_index],
            }
        )

    print(
        f"object {object_id}: fitted {sum(q is not None for q in raw_quads)}/"
        f"{len(frame_files)}, cleaned {sum(q is not None for q in smooth)}/"
        f"{len(frame_files)}"
    )
    if failure_reasons:
        print("  failures:", dict(failure_reasons))
    return geometry


def overlay(frame, mask, color, alpha_value):
    if mask is None:
        return frame
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    alpha = (mask.astype(np.float32) / 255.0 * alpha_value)[..., None]
    color_image = np.empty_like(frame)
    color_image[:] = color
    result = (
        color_image.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2, cv2.LINE_AA)
    return result


def generate_comparison_video(
    frame_files, object_ids, raw_root, clean_root, preview_video, fps
):
    first = cv2.imread(str(frame_files[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frame_files[0]}")
    height, width = first.shape[:2]
    display_width = 960
    display_height = max(1, int(round(height * display_width / width)))
    preview_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(preview_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (display_width * 2, display_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {preview_video}")

    try:
        for frame_index, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            raw_view, clean_view = frame.copy(), frame.copy()
            for object_id in object_ids:
                raw = cv2.imread(
                    str(raw_root / f"object_{object_id}" / f"{frame_index + 1:06d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                clean = cv2.imread(
                    str(clean_root / f"object_{object_id}" / f"{frame_index + 1:06d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                color = COLORS.get(object_id, (255, 255, 255))
                raw_view = overlay(raw_view, raw, color, 0.42)
                clean_view = overlay(clean_view, clean, color, 0.42)
            for view, title in (
                (raw_view, "Raw SAM2"),
                (clean_view, "PCA fit + temporal smoothing"),
            ):
                cv2.rectangle(view, (0, 0), (width, 58), (0, 0, 0), -1)
                cv2.putText(
                    view,
                    f"{title}   Frame {frame_index + 1:06d}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(
                np.hstack(
                    [
                        cv2.resize(raw_view, (display_width, display_height)),
                        cv2.resize(clean_view, (display_width, display_height)),
                    ]
                )
            )
    finally:
        writer.release()


def main():
    args = parse_args()
    frames_dir = args.frames.expanduser().resolve()
    annotations_file = args.annotations.expanduser().resolve()
    raw_root = args.raw_masks.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    clean_root = output_dir / "clean_masks"
    geometry_file = output_dir / "geometry.json"
    preview_video = output_dir / "geometry_comparison.mp4"

    if not annotations_file.exists():
        raise FileNotFoundError(annotations_file)
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    data = json.loads(annotations_file.read_text(encoding="utf-8"))
    frame_files = sorted(frames_dir.glob(args.frame_pattern))
    if not frame_files:
        raise RuntimeError(f"No frames matched {frames_dir / args.frame_pattern}")
    object_ids = sorted({int(item["object_id"]) for item in data["annotations"]})
    if not object_ids:
        raise ValueError("No annotated objects")

    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = {
        str(object_id): create_clean_masks(
            object_id, frame_files, raw_root, clean_root
        )
        for object_id in object_ids
    }
    geometry_file.write_text(
        json.dumps(
            {
                "video_id": data.get("video_id"),
                "frame_count": len(frame_files),
                "object_names": data.get("object_names", {}),
                "objects": geometry,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    generate_comparison_video(
        frame_files,
        object_ids,
        raw_root,
        clean_root,
        preview_video,
        args.fps,
    )
    print("Clean masks:", clean_root)
    print("Geometry:", geometry_file)
    print("Comparison preview:", preview_video)


if __name__ == "__main__":
    main()
