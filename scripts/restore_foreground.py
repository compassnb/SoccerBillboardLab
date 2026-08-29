#!/usr/bin/env python3
"""Restore SAM2-segmented people over rendered billboard replacements."""

import argparse
import json
import math
import pickle
import zipfile
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore detected foreground people over replaced billboards."
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--replaced-frames", type=Path, required=True)
    parser.add_argument("--clean-masks", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="SoccerMaster pklz")
    parser.add_argument(
        "--state-video-id",
        default=None,
        help="Zip member prefix; defaults to annotations.video_id",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/final")
    )
    parser.add_argument("--frame-pattern", default="*.jpg")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize_id(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return str(int(value))
    return str(value)


def valid_bbox(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    array = np.asarray(value, dtype=np.float32)
    return (
        array.shape == (4,)
        and np.all(np.isfinite(array))
        and array[2] > 2
        and array[3] > 2
    )


def expand_bbox(bbox_ltwh, image_width, image_height):
    x, y, width, height = np.asarray(bbox_ltwh, dtype=np.float32)
    pad_x = width * 0.06
    pad_top = height * 0.04
    pad_bottom = height * 0.07
    return np.array(
        [
            max(0.0, x - pad_x),
            max(0.0, y - pad_top),
            min(float(image_width - 1), x + width + pad_x),
            min(float(image_height - 1), y + height + pad_bottom),
        ],
        dtype=np.float32,
    )


def load_state(state_path, video_id):
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    image_member = f"{video_id}_image.pkl"
    detection_member = f"{video_id}.pkl"
    with zipfile.ZipFile(state_path, "r") as archive:
        members = set(archive.namelist())
        missing = {image_member, detection_member}.difference(members)
        if missing:
            raise KeyError(
                f"State archive is missing {sorted(missing)}; "
                f"available members: {sorted(members)}"
            )
        with archive.open(image_member) as file_pointer:
            images = pickle.load(file_pointer)
        with archive.open(detection_member) as file_pointer:
            detections = pickle.load(file_pointer)

    if "frame" in images.columns:
        images = images.sort_values("frame", kind="stable")
    images = images.reset_index(drop=True)
    groups = {
        normalize_id(image_id): group.copy()
        for image_id, group in detections.groupby("image_id")
    }
    return images, groups


def build_frame_row_map(images):
    mapping = {}
    for row_index, row in images.iterrows():
        try:
            frame_index = int(row.get("frame", row_index))
        except Exception:
            frame_index = int(row_index)
        mapping[frame_index] = row
    return mapping


def bbox_intersects_billboard(box, billboard_mask):
    height, width = billboard_mask.shape
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(width, int(np.ceil(box[2])))
    y2 = min(height, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return False
    return np.count_nonzero(billboard_mask[y1:y2, x1:x2]) >= 4


def get_overlapping_boxes(
    detections, billboard_mask, image_width, image_height
):
    if detections is None or detections.empty:
        return []
    boxes = []
    for _, detection in detections.iterrows():
        if not valid_bbox(detection.get("bbox_ltwh")):
            continue
        try:
            confidence = float(detection.get("bbox_conf", 1.0))
        except Exception:
            confidence = 0.0
        if not np.isfinite(confidence) or confidence < 0.15:
            continue
        if str(detection.get("role", "")).lower() == "ball":
            continue
        box = expand_bbox(detection["bbox_ltwh"], image_width, image_height)
        if bbox_intersects_billboard(box, billboard_mask):
            boxes.append(box)
    return boxes


def segment_foreground(predictor, original, boxes, billboard_mask):
    height, width = original.shape[:2]
    combined = np.zeros((height, width), dtype=np.uint8)
    if not boxes:
        return combined, 0

    predictor.set_image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    successful = 0
    for box in boxes:
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        mask = np.squeeze(np.asarray(masks))
        if mask.ndim != 2:
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = mask > 0

        x1, y1, x2, y2 = box
        extra_x = (x2 - x1) * 0.04
        extra_y = (y2 - y1) * 0.04
        gx1 = max(0, int(np.floor(x1 - extra_x)))
        gy1 = max(0, int(np.floor(y1 - extra_y)))
        gx2 = min(width, int(np.ceil(x2 + extra_x)))
        gy2 = min(height, int(np.ceil(y2 + extra_y)))
        gated = np.zeros_like(mask)
        gated[gy1:gy2, gx1:gx2] = mask[gy1:gy2, gx1:gx2]
        gated &= billboard_mask > 0
        if np.count_nonzero(gated) < 4:
            continue
        combined[gated] = 255
        successful += 1

    if successful == 0:
        return combined, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel, iterations=1
    )
    combined = cv2.dilate(combined, kernel, iterations=1)
    combined[billboard_mask == 0] = 0
    return combined, successful


def restore_foreground(original, replacement, foreground_mask):
    if not np.any(foreground_mask):
        return replacement
    soft_mask = cv2.GaussianBlur(
        foreground_mask, (5, 5), sigmaX=1.2, sigmaY=1.2
    )
    alpha = (soft_mask.astype(np.float32) / 255.0)[..., None]
    result = (
        original.astype(np.float32) * alpha
        + replacement.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def load_billboard_union(frame_index, object_ids, clean_root, image_shape):
    height, width = image_shape[:2]
    combined = np.zeros((height, width), dtype=np.uint8)
    for object_id in object_ids:
        mask_path = (
            clean_root / f"object_{object_id}" / f"{frame_index + 1:06d}.png"
        )
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        combined = cv2.max(combined, mask)
    return combined


def main():
    args = parse_args()
    frames_dir = args.frames.expanduser().resolve()
    replaced_frames = args.replaced_frames.expanduser().resolve()
    clean_root = args.clean_masks.expanduser().resolve()
    annotations_file = args.annotations.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    checkpoint = args.sam2_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    foreground_root = output_dir / "foreground_masks"
    final_frames = output_dir / "final_frames"
    final_video = output_dir / "final_replacement.mp4"
    summary_file = output_dir / "foreground_summary.json"

    for required_file in (annotations_file, state_path, checkpoint):
        if not required_file.exists():
            raise FileNotFoundError(required_file)
    for required_dir in (frames_dir, replaced_frames, clean_root):
        if not required_dir.is_dir():
            raise NotADirectoryError(required_dir)

    annotations = json.loads(annotations_file.read_text(encoding="utf-8"))
    video_id = str(args.state_video_id or annotations.get("video_id", "")).strip()
    if not video_id:
        raise ValueError("Set annotations.video_id or pass --state-video-id")
    object_ids = sorted(
        {int(item["object_id"]) for item in annotations["annotations"]}
    )
    frame_files = sorted(frames_dir.glob(args.frame_pattern))
    if args.max_frames > 0:
        frame_files = frame_files[: args.max_frames]
    if not frame_files:
        raise RuntimeError(f"No frames matched {frames_dir / args.frame_pattern}")

    images, detection_groups = load_state(state_path, video_id)
    frame_rows = build_frame_row_map(images)
    print(f"frames={len(frame_files)}, video_id={video_id}, objects={object_ids}")
    print("Loading SAM2 Image Predictor...")
    model = build_sam2(args.sam2_config, str(checkpoint), device=args.device)
    predictor = SAM2ImagePredictor(model)

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    first = cv2.imread(str(frame_files[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frame_files[0]}")
    height, width = first.shape[:2]
    foreground_root.mkdir(parents=True, exist_ok=True)
    final_frames.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(final_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {final_video}")

    total_boxes = 0
    total_segmented = 0
    frames_with_foreground = 0
    missing_replacement = 0
    try:
        with torch.inference_mode(), autocast_context:
            for frame_index, frame_path in enumerate(frame_files):
                original = cv2.imread(str(frame_path))
                if original is None:
                    raise RuntimeError(f"Could not read {frame_path}")
                replacement_path = replaced_frames / f"{frame_index + 1:06d}.jpg"
                replacement = cv2.imread(str(replacement_path))
                if replacement is None:
                    replacement = original.copy()
                    missing_replacement += 1

                billboard_mask = load_billboard_union(
                    frame_index, object_ids, clean_root, original.shape
                )
                row = frame_rows.get(frame_index)
                if row is None and frame_index < len(images):
                    row = images.iloc[frame_index]
                if row is None:
                    detections = None
                else:
                    detections = detection_groups.get(
                        normalize_id(row.get("id", ""))
                    )
                boxes = get_overlapping_boxes(
                    detections, billboard_mask, width, height
                )
                foreground_mask, successful = segment_foreground(
                    predictor, original, boxes, billboard_mask
                )
                final = restore_foreground(
                    original, replacement, foreground_mask
                )

                mask_output = foreground_root / f"{frame_index + 1:06d}.png"
                frame_output = final_frames / f"{frame_index + 1:06d}.jpg"
                if not cv2.imwrite(str(mask_output), foreground_mask):
                    raise RuntimeError(f"Could not write {mask_output}")
                if not cv2.imwrite(
                    str(frame_output), final, [cv2.IMWRITE_JPEG_QUALITY, 95]
                ):
                    raise RuntimeError(f"Could not write {frame_output}")
                writer.write(final)

                total_boxes += len(boxes)
                total_segmented += successful
                if np.any(foreground_mask):
                    frames_with_foreground += 1
                if (frame_index + 1) % 10 == 0:
                    print(
                        f"processed {frame_index + 1}/{len(frame_files)}; "
                        f"boxes={len(boxes)}, segmented={successful}",
                        flush=True,
                    )
    finally:
        writer.release()

    summary = {
        "video_id": video_id,
        "processed_frames": len(frame_files),
        "frames_with_foreground": frames_with_foreground,
        "candidate_boxes": total_boxes,
        "successful_segmentations": total_segmented,
        "missing_replacement_frames": missing_replacement,
        "foreground_masks": str(foreground_root),
        "final_frames": str(final_frames),
        "final_video": str(final_video),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Foreground masks:", foreground_root)
    print("Final frames:", final_frames)
    print("Final video:", final_video)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
