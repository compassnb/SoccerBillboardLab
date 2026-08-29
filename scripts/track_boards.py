#!/usr/bin/env python3
"""Track multiple billboard planes from multi-frame SAM2 point prompts."""

import argparse
import inspect
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 255),
    3: (0, 165, 255),
    4: (255, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track billboard masks with SAM2 Video and multi-frame prompts."
    )
    parser.add_argument("--frames", type=Path, required=True, help="Frame directory")
    parser.add_argument(
        "--annotations", type=Path, required=True, help="Annotation JSON"
    )
    parser.add_argument(
        "--sam2-checkpoint", type=Path, required=True, help="SAM2 checkpoint"
    )
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 Hydra model config",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/tracking")
    )
    parser.add_argument("--frame-pattern", default="*.jpg")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_annotations(frames_dir, frame_pattern, annotations_file):
    if not annotations_file.exists():
        raise FileNotFoundError(annotations_file)
    if not frames_dir.is_dir():
        raise NotADirectoryError(frames_dir)

    data = json.loads(annotations_file.read_text(encoding="utf-8"))
    required = {"frame_count", "object_names", "annotations"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Annotation JSON is missing: {sorted(missing)}")

    frame_files = sorted(frames_dir.glob(frame_pattern))
    if not frame_files:
        raise RuntimeError(f"No frames matched {frames_dir / frame_pattern}")
    if int(data["frame_count"]) != len(frame_files):
        raise ValueError(
            f"Annotation JSON declares {data['frame_count']} frames, "
            f"but {len(frame_files)} files were found"
        )

    names = {int(key): str(value) for key, value in data["object_names"].items()}
    grouped = defaultdict(lambda: {"points": [], "labels": []})
    for record_index, annotation in enumerate(data["annotations"]):
        frame_index = int(annotation["frame_index"])
        object_id = int(annotation["object_id"])
        points = np.asarray(annotation["points"], dtype=np.float32)
        labels = np.asarray(annotation["labels"], dtype=np.int32)

        if frame_index < 0 or frame_index >= len(frame_files):
            raise ValueError(f"Record {record_index}: frame_index is out of range")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Record {record_index}: invalid points shape {points.shape}")
        if labels.ndim != 1 or len(labels) != len(points):
            raise ValueError(f"Record {record_index}: points/labels length mismatch")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Record {record_index}: labels must be 0 or 1")
        if object_id not in names:
            raise ValueError(f"Object {object_id} is missing from object_names")

        key = (frame_index, object_id)
        grouped[key]["points"].append(points)
        grouped[key]["labels"].append(labels)

    prompts = []
    for (frame_index, object_id), values in sorted(grouped.items()):
        prompts.append(
            {
                "frame_index": frame_index,
                "object_id": object_id,
                "object_name": names[object_id],
                "points": np.concatenate(values["points"], axis=0),
                "labels": np.concatenate(values["labels"], axis=0),
            }
        )
    if not prompts:
        raise ValueError("The annotation list is empty")
    return data, names, prompts, frame_files


def initialize_state(predictor, frames_dir):
    parameters = inspect.signature(predictor.init_state).parameters
    kwargs = {"video_path": str(frames_dir)}
    if "offload_video_to_cpu" in parameters:
        kwargs["offload_video_to_cpu"] = True
    if "offload_state_to_cpu" in parameters:
        kwargs["offload_state_to_cpu"] = True
    if "async_loading_frames" in parameters:
        kwargs["async_loading_frames"] = True
    print("init_state:", kwargs)
    return predictor.init_state(**kwargs)


def save_output_masks(frame_index, object_ids, mask_logits, raw_root):
    for position, object_id in enumerate(object_ids):
        object_id = int(object_id)
        mask = (mask_logits[position] > 0.0).detach().cpu().numpy().squeeze()
        if mask.ndim != 2:
            print("Skipping invalid mask:", frame_index, object_id, mask.shape)
            continue
        object_dir = raw_root / f"object_{object_id}"
        object_dir.mkdir(parents=True, exist_ok=True)
        output = object_dir / f"{frame_index + 1:06d}.png"
        if not cv2.imwrite(str(output), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Could not write mask: {output}")


def propagate_direction(
    predictor, inference_state, start_frame, reverse, raw_root
):
    direction = "reverse" if reverse else "forward"
    print(f"Starting {direction} propagation...")
    parameters = inspect.signature(predictor.propagate_in_video).parameters
    kwargs = {"inference_state": inference_state}
    if "start_frame_idx" in parameters:
        kwargs["start_frame_idx"] = start_frame
    if "reverse" in parameters:
        kwargs["reverse"] = reverse

    count = 0
    for frame_index, object_ids, mask_logits in predictor.propagate_in_video(**kwargs):
        save_output_masks(frame_index, object_ids, mask_logits, raw_root)
        count += 1
        if count % 10 == 0:
            print(f"{direction}: {count} frames; current={frame_index + 1}", flush=True)
    print(f"Finished {direction} propagation: {count} frames")


def colorize_mask(frame, mask, color, alpha_value=0.45):
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


def generate_preview_video(
    frame_files, names, object_ids, raw_root, preview_video, fps
):
    first = cv2.imread(str(frame_files[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frame_files[0]}")
    height, width = first.shape[:2]
    preview_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(preview_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {preview_video}")

    missing_counts = {object_id: 0 for object_id in object_ids}
    try:
        for index, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read {frame_path}")
            preview = frame
            for object_id in object_ids:
                mask_path = raw_root / f"object_{object_id}" / f"{index + 1:06d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    missing_counts[object_id] += 1
                    continue
                preview = colorize_mask(
                    preview, mask, COLORS.get(object_id, (255, 255, 255))
                )

            panel_height = 50 + 30 * len(object_ids)
            cv2.rectangle(preview, (12, 12), (780, panel_height), (0, 0, 0), -1)
            cv2.putText(
                preview,
                f"Frame {index + 1:06d}",
                (25, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for row, object_id in enumerate(object_ids):
                cv2.putText(
                    preview,
                    f"Object {object_id}: {names[object_id]}",
                    (25, 76 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    COLORS.get(object_id, (255, 255, 255)),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(preview)
    finally:
        writer.release()
    return missing_counts


def main():
    args = parse_args()
    frames_dir = args.frames.expanduser().resolve()
    annotations_file = args.annotations.expanduser().resolve()
    checkpoint = args.sam2_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw_root = output_dir / "raw_masks"
    preview_video = output_dir / "tracking_preview.mp4"
    summary_file = output_dir / "tracking_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    data, names, prompts, frame_files = load_annotations(
        frames_dir, args.frame_pattern, annotations_file
    )
    object_ids = sorted({prompt["object_id"] for prompt in prompts})
    frames_by_object = defaultdict(list)
    for prompt in prompts:
        frames_by_object[prompt["object_id"]].append(prompt["frame_index"])
        print(
            f"frame={prompt['frame_index'] + 1:06d} "
            f"object={prompt['object_id']} ({prompt['object_name']}) "
            f"positive={int(np.count_nonzero(prompt['labels'] == 1))} "
            f"negative={int(np.count_nonzero(prompt['labels'] == 0))}"
        )

    print("Loading SAM2 Video Predictor...")
    predictor = build_sam2_video_predictor(
        args.sam2_config, str(checkpoint), device=args.device
    )
    inference_state = initialize_state(predictor, frames_dir)
    predictor.reset_state(inference_state)
    reference_frame = min(prompt["frame_index"] for prompt in prompts)

    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    with torch.inference_mode(), autocast_context:
        for prompt in prompts:
            frame_index, output_ids, mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt["frame_index"],
                obj_id=prompt["object_id"],
                points=prompt["points"],
                labels=prompt["labels"],
            )
            save_output_masks(frame_index, output_ids, mask_logits, raw_root)
            print(
                f"Added object {prompt['object_id']} at frame {frame_index + 1:06d}"
            )

        propagate_direction(
            predictor, inference_state, reference_frame, False, raw_root
        )
        propagate_direction(
            predictor, inference_state, reference_frame, True, raw_root
        )

    missing_counts = generate_preview_video(
        frame_files, names, object_ids, raw_root, preview_video, args.fps
    )
    summary = {
        "video_id": data.get("video_id"),
        "frame_count": len(frame_files),
        "objects": {
            str(object_id): {
                "name": names[object_id],
                "conditioning_frame_indices": sorted(set(frames_by_object[object_id])),
                "missing_mask_count": missing_counts[object_id],
            }
            for object_id in object_ids
        },
        "raw_masks": str(raw_root),
        "preview_video": str(preview_video),
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Raw masks:", raw_root)
    print("Preview:", preview_video)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
