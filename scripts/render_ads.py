#!/usr/bin/env python3
"""Render advertisement textures into fitted billboard quadrilaterals."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MIN_REPEATS = 1
MAX_REPEATS = 40


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perspective-warp advertisements into fitted billboard planes."
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument(
        "--clean-masks",
        type=Path,
        required=True,
        help="Directory containing object_1, object_2, ...",
    )
    parser.add_argument(
        "--advertisement", type=Path, required=True, help="Default ad image"
    )
    parser.add_argument(
        "--ad-map",
        type=Path,
        default=None,
        help='Optional JSON such as {"1":"far.png","2":"goal.png"}',
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rendering")
    )
    parser.add_argument("--frame-pattern", default="*.jpg")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--opacity", type=float, default=0.96)
    parser.add_argument("--tile-height", type=int, default=256)
    return parser.parse_args()


def load_advertisement(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Unsupported advertisement shape: {image.shape}")
    if image.shape[2] == 4:
        return image[:, :, :3], image[:, :, 3]
    return image, np.full(image.shape[:2], 255, dtype=np.uint8)


def load_ad_paths(default_path, ad_map_path, object_ids):
    paths = {object_id: default_path for object_id in object_ids}
    if ad_map_path is None:
        return paths
    if not ad_map_path.exists():
        raise FileNotFoundError(ad_map_path)
    mapping = json.loads(ad_map_path.read_text(encoding="utf-8"))
    for key, value in mapping.items():
        object_id = int(key)
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (ad_map_path.parent / path).resolve()
        paths[object_id] = path
    return paths


def quad_dimensions(quad):
    quad = np.asarray(quad, dtype=np.float64)
    length = 0.5 * (
        np.linalg.norm(quad[1] - quad[0])
        + np.linalg.norm(quad[2] - quad[3])
    )
    height = 0.5 * (
        np.linalg.norm(quad[3] - quad[0])
        + np.linalg.norm(quad[2] - quad[1])
    )
    return float(length), float(height)


def choose_repeat_count(records, ad_ratio):
    board_ratios = []
    for record in records:
        quad = record.get("quad")
        if quad is None:
            continue
        length, height = quad_dimensions(quad)
        if length > 1.0 and height > 1.0:
            board_ratios.append(length / height)
    if not board_ratios:
        return None, None
    median_ratio = float(np.median(board_ratios))
    repeat_count = int(
        np.clip(round(median_ratio / ad_ratio), MIN_REPEATS, MAX_REPEATS)
    )
    return max(1, repeat_count), median_ratio


def build_repeated_strip(ad_bgr, ad_alpha, repeat_count, tile_height):
    ad_height, ad_width = ad_bgr.shape[:2]
    tile_width = max(1, int(round(tile_height * ad_width / ad_height)))
    tile_bgr = cv2.resize(
        ad_bgr, (tile_width, tile_height), interpolation=cv2.INTER_AREA
    )
    tile_alpha = cv2.resize(
        ad_alpha, (tile_width, tile_height), interpolation=cv2.INTER_AREA
    )
    return (
        np.tile(tile_bgr, (1, repeat_count, 1)),
        np.tile(tile_alpha, (1, repeat_count)),
    )


def warp_strip(strip_bgr, strip_alpha, quad, output_width, output_height):
    source_height, source_width = strip_bgr.shape[:2]
    source = np.float32(
        [
            [0, 0],
            [source_width - 1, 0],
            [source_width - 1, source_height - 1],
            [0, source_height - 1],
        ]
    )
    destination = np.asarray(quad, dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)
    warped_bgr = cv2.warpPerspective(
        strip_bgr,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_alpha = cv2.warpPerspective(
        strip_alpha,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_bgr, warped_alpha


def composite_billboard(
    frame, warped_bgr, warped_alpha, clean_mask, opacity
):
    if clean_mask.shape != frame.shape[:2]:
        clean_mask = cv2.resize(
            clean_mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    styled = cv2.addWeighted(warped_bgr, 0.93, frame, 0.07, 0.0)
    alpha = (
        warped_alpha.astype(np.float32)
        / 255.0
        * (clean_mask.astype(np.float32) / 255.0)
        * opacity
    )[..., None]
    return (
        styled.astype(np.float32) * alpha
        + frame.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)


def main():
    args = parse_args()
    frames_dir = args.frames.expanduser().resolve()
    geometry_file = args.geometry.expanduser().resolve()
    clean_root = args.clean_masks.expanduser().resolve()
    default_ad = args.advertisement.expanduser().resolve()
    ad_map = args.ad_map.expanduser().resolve() if args.ad_map else None
    output_dir = args.output_dir.expanduser().resolve()
    output_frames = output_dir / "replaced_frames"
    output_video = output_dir / "replacement_no_occlusion.mp4"
    summary_file = output_dir / "rendering_summary.json"

    if not geometry_file.exists():
        raise FileNotFoundError(geometry_file)
    if not clean_root.is_dir():
        raise NotADirectoryError(clean_root)
    if not 0.0 <= args.opacity <= 1.0:
        raise ValueError("--opacity must be between 0 and 1")
    if args.tile_height < 8:
        raise ValueError("--tile-height must be at least 8")

    geometry = json.loads(geometry_file.read_text(encoding="utf-8"))
    objects = geometry["objects"]
    object_ids = sorted(int(object_id) for object_id in objects)
    frame_files = sorted(frames_dir.glob(args.frame_pattern))
    if not frame_files:
        raise RuntimeError(f"No frames matched {frames_dir / args.frame_pattern}")

    ad_paths = load_ad_paths(default_ad, ad_map, object_ids)
    strips, repeat_counts, median_ratios, ad_sizes = {}, {}, {}, {}
    for object_id in object_ids:
        ad_bgr, ad_alpha = load_advertisement(ad_paths[object_id])
        ad_ratio = ad_bgr.shape[1] / ad_bgr.shape[0]
        repeat_count, median_ratio = choose_repeat_count(
            objects[str(object_id)], ad_ratio
        )
        if repeat_count is None:
            continue
        strips[object_id] = build_repeated_strip(
            ad_bgr, ad_alpha, repeat_count, args.tile_height
        )
        repeat_counts[object_id] = repeat_count
        median_ratios[object_id] = median_ratio
        ad_sizes[object_id] = [int(ad_bgr.shape[1]), int(ad_bgr.shape[0])]
        print(
            f"object {object_id}: board_ratio={median_ratio:.2f}, "
            f"ad={ad_paths[object_id]}, repeats={repeat_count}"
        )
    if not strips:
        raise RuntimeError("No valid billboard geometry")

    record_maps = {
        object_id: {
            int(record["frame_index"]): record
            for record in objects[str(object_id)]
        }
        for object_id in object_ids
    }
    first = cv2.imread(str(frame_files[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frame_files[0]}")
    height, width = first.shape[:2]
    output_frames.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_video}")

    rendered_counts = {object_id: 0 for object_id in object_ids}
    try:
        for frame_index, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read {frame_path}")
            result = frame.copy()
            for object_id in object_ids:
                if object_id not in strips:
                    continue
                record = record_maps[object_id].get(frame_index)
                quad = None if record is None else record.get("quad")
                if quad is None:
                    continue
                clean_path = (
                    clean_root
                    / f"object_{object_id}"
                    / f"{frame_index + 1:06d}.png"
                )
                clean_mask = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
                if clean_mask is None or not np.any(clean_mask):
                    continue
                strip_bgr, strip_alpha = strips[object_id]
                warped_bgr, warped_alpha = warp_strip(
                    strip_bgr, strip_alpha, quad, width, height
                )
                result = composite_billboard(
                    result, warped_bgr, warped_alpha, clean_mask, args.opacity
                )
                rendered_counts[object_id] += 1

            output_frame = output_frames / f"{frame_index + 1:06d}.jpg"
            if not cv2.imwrite(
                str(output_frame), result, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Could not write {output_frame}")
            writer.write(result)
            if (frame_index + 1) % 20 == 0:
                print(f"rendered {frame_index + 1}/{len(frame_files)}", flush=True)
    finally:
        writer.release()

    summary = {
        "video_id": geometry.get("video_id"),
        "output_video": str(output_video),
        "output_frames": str(output_frames),
        "objects": {
            str(object_id): {
                "advertisement": str(ad_paths[object_id]),
                "advertisement_size": ad_sizes.get(object_id),
                "repeat_count": repeat_counts.get(object_id),
                "median_board_ratio": median_ratios.get(object_id),
                "rendered_frames": rendered_counts[object_id],
            }
            for object_id in object_ids
        },
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Replaced frames:", output_frames)
    print("Preview:", output_video)
    print("Summary:", summary_file)


if __name__ == "__main__":
    main()
