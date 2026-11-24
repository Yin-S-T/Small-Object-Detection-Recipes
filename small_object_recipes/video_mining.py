"""Local frame-difference proposals for human review; proposals are not labels."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MotionRegion:
    xyxy: tuple[int, int, int, int]
    changed_pixels: int


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.uint8)


def motion_regions(
    previous: Image.Image,
    current: Image.Image,
    *,
    difference_threshold: int = 24,
    min_changed_pixels: int = 12,
    padding: int = 4,
) -> list[MotionRegion]:
    """Return connected changed regions between two equal-size RGB frames.

    This simple baseline is intentionally class-agnostic. Camera shake, exposure
    changes, and animated backgrounds can produce large or noisy proposals.
    """
    if previous.size != current.size:
        raise ValueError("frames must have the same dimensions")
    if not 0 <= difference_threshold <= 255 or min_changed_pixels < 1 or padding < 0:
        raise ValueError("invalid motion threshold, minimum area, or padding")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the optional video dependency: pip install opencv-python-headless") from exc
    diff = np.abs(_gray(current).astype(np.int16) - _gray(previous).astype(np.int16))
    mask = (diff >= difference_threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape
    regions: list[MotionRegion] = []
    # OpenCV connected components keep CPU work in optimized native code.
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if area >= min_changed_pixels:
            box = (max(0, x - padding), max(0, y - padding),
                   min(width, x + component_width + padding), min(height, y + component_height + padding))
            regions.append(MotionRegion(box, area))
    return sorted(regions, key=lambda region: (region.xyxy[1], region.xyxy[0]))


def mine_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    every_n_frames: int = 1,
    difference_threshold: int = 24,
    min_changed_pixels: int = 12,
    padding: int = 4,
) -> int:
    """Save crops around changed regions from a local video for later review."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install the optional video dependency: pip install opencv-python-headless") from exc
    source = Path(video_path)
    destination = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be at least 1")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("output directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("could not open the supplied video")
    rows: list[dict[str, int | str]] = []
    previous: Image.Image | None = None
    frame_index = -1
    saved = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % every_n_frames:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame_rgb)
            if previous is not None:
                for region_index, region in enumerate(motion_regions(
                    previous, frame,
                    difference_threshold=difference_threshold,
                    min_changed_pixels=min_changed_pixels,
                    padding=padding,
                )):
                    crop = frame.crop(region.xyxy)
                    name = f"frame_{frame_index:08d}_region_{region_index:04d}.png"
                    crop.save(destination / name)
                    x1, y1, x2, y2 = region.xyxy
                    rows.append({"file": name, "frame_index": frame_index, "region_index": region_index,
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "changed_pixels": region.changed_pixels, "review_status": "unreviewed"})
                    saved += 1
            previous = frame
    finally:
        capture.release()
    with (destination / "review_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ("file", "frame_index", "region_index", "x1", "y1", "x2", "y2", "changed_pixels", "review_status")
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="local video file supplied by the user")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--every-n-frames", type=int, default=1)
    parser.add_argument("--difference-threshold", type=int, default=24)
    parser.add_argument("--min-changed-pixels", type=int, default=12)
    parser.add_argument("--padding", type=int, default=4)
    args = parser.parse_args()
    count = mine_video(args.video, args.output_dir, every_n_frames=args.every_n_frames,
                       difference_threshold=args.difference_threshold,
                       min_changed_pixels=args.min_changed_pixels, padding=args.padding)
    print(f"Saved {count} motion proposal crops for manual review.")


if __name__ == "__main__":
    main()
