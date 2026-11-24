"""Fixed overlapping-window inference and source-coordinate box merging."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from PIL import Image


@dataclass(frozen=True)
class Slice:
    index: int
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One detection with xyxy coordinates in the current image's pixel space."""

    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int = 0
    source_slice: int | None = None


def _axis_starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= 0 or tile <= 0:
        raise ValueError("image length and tile size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1 - overlap)))
    final_start = length - tile
    return sorted(set(range(0, final_start + 1, step)) | {final_start})


def make_slices(width: int, height: int, tile_size: int, overlap: float = 0.2) -> list[Slice]:
    """Create a row-major grid; the final windows align with the far image edges."""
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("image dimensions and tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    xs = _axis_starts(width, tile_size, overlap)
    ys = _axis_starts(height, tile_size, overlap)
    result: list[Slice] = []
    for top in ys:
        for left in xs:
            result.append(Slice(len(result), (left, top, min(left + tile_size, width), min(top + tile_size, height))))
    return result


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def merge_detections(detections: Iterable[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Class-aware greedy NMS with stable score ordering."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    items = list(detections)
    for item in items:
        if not math.isfinite(item.score) or not all(math.isfinite(v) for v in item.xyxy):
            raise ValueError("detection coordinates and score must be finite")
        x1, y1, x2, y2 = item.xyxy
        if x1 >= x2 or y1 >= y2:
            raise ValueError("detection boxes must have positive width and height")
    order = sorted(range(len(items)), key=lambda i: (-items[i].score, i))
    kept: list[Detection] = []
    while order:
        current_index = order.pop(0)
        current = items[current_index]
        kept.append(current)
        order = [i for i in order if items[i].class_id != current.class_id
                 or _iou(current.xyxy, items[i].xyxy) <= iou_threshold]
    return kept


def tiled_predict(
    image: Image.Image,
    predictor: Callable[[Image.Image], Iterable[Detection]],
    *,
    tile_size: int,
    overlap: float = 0.2,
    include_full_image: bool = True,
    merge_iou: float = 0.5,
) -> list[Detection]:
    """Run a caller-provided detector on windows and return source-image boxes."""
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image must have nonzero dimensions")
    windows = make_slices(image.width, image.height, tile_size, overlap)
    if include_full_image and (image.width > tile_size or image.height > tile_size):
        windows = [Slice(-1, (0, 0, image.width, image.height)), *windows]

    mapped: list[Detection] = []
    for window in windows:
        x0, y0, x1, y1 = window.xyxy
        crop = image.crop(window.xyxy)
        for item in predictor(crop):
            bx0, by0, bx1, by1 = item.xyxy
            if not all(math.isfinite(v) for v in item.xyxy + (item.score,)):
                raise ValueError("predictor returned a nonfinite detection")
            box = (max(0.0, min(float(image.width), x0 + bx0)),
                   max(0.0, min(float(image.height), y0 + by0)),
                   max(0.0, min(float(image.width), x0 + bx1)),
                   max(0.0, min(float(image.height), y0 + by1)))
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            mapped.append(Detection(box, item.score, item.class_id, window.index))
    return merge_detections(mapped, merge_iou)
