"""Geometry-safe image compositing for annotated object crops."""

from __future__ import annotations

from PIL import Image


def paste_scaled_object(
    background: Image.Image,
    object_crop: Image.Image,
    object_box: tuple[float, float, float, float],
    *,
    scale: float,
    top_left: tuple[int, int],
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Paste a resized crop and transform its xyxy annotation into background pixels.

    The crop may contain context around the annotated object. `object_box` refers
    to pixel coordinates in that crop. Pixels outside the background are rejected
    rather than silently clipped so that image and target geometry stay aligned.
    """
    if background.mode != "RGB" or object_crop.mode != "RGB":
        raise ValueError("background and object_crop must be RGB")
    if scale <= 0:
        raise ValueError("scale must be positive")
    x1, y1, x2, y2 = object_box
    if not (0 <= x1 < x2 <= object_crop.width and 0 <= y1 < y2 <= object_crop.height):
        raise ValueError("object_box must be a valid box inside object_crop")
    new_size = (max(1, round(object_crop.width * scale)), max(1, round(object_crop.height * scale)))
    resized = object_crop.resize(new_size, Image.Resampling.BILINEAR)
    left, top = top_left
    right, bottom = left + new_size[0], top + new_size[1]
    if left < 0 or top < 0 or right > background.width or bottom > background.height:
        raise ValueError("scaled crop must fit fully inside the background")
    result = background.copy()
    result.paste(resized, (left, top))
    scaled_box = (left + x1 * new_size[0] / object_crop.width,
                  top + y1 * new_size[1] / object_crop.height,
                  left + x2 * new_size[0] / object_crop.width,
                  top + y2 * new_size[1] / object_crop.height)
    return result, scaled_box
