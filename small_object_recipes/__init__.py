"""Dataset-agnostic small-object detection recipes."""

from .tiling import Detection, Slice, make_slices, tiled_predict

__all__ = ["Detection", "Slice", "make_slices", "tiled_predict"]
