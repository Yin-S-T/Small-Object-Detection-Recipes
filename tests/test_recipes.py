import unittest
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from small_object_recipes.data_synthesis import paste_scaled_object
from small_object_recipes.tiling import Detection, make_slices, tiled_predict
from small_object_recipes.video_mining import mine_video, motion_regions

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = object


class TilingTests(unittest.TestCase):
    def test_edge_aligned_overlapping_grid(self):
        windows = make_slices(1000, 700, 400, 0.25)
        self.assertEqual(windows[0].xyxy, (0, 0, 400, 400))
        self.assertEqual(windows[-1].xyxy[2:], (1000, 700))
        self.assertGreater(len(windows), 1)

    def test_local_boxes_are_projected_and_duplicate_boxes_merged(self):
        image = Image.new("RGB", (800, 400), "black")
        calls = 0

        def predictor(crop):
            nonlocal calls
            call = calls
            calls += 1
            if call == 0:  # full image
                return [Detection((350, 100, 370, 120), .7)]
            if call == 1:  # first tile at x=0
                return [Detection((350, 100, 370, 120), .8)]
            if call == 2:  # overlapping tile at x=300
                return [Detection((50, 100, 70, 120), .9)]
            return []

        detections = tiled_predict(image, predictor, tile_size=400, overlap=.25)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].xyxy, (350, 100, 370, 120))
        self.assertEqual(detections[0].score, .9)


class SynthesisTests(unittest.TestCase):
    def test_object_box_tracks_resize_and_placement(self):
        background = Image.new("RGB", (64, 64), "black")
        crop = Image.new("RGB", (20, 10), "white")
        result, box = paste_scaled_object(background, crop, (5, 2, 15, 8), scale=.5, top_left=(10, 20))
        self.assertEqual(result.size, (64, 64))
        self.assertEqual(box, (12.5, 21.0, 17.5, 24.0))


class VideoMiningTests(unittest.TestCase):
    def test_changed_patch_is_proposed_but_not_classified(self):
        previous = Image.new("RGB", (40, 30), "black")
        current = previous.copy()
        ImageDraw.Draw(current).rectangle((10, 8, 14, 12), fill="white")
        regions = motion_regions(previous, current, difference_threshold=20, min_changed_pixels=4, padding=2)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].xyxy, (8, 6, 17, 15))
        self.assertEqual(regions[0].changed_pixels, 25)

    def test_local_video_pipeline_exports_only_unreviewed_crops(self):
        first = np.zeros((24, 32, 3), dtype=np.uint8)
        second = first.copy()
        second[8:13, 10:15] = 255

        class FakeCapture:
            def __init__(self, _path):
                self.frames = iter((first, second))

            def isOpened(self):
                return True

            def read(self):
                frame = next(self.frames, None)
                return (frame is not None, frame)

            def release(self):
                pass

        class FakeCV2:
            COLOR_BGR2RGB = 1

            @staticmethod
            def VideoCapture(path):
                return FakeCapture(path)

            @staticmethod
            def cvtColor(frame, _code):
                return frame[:, :, ::-1]

            @staticmethod
            def connectedComponentsWithStats(mask, connectivity=8):
                ys, xs = np.nonzero(mask)
                height, width = mask.shape
                if not len(xs):
                    return 1, np.zeros(mask.shape, dtype=np.int32), np.array([[0, 0, width, height, 0]]), np.zeros((1, 2))
                component = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                             int(ys.max() - ys.min() + 1), int(len(xs))]
                stats = np.array([[0, 0, width, height, width * height], component], dtype=np.int32)
                return 2, np.zeros(mask.shape, dtype=np.int32), stats, np.zeros((2, 2))

        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.video"
            source.write_bytes(b"placeholder")
            output = root / "crops"
            with patch.dict("sys.modules", {"cv2": FakeCV2}):
                saved = mine_video(source, output, min_changed_pixels=4, padding=1)
            self.assertEqual(saved, 1)
            files = list(output.iterdir())
            self.assertEqual(len(files), 2)
            crop = next(path for path in files if path.suffix == ".png")
            with Image.open(crop) as image:
                self.assertEqual(image.size, (7, 7))
            with (output / "review_manifest.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["review_status"], "unreviewed")
            self.assertNotIn(str(source), (output / "review_manifest.csv").read_text(encoding="utf-8"))


@unittest.skipIf(torch is None, "PyTorch is optional")
class DinoAdapterTests(unittest.TestCase):
    def test_classifier_uses_cls_and_patch_tokens_after_register_tokens(self):
        from small_object_recipes.dinov3_adapter import DinoV3BinaryClassifier

        class ToyBackbone(nn.Module):
            def forward(self, pixel_values):
                batch = pixel_values.shape[0]
                tokens = torch.ones((batch, 1 + 2 + 4, 8), dtype=pixel_values.dtype)
                tokens[:, 0] = 2
                return SimpleNamespace(last_hidden_state=tokens)

        model = DinoV3BinaryClassifier(ToyBackbone(), hidden_size=8, num_register_tokens=2)
        logits = model(torch.zeros((3, 3, 32, 32)))
        self.assertEqual(tuple(logits.shape), (3, 2))


if __name__ == "__main__":
    unittest.main()
