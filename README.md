# Small Object Detection Recipes

Small, reusable building blocks for small-object detection experiments. The repository focuses on dataset-agnostic examples for overlapping-window inference, motion-based video review, image-level data synthesis, and parameter-efficient vision-model adaptation.

The examples operate only on files supplied by the user. No dataset, video, trained checkpoint, or prediction artifact is included. Motion regions are unlabeled proposals; motion alone does not identify an object class and must not be treated as ground truth.

## Scope

- Generate overlapping image windows, map local detections back to the source image, and merge duplicate boxes.
- Sample a local video file and export crops around changed regions for human review.
- Place a resized, annotated object crop into a background image while updating its box coordinates.
- Build a small classifier head over a caller-provided DINOv3-compatible backbone, with optional PEFT/LoRA wrapping.

This is an educational recipe collection, not a complete detector, tracking system, or production video service. The tiling example is a fixed overlapping grid. The video example uses frame differences to propose regions and is sensitive to camera motion, lighting changes, and moving backgrounds.
The image-synthesis example uses direct pixel compositing without blending or physical-scene simulation.

## Install

```bash
python -m pip install -e ".[video,test]"
```

PyTorch and PEFT are optional and are needed only for the fine-tuning example. Install versions appropriate for your platform and review the licenses for each model and dependency before use. This repository does not distribute DINOv3 source code or weights.

## Quick example: tiled inference

```python
from PIL import Image
from small_object_recipes.tiling import Detection, tiled_predict

def predict_one(crop):
    # Replace with your detector. Coordinates are xyxy in this crop's pixels.
    return [Detection((12, 16, 30, 38), 0.91, class_id=0)]

image = Image.new("RGB", (1280, 720), "white")
result = tiled_predict(image, predict_one, tile_size=512, overlap=0.2)
for detection in result:
    print(detection.xyxy, detection.score, detection.class_id)
```

## Quick example: motion proposals from a video

```bash
small-object-video-mine input.mp4 --output-dir review_crops --every-n-frames 3
```

The output contains image crops and a CSV manifest marked `unreviewed`. Review and label candidates before using them for training. The tool processes the local video supplied on the command line and does not upload it.

## Fine-tuning notes

`DinoV3BinaryClassifier` accepts a backbone object supplied by the caller. The caller remains responsible for selecting a compatible model revision, image processor, input resolution, license, and target modules. The example intentionally does not download model files or include pretrained or fine-tuned weights.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License status

No license is included yet. Add a license only after confirming that the repository owner has the right to license every authored file and that included dependencies and model interfaces are compatible with the intended distribution.
