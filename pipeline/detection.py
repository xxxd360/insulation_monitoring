from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class YOLODetector:
    def __init__(self, weights: Path, imgsz: int, conf: float, iou: float, device: str | None = None):
        import sys

        root = Path(__file__).resolve().parents[1]
        local_ultralytics = root / "ultralytics"
        if str(local_ultralytics) not in sys.path:
            sys.path.insert(0, str(local_ultralytics))
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

    def predict(self, image: Image.Image | np.ndarray) -> tuple[list[Detection], Image.Image]:
        source = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else image
        results = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            conf = boxes.conf.detach().cpu().numpy()
            for box, class_id, score in zip(xyxy, cls, conf):
                detections.append(Detection(class_id, str(names[int(class_id)]), float(score), tuple(map(float, box))))
        plotted = result.plot()[:, :, ::-1]
        return detections, Image.fromarray(plotted)


def filter_classes(detections: Iterable[Detection], class_names: Iterable[str]) -> list[Detection]:
    accepted = {name.casefold() for name in class_names}
    return [d for d in detections if d.class_name.casefold() in accepted]


def exclude_classes(detections: Iterable[Detection], class_names: Iterable[str]) -> list[Detection]:
    excluded = {name.casefold() for name in class_names}
    return [d for d in detections if d.class_name.casefold() not in excluded]


def crop_detection(image: Image.Image, detection: Detection, padding: float = 0.05) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = detection.xyxy
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - box_w * padding))
    y1 = max(0, int(y1 - box_h * padding))
    x2 = min(width, int(x2 + box_w * padding))
    y2 = min(height, int(y2 + box_h * padding))
    return image.crop((x1, y1, x2, y2))
