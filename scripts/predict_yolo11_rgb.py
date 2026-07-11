from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
if str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO11 inference on rgb images.")
    parser.add_argument("--weights", type=Path, default=ROOT / "runs" / "detect" / "yolo11_rgb" / "weights" / "best.pt")
    parser.add_argument("--source", type=Path, default=ROOT / "dataset" / "images" / "rgb")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "predict")
    parser.add_argument("--name", type=str, default="yolo11_rgb")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"找不到权重文件: {args.weights}")
    if not args.source.exists():
        raise FileNotFoundError(f"找不到推理输入: {args.source}")

    model = YOLO(str(args.weights))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )


if __name__ == "__main__":
    main()
