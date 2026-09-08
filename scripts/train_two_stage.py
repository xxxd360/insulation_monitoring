from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
if str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO


def train_one(data: Path, model_name: str, output: Path, name: str, epochs: int, imgsz: int, batch: int, device: str | None) -> None:
    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output),
        name=name,
        workers=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the independent stage-1 and stage-2 YOLO models.")
    parser.add_argument("--stage1-data", type=Path, default=ROOT / "dataset/two_stage/stage1/data.yaml")
    parser.add_argument("--stage2-data", type=Path, default=ROOT / "dataset/two_stage/stage2/data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/detect")
    args = parser.parse_args()
    for data in (args.stage1_data, args.stage2_data):
        if not data.exists():
            raise FileNotFoundError(f"找不到数据配置: {data}，请先运行 prepare_two_stage_dataset.py")
    train_one(args.stage1_data, args.model, args.output, "stage1_insulator", args.epochs, args.imgsz, args.batch, args.device)
    train_one(args.stage2_data, args.model, args.output, "stage2_defect", args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
