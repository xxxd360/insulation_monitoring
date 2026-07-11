from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
if str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class LabelMap:
    def __init__(self) -> None:
        self.name_to_id: dict[str, int] = {}

    def get_id(self, name: str) -> int:
        if name not in self.name_to_id:
            self.name_to_id[name] = len(self.name_to_id)
        return self.name_to_id[name]

    @property
    def names(self) -> dict[int, str]:
        return {idx: name for name, idx in self.name_to_id.items()}


def labelme_to_yolo_lines(json_path: Path, label_map: LabelMap) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_w = float(data["imageWidth"])
    img_h = float(data["imageHeight"])

    lines: list[str] = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue

        cls_id = label_map.get_id(str(shape["label"]))
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]

        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(img_w, max(xs))
        y2 = min(img_h, max(ys))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (x1 + x2) / 2.0 / img_w
        y_center = (y1 + y2) / 2.0 / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return lines


def prepare_dataset(rgb_dir: Path, label_dir: Path, out_dir: Path, val_ratio: float, seed: int) -> Path:
    images = sorted([p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise FileNotFoundError(f"没有找到图片: {rgb_dir}")

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []
    for img_path in images:
        json_path = label_dir / f"{img_path.stem}.json"
        if json_path.exists():
            pairs.append((img_path, json_path))
        else:
            missing_labels.append(img_path.name)


    random.seed(seed)
    random.shuffle(pairs)

    val_count = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    label_map = LabelMap()
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for img_path, json_path in split_pairs:
            dst_img = out_dir / "images" / split / img_path.name
            dst_label = out_dir / "labels" / split / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dst_img)
            lines = labelme_to_yolo_lines(json_path, label_map)
            dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = out_dir / "data.yaml"
    data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "names": label_map.names,
    }
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"训练集: {len(train_pairs)} 张")
    print(f"验证集: {len(val_pairs)} 张")
    print(f"类别: {label_map.names}")
    print(f"YOLO 数据集已生成: {out_dir}")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO11 on dataset/images/rgb with LabelMe json labels.")
    parser.add_argument("--rgb-dir", type=Path, default=ROOT / "dataset" / "images" / "rgb")
    parser.add_argument("--label-dir", type=Path, default=ROOT / "dataset" / "labels")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dataset" / "yolo_rgb")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    parser.add_argument("--name", type=str, default="yolo11_rgb")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    data_yaml = prepare_dataset(args.rgb_dir, args.label_dir, args.out_dir, args.val_ratio, args.seed)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
