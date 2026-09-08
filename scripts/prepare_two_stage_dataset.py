from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Box:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


def read_boxes(path: Path) -> tuple[int, int, list[Box]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = int(data["imageWidth"]), int(data["imageHeight"])
    boxes: list[Box] = []
    for shape in data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
        x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
        if x2 > x1 and y2 > y1:
            boxes.append(Box(str(shape["label"]), x1, y1, x2, y2))
    return width, height, boxes


def yolo_line(box: Box, width: float, height: float, class_id: int) -> str:
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    bw = (box.x2 - box.x1) / width
    bh = (box.y2 - box.y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


def prepare(image_dir: Path, label_dir: Path, out_dir: Path, val_ratio: float, seed: int) -> None:
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTS)
    records = []
    for image in images:
        source_stem = image.stem.removesuffix("_fusion")
        label = label_dir / f"{source_stem}.json"
        if label.exists():
            records.append((image, label))
    if not records:
        raise FileNotFoundError("没有找到带 LabelMe 标注的 RGB 图片")
    random.Random(seed).shuffle(records)
    split_at = max(1, int(len(records) * val_ratio)) if len(records) > 1 else 0
    splits = {"val": records[:split_at], "train": records[split_at:]}

    stage1_names = {"insulation", "insulator", "绝缘子", "绝缘部件"}
    stage2_names: dict[str, int] = {}
    stage2_records: dict[str, list[tuple[Path, list[str]]]] = {"train": [], "val": []}
    for split, items in splits.items():
        (out_dir / "stage1" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "stage1" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "stage2" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "stage2" / "labels" / split).mkdir(parents=True, exist_ok=True)

        for image_path, json_path in items:
            width, height, boxes = read_boxes(json_path)
            shutil.copy2(image_path, out_dir / "stage1" / "images" / split / image_path.name)
            stage1_lines = [yolo_line(b, width, height, 0) for b in boxes if b.label.casefold() in stage1_names]
            (out_dir / "stage1" / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(stage1_lines) + ("\n" if stage1_lines else ""), encoding="utf-8")

            image = Image.open(image_path).convert("RGB")
            insulators = [b for b in boxes if b.label.casefold() in stage1_names]
            defects = [b for b in boxes if b.label.casefold() not in stage1_names]
            for index, insulator in enumerate(insulators, start=1):
                crop = Box("crop", insulator.x1, insulator.y1, insulator.x2, insulator.y2)
                crop_img = image.crop((int(crop.x1), int(crop.y1), int(crop.x2), int(crop.y2)))
                crop_name = f"{image_path.stem}_insulator_{index:02d}.jpg"
                crop_img.save(out_dir / "stage2" / "images" / split / crop_name, quality=95)
                crop_w, crop_h = crop_img.size
                lines: list[str] = []
                for defect in defects:
                    if intersection(defect, crop) <= 0:
                        continue
                    label = defect.label
                    if label not in stage2_names:
                        stage2_names[label] = len(stage2_names)
                    local = Box(label, defect.x1 - crop.x1, defect.y1 - crop.y1, defect.x2 - crop.x1, defect.y2 - crop.y1)
                    local = Box(label, max(0, local.x1), max(0, local.y1), min(crop_w, local.x2), min(crop_h, local.y2))
                    if local.x2 > local.x1 and local.y2 > local.y1:
                        lines.append(yolo_line(local, crop_w, crop_h, stage2_names[label]))
                (out_dir / "stage2" / "labels" / split / f"{Path(crop_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_yaml(root: Path, names: dict[int, str], filename: str) -> None:
        lines = [
            f"path: {str(root).replace(chr(92), '/')}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        lines.extend(f"  {index}: {name}" for index, name in names.items())
        (root / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_yaml(out_dir / "stage1", {0: "insulation"}, "data.yaml")
    write_yaml(out_dir / "stage2", {v: k for k, v in stage2_names.items()}, "data.yaml")
    print(f"stage1 数据集: {out_dir / 'stage1'}")
    print(f"stage2 数据集: {out_dir / 'stage2'}")
    print(f"stage2 类别: {stage2_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent YOLO datasets for the two-stage detector.")
    parser.add_argument("--image-dir", "--rgb-dir", dest="image_dir", type=Path, default=Path("dataset/images/after_fusion"))
    parser.add_argument("--label-dir", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/two_stage"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.image_dir, args.label_dir, args.out_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
