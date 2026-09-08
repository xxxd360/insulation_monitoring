from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reviewable RGB/IR pair manifest by sorted index.")
    parser.add_argument("--rgb-dir", type=Path, default=Path("dataset/images/rgb"))
    parser.add_argument("--ir-dir", type=Path, default=Path("dataset/images/ir"))
    parser.add_argument("--output", type=Path, default=Path("dataset/pairs.csv"))
    args = parser.parse_args()
    rgb = sorted(p for p in args.rgb_dir.iterdir() if p.is_file() and p.suffix.casefold() in EXTS)
    ir = sorted(p for p in args.ir_dir.iterdir() if p.is_file() and p.suffix.casefold() in EXTS)
    count = min(len(rgb), len(ir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rgb", "ir"])
        writer.writeheader()
        for rgb_path, ir_path in zip(rgb[:count], ir[:count]):
            writer.writerow({"rgb": rgb_path.name, "ir": ir_path.name})
    print(f"已生成 {count} 对候选配对: {args.output}")
    if len(rgb) != len(ir):
        print("警告：RGB/IR 数量不同，CSV 只是按排序索引生成的候选清单，请人工复核后再用于正式推理。")


if __name__ == "__main__":
    main()
