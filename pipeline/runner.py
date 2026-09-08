from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .config import PipelineConfig
from .detection import YOLODetector, crop_detection, exclude_classes, filter_classes
from .fusion_engine import DenseFuseEngine
from .risk import assess_risk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.casefold() in IMAGE_EXTS)


def _pair_inputs(rgb_dir: Path, ir_dir: Path, pair_manifest: Path | None = None, allow_index_pairing: bool = False) -> list[tuple[Path, Path]]:
    if pair_manifest is not None:
        pairs: list[tuple[Path, Path]] = []
        with pair_manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                rgb_value, ir_value = (row.get("rgb") or "").strip(), (row.get("ir") or "").strip()
                if not rgb_value or not ir_value:
                    continue
                rgb_path = Path(rgb_value)
                ir_path = Path(ir_value)
                if not rgb_path.is_absolute():
                    rgb_path = rgb_dir / rgb_path
                if not ir_path.is_absolute():
                    ir_path = ir_dir / ir_path
                if not rgb_path.exists() or not ir_path.exists():
                    raise FileNotFoundError(f"配对清单中的文件不存在: rgb={rgb_path}, ir={ir_path}")
                pairs.append((rgb_path, ir_path))
        if not pairs:
            raise ValueError(f"配对清单没有有效记录: {pair_manifest}")
        return pairs
    rgb_paths = _image_files(rgb_dir)
    ir_paths = _image_files(ir_dir)
    if len(rgb_paths) != len(ir_paths):
        if not allow_index_pairing:
            raise ValueError(f"RGB/IR 数量不一致: rgb={len(rgb_paths)}, ir={len(ir_paths)}。请先建立配对清单，或显式使用 --allow-index-pairing。")
        count = min(len(rgb_paths), len(ir_paths))
        print(f"[warning] RGB/IR 数量不一致，按排序索引取前 {count} 对；这只适合临时验证。")
        return list(zip(rgb_paths[:count], ir_paths[:count]))
    return list(zip(rgb_paths, ir_paths))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> list[dict]:
    dirs = config.make_output_dirs()
    pairs = _pair_inputs(config.rgb_dir, config.ir_dir, config.pair_manifest, config.allow_index_pairing)
    fusion = DenseFuseEngine(config.fusion_weights, config.image_size, config.device) if config.use_fusion else None
    stage1 = YOLODetector(config.stage1_weights, config.detector_imgsz, config.confidence, config.iou, config.device)
    stage2 = YOLODetector(config.stage2_weights, config.detector_imgsz, config.confidence, config.iou, config.device)
    records: list[dict] = []

    for index, (rgb_path, ir_path) in enumerate(pairs, start=1):
        sample_id = f"{index:04d}_{rgb_path.stem}"
        rgb = Image.open(rgb_path).convert("RGB")
        if fusion is not None:
            fused = fusion.fuse(rgb_path, ir_path)
            detector_image = fusion.as_detector_image(fused)
            analysis_image = fused.convert("RGB")
            fused_path = dirs["fused"] / f"{sample_id}_fusion.png"
            fused.save(fused_path)
        else:
            detector_image = rgb
            analysis_image = rgb
            fused_path = None

        stage1_dets, stage1_plot = stage1.predict(detector_image)
        stage1_dets = filter_classes(stage1_dets, config.stage1_classes)
        stage1_plot.save(dirs["stage1"] / f"{sample_id}_stage1.jpg")
        sample_record = {
            "sample_id": sample_id,
            "rgb": str(rgb_path),
            "ir": str(ir_path),
            "fused": str(fused_path) if fused_path else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "insulators": [],
        }

        for ins_index, insulator in enumerate(stage1_dets, start=1):
            # Crop the same image domain used by stage 1. This keeps the
            # second detector aligned with the first detector's coordinates.
            crop = crop_detection(analysis_image, insulator)
            crop_id = f"{sample_id}_insulator_{ins_index:02d}"
            crop_path = dirs["insulator_crops"] / f"{crop_id}.jpg"
            crop.save(crop_path)
            stage2_dets, stage2_plot = stage2.predict(crop)
            stage2_dets = exclude_classes(stage2_dets, config.stage1_classes)
            stage2_plot.save(dirs["stage2"] / f"{crop_id}_stage2.jpg")
            crop_area = float(crop.width * crop.height)
            assessment = assess_risk(stage2_dets, crop_area, config.risk)
            sample_record["insulators"].append({
                "bbox": list(insulator.xyxy),
                "class": insulator.class_name,
                "confidence": insulator.confidence,
                "crop": str(crop_path),
                "defects": [
                    {"class": d.class_name, "confidence": d.confidence, "bbox": list(d.xyxy), "area": d.area}
                    for d in stage2_dets
                ],
                "risk": assessment.to_dict(),
            })

        record_path = dirs["records"] / f"{sample_id}.json"
        _save_json(record_path, sample_record)
        records.append(sample_record)
        print(f"[{index}/{len(pairs)}] {sample_id}: 绝缘子={len(stage1_dets)}")

    _save_json(config.output_dir / "summary.json", {"config": {k: str(v) for k, v in vars(config).items() if k != "risk"}, "records": records})
    return records
