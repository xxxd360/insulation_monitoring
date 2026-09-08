from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RiskConfig:
    """Heuristic risk thresholds used until calibrated labels are available."""

    # Relative defect area inside the detected insulation crop.
    attention_area: float = 0.02
    warning_area: float = 0.08
    high_risk_area: float = 0.20
    # Class names that are treated as intrinsically more serious.
    high_risk_classes: set[str] = field(default_factory=lambda: {"defect", "peel", "burn", "flashover"})
    warning_classes: set[str] = field(default_factory=lambda: {"dirty", "crack", "污秽", "裂纹"})


@dataclass
class PipelineConfig:
    """Paths and runtime options for the complete inference pipeline."""

    rgb_dir: Path = ROOT / "dataset" / "images" / "rgb"
    ir_dir: Path = ROOT / "dataset" / "images" / "ir"
    pair_manifest: Path | None = None
    output_dir: Path = ROOT / "runs" / "pipeline"
    fusion_weights: Path = ROOT / "config" / "dense" / "densefuse.pth"
    stage1_weights: Path = ROOT / "runs" / "detect" / "yolo11_rgb" / "weights" / "best.pt"
    stage2_weights: Path = ROOT / "runs" / "detect" / "yolo11_rgb" / "weights" / "best.pt"
    image_size: tuple[int, int] = (256, 256)
    detector_imgsz: int = 640
    confidence: float = 0.25
    iou: float = 0.7
    device: str | None = None
    use_fusion: bool = True
    allow_index_pairing: bool = False
    stage1_classes: tuple[str, ...] = ("insulation", "insulator", "绝缘子", "绝缘部件")
    risk: RiskConfig = field(default_factory=RiskConfig)

    def make_output_dirs(self) -> dict[str, Path]:
        dirs = {
            "fused": self.output_dir / "fused",
            "insulator_crops": self.output_dir / "insulator_crops",
            "stage1": self.output_dir / "stage1",
            "stage2": self.output_dir / "stage2",
            "records": self.output_dir / "records",
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs
