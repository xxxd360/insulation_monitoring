from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion -> insulator detection -> defect detection -> risk assessment.")
    parser.add_argument("--rgb-dir", type=Path, default=None)
    parser.add_argument("--ir-dir", type=Path, default=None)
    parser.add_argument("--pair-manifest", type=Path, default=None, help="CSV 文件，列名为 rgb,ir")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage1-weights", type=Path, default=None)
    parser.add_argument("--stage2-weights", type=Path, default=None)
    parser.add_argument("--fusion-weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--allow-index-pairing", action="store_true", help="RGB/IR 数量不一致时按排序索引临时配对")
    args = parser.parse_args()

    config = PipelineConfig()
    for name in ("rgb_dir", "ir_dir", "pair_manifest", "output_dir", "stage1_weights", "stage2_weights", "fusion_weights"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    config.confidence = args.conf
    config.device = args.device
    config.use_fusion = not args.no_fusion
    config.allow_index_pairing = args.allow_index_pairing
    run_pipeline(config)


if __name__ == "__main__":
    main()
