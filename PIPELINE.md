# 端到端检测流程

当前工程按原始方案组织为：

```text
RGB/IR 配对 → DenseFuse 融合 → 一级绝缘子检测 → bbox 裁剪
→ 二级缺陷检测 → 面积/类别风险评估 → 图片和 JSON 记录
```

流程图见项目根目录的 `pipeline_flow.svg`。

## 目录

```text
pipeline/
  config.py          路径、阈值和运行参数
  fusion_engine.py   DenseFuse checkpoint 加载与融合
  detection.py       YOLO 检测、类别过滤和 bbox 裁剪
  risk.py            四级风险启发式评估
  runner.py          全流程编排与结果落盘
  run_pipeline.py    命令行入口
```

## 运行

在项目根目录执行：

```powershell
python -m pipeline.run_pipeline
```

当前数据目录中的 RGB 和 IR 数量不一致。正式运行前应建立一份按样本 ID 对齐的配对清单；仅做临时流程验证时可显式使用 `--allow-index-pairing`，程序会按排序索引取前 `min(N_rgb, N_ir)` 对，并在终端给出警告。

可以先生成待人工复核的候选清单：

```powershell
python scripts/create_pair_manifest.py
```

清单确认后运行：

```powershell
python -m pipeline.run_pipeline --pair-manifest dataset/pairs.csv
```

CSV 需要包含 `rgb,ir` 两列；相对路径分别相对于 `--rgb-dir` 和 `--ir-dir` 解析。

常用参数：

```powershell
python -m pipeline.run_pipeline `
  --stage1-weights runs/detect/yolo11_rgb/weights/best.pt `
  --stage2-weights runs/detect/yolo11_rgb/weights/best.pt `
  --output-dir runs/pipeline
```

## 训练两个独立检测器

先运行 DenseFuse 推理，生成保留原始文件名的融合图：

```powershell
python scripts/densefuse_inference.py
```

再根据融合图上的绝缘子框和缺陷框生成两级数据集：

```powershell
python scripts/prepare_two_stage_dataset.py --image-dir dataset/images/after_fusion
```

再分别训练一级和二级 YOLO：

```powershell
python scripts/train_two_stage.py
```

训练完成后，将 `runs/detect/stage1_insulator/weights/best.pt` 和
`runs/detect/stage2_defect/weights/best.pt` 传给 `pipeline.run_pipeline`。

输出目录包括：

- `fused/`：融合图像
- `stage1/`：一级检测可视化
- `insulator_crops/`：绝缘子裁剪图
- `stage2/`：二级检测可视化
- `records/`：每张图的 JSON 记录
- `summary.json`：总体汇总

## 当前模型边界

现有仓库只有一个已训练的 YOLO 权重，且它是四类检测模型。为了让流程可以先跑通，入口允许把同一权重同时作为一级和二级模型；实际项目中应分别训练：

1. 一级模型：只检测绝缘子/绝缘部件。
2. 二级模型：只在绝缘子裁剪图中检测污秽、缺陷、剥落等异常。

当前融合输出是单通道图像复制为三通道后送入 YOLO。因此两级 YOLO 应使用融合图训练；仓库中现有的 RGB 权重只能用于流程冒烟或临时占位，不能直接视为融合输入上的最终性能。

风险等级目前是可解释的启发式规则，不代表已完成绝缘电气状态标定；后续应使用配对电气试验数据校准阈值和类别严重程度。
