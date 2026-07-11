from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

# Allow running this script from the project root or image_fusion directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from image_fusion.DenseFuse import DenseFuseNet


RGB_DIR = ROOT / "dataset" / "images" / "rgb"
IR_DIR = ROOT / "dataset" / "images" / "ir"
OUT_DIR = ROOT / "dataset" / "images" / "after_fusion"
WEIGHT_PATH = ROOT / "config" / "dense" / "densefuse.pth"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (256, 256)


def image_files(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def load_model(device: str):
    model = DenseFuseNet(in_channels=1, base_channels=16, growth_rate=16, num_layers=4).to(device)
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb_paths = image_files(RGB_DIR)
    ir_paths = image_files(IR_DIR)

    if len(rgb_paths) != len(ir_paths):
        raise ValueError(f"rgb 和 ir 数量不一致: rgb={len(rgb_paths)}, ir={len(ir_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    to_tensor = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    for idx, (rgb_path, ir_path) in enumerate(zip(rgb_paths, ir_paths), start=1):
        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img = Image.open(ir_path).convert("L")
        original_size = rgb_img.size

        rgb = to_tensor(rgb_img).unsqueeze(0).to(device)
        ir = to_tensor(ir_img).unsqueeze(0).to(device)
        rgb_gray = TF.rgb_to_grayscale(rgb)

        with torch.no_grad():
            fused, _, _, _ = model(ir, rgb_gray, fusion_mode="l1")

        fused = fused.squeeze(0).cpu().clamp(0, 1)
        fused_img = transforms.ToPILImage()(fused)
        fused_img = fused_img.resize(original_size, Image.BILINEAR)

        save_path = OUT_DIR / f"{idx:04d}_fusion.png"
        fused_img.save(save_path)
        print(f"[{idx}/{len(rgb_paths)}] saved: {save_path.name}")

    print(f"完成，融合结果已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
