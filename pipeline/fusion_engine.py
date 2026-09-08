from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from image_fusion.DenseFuse import DenseFuseNet


class DenseFuseEngine:
    """Load DenseFuse and turn an RGB/IR pair into a grayscale fused image."""

    def __init__(self, weight_path: Path, image_size: tuple[int, int], device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.model = DenseFuseNet(in_channels=1, base_channels=16, growth_rate=16, num_layers=4).to(self.device)
        checkpoint = torch.load(weight_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.to_tensor = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    @torch.inference_mode()
    def fuse(self, rgb_path: Path, ir_path: Path) -> Image.Image:
        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img = Image.open(ir_path).convert("L")
        rgb = self.to_tensor(rgb_img).unsqueeze(0).to(self.device)
        ir = self.to_tensor(ir_img).unsqueeze(0).to(self.device)
        rgb_gray = TF.rgb_to_grayscale(rgb)
        fused, _, _, _ = self.model(ir, rgb_gray, fusion_mode="l1")
        fused = fused.squeeze(0).cpu().clamp(0, 1)
        return transforms.ToPILImage()(fused).resize(rgb_img.size, Image.Resampling.BILINEAR)

    @staticmethod
    def as_detector_image(image: Image.Image) -> np.ndarray:
        """YOLO accepts RGB arrays; replicate the fused grayscale channel."""
        gray = np.asarray(image.convert("L"))
        return np.repeat(gray[..., None], 3, axis=2)
