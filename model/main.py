import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

'''图像融合模型'''
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from image_fusion.DenseFuse import DenseFuseNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model1 = DenseFuseNet().to(device)

checkpoint = torch.load(
    'D:\Project\insulation_monitoring\config\dense\densefuse.pth',
    map_location=device
)

model1.load_state_dict(checkpoint)
model1.eval()


