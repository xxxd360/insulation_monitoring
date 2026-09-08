import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF


# ============================================================
# 路径配置
# ============================================================

rgb_dir = r"D:\Project\insulation_monitoring\dataset\images\rgb"
ir_dir = r"D:\Project\insulation_monitoring\dataset\images\ir"

# RIFT 配准后的 RGB 图像会保存到这里
registered_rgb_dir = r"D:\Project\insulation_monitoring\dataset\images\rgb_registered"

# 这里填写你的 RIFT 仓库路径
# 例如：RIFT_REPO_DIR = r"D:\Project\RIFT"
RIFT_REPO_DIR = r"D:\Project\RIFT"


# ============================================================
# RIFT 配准模块
# ============================================================

class RIFTRegistrar:
    """
    使用 RIFT 将 RGB 图像配准到 IR 图像。

    当前设计：
    - IR 图像作为 fixed/reference image
    - RGB 图像作为 moving image
    - RIFT 提取跨模态匹配点
    - RANSAC 估计 Homography
    - cv2.warpPerspective 将 RGB warp 到 IR 坐标系

    注意：
    不同 RIFT 仓库的 Python 接口不一样，所以你主要需要根据自己的 RIFT 代码
    修改 _match_points() 这个函数。
    """

    def __init__(
        self,
        rift_repo_dir=None,
        min_matches=12,
        min_inliers=8,
        ransac_reproj_threshold=4.0,
    ):
        self.rift_repo_dir = rift_repo_dir
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold

        if self.rift_repo_dir is not None:
            rift_path = str(Path(self.rift_repo_dir))
            if rift_path not in sys.path:
                sys.path.append(rift_path)

    @staticmethod
    def _pil_rgb_to_numpy(rgb_img):
        return np.array(rgb_img.convert("RGB"))

    @staticmethod
    def _pil_ir_to_numpy(ir_img):
        return np.array(ir_img.convert("L"))

    def _match_points(self, moving_rgb, fixed_gray):
        """
        返回 RGB 图像和 IR 图像之间的匹配点。

        参数：
        - moving_rgb: RGB 图像，numpy, H x W x 3, uint8
        - fixed_gray: IR 灰度图，numpy, H x W, uint8

        返回：
        - mkpts_moving: RGB 图像中的点，N x 2，格式为 [[x, y], ...]
        - mkpts_fixed: IR 图像中的点，N x 2，格式为 [[x, y], ...]

        你需要根据自己的 RIFT 代码修改这里。

        常见情况：
        1. 如果你的 RIFT 函数返回 matches：
           matches = rift_match(img1, img2)
           mkpts1 = matches[:, 0:2]
           mkpts2 = matches[:, 2:4]

        2. 如果你的 RIFT 函数返回两个点集：
           mkpts1, mkpts2 = rift_match(img1, img2)

        3. 如果你的 RIFT 是 MATLAB 版本，建议先离线生成匹配点或配准图，
           不建议在 PyTorch Dataset 里面直接调 MATLAB。
        """

        # ------------------------------------------------------------
        # 下面这段是接口示例。
        # 你需要按你下载的 RIFT 仓库实际函数名进行修改。
        # ------------------------------------------------------------

        try:
            # 示例 1：
            # 假设你的 RIFT 仓库里面有 rift.py，并提供 rift_match 函数。
            #
            # rift_match(moving_gray, fixed_gray) 返回：
            # - mkpts_moving: N x 2
            # - mkpts_fixed: N x 2
            #
            from rift import rift_match

            moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)

            mkpts_moving, mkpts_fixed = rift_match(
                moving_gray,
                fixed_gray,
            )

        except ImportError as e:
            raise ImportError(
                "无法导入 RIFT。\n"
                "请确认：\n"
                "1. 你已经下载 RIFT Python 版本代码；\n"
                "2. RIFT_REPO_DIR 已经设置为你的 RIFT 仓库路径；\n"
                "3. 仓库中存在可以导入的 rift_match 函数。\n\n"
                "如果你的 RIFT 函数名不是 rift_match，请修改 "
                "RIFTRegistrar._match_points() 里的 import 和调用方式。"
            ) from e

        mkpts_moving = np.asarray(mkpts_moving, dtype=np.float32)
        mkpts_fixed = np.asarray(mkpts_fixed, dtype=np.float32)

        if mkpts_moving.ndim != 2 or mkpts_moving.shape[1] != 2:
            raise ValueError("mkpts_moving must have shape [N, 2]")

        if mkpts_fixed.ndim != 2 or mkpts_fixed.shape[1] != 2:
            raise ValueError("mkpts_fixed must have shape [N, 2]")

        return mkpts_moving, mkpts_fixed

    def register(self, rgb_img, ir_img):
        """
        将 RGB 图像配准到 IR 图像。

        输入：
        - rgb_img: PIL RGB
        - ir_img: PIL L

        输出：
        - registered_rgb_img: PIL RGB
        """
        moving_rgb = self._pil_rgb_to_numpy(rgb_img)
        fixed_gray = self._pil_ir_to_numpy(ir_img)

        h_fixed, w_fixed = fixed_gray.shape[:2]

        try:
            mkpts_moving, mkpts_fixed = self._match_points(
                moving_rgb=moving_rgb,
                fixed_gray=fixed_gray,
            )

            if len(mkpts_moving) < self.min_matches or len(mkpts_fixed) < self.min_matches:
                print("[RIFT registration warning] 匹配点数量不足，使用原始 RGB。")
                return rgb_img.convert("RGB")

            H, mask = cv2.findHomography(
                mkpts_moving,
                mkpts_fixed,
                cv2.RANSAC,
                self.ransac_reproj_threshold,
            )

            if H is None or mask is None:
                print("[RIFT registration warning] Homography 估计失败，使用原始 RGB。")
                return rgb_img.convert("RGB")

            inliers = int(mask.ravel().sum())
            if inliers < self.min_inliers:
                print(
                    f"[RIFT registration warning] RANSAC 内点过少：{inliers}，"
                    "使用原始 RGB。"
                )
                return rgb_img.convert("RGB")

            registered_rgb = cv2.warpPerspective(
                moving_rgb,
                H,
                dsize=(w_fixed, h_fixed),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            registered_rgb_img = Image.fromarray(registered_rgb).convert("RGB")
            return registered_rgb_img

        except Exception as e:
            print(f"[RIFT registration warning] 配准失败，使用原始 RGB。原因：{e}")
            return rgb_img.convert("RGB")


# ============================================================
# Dataset
# ============================================================

class RGBIRDataset(Dataset):
    def __init__(
        self,
        rgb_dir,
        ir_dir,
        image_size=(256, 256),
        enable_registration=True,
        registered_rgb_dir=None,
        rift_repo_dir=None,
    ):
        self.rgb_paths = sorted(Path(rgb_dir).glob("*"))
        self.ir_paths = sorted(Path(ir_dir).glob("*"))

        self.image_size = image_size
        self.enable_registration = enable_registration

        assert len(self.rgb_paths) == len(self.ir_paths), \
            "RGB and IR datasets must have the same number of images"

        self.registered_rgb_dir = None
        if registered_rgb_dir is not None:
            self.registered_rgb_dir = Path(registered_rgb_dir)
            self.registered_rgb_dir.mkdir(parents=True, exist_ok=True)

        height, width = image_size

        self.rgb_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

        self.ir_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

        self.rift_repo_dir = rift_repo_dir
        self.registrar = None

    def _lazy_init_registrar(self):
        if self.registrar is None:
            self.registrar = RIFTRegistrar(
                rift_repo_dir=self.rift_repo_dir,
                min_matches=12,
                min_inliers=8,
                ransac_reproj_threshold=4.0,
            )

    def _get_registered_rgb_path(self, index):
        if self.registered_rgb_dir is None:
            return None

        rgb_path = self.rgb_paths[index]
        return self.registered_rgb_dir / rgb_path.name

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        rgb_img = Image.open(self.rgb_paths[index]).convert("RGB")
        ir_img = Image.open(self.ir_paths[index]).convert("L")

        if self.enable_registration:
            registered_path = self._get_registered_rgb_path(index)

            if registered_path is not None and registered_path.exists():
                rgb_img = Image.open(registered_path).convert("RGB")
            else:
                self._lazy_init_registrar()
                rgb_img = self.registrar.register(rgb_img, ir_img)

                if registered_path is not None:
                    rgb_img.save(registered_path)

        rgb_tensor = self.rgb_transform(rgb_img)  # [3, H, W]
        ir_tensor = self.ir_transform(ir_img)     # [1, H, W]

        return rgb_tensor, ir_tensor


# ============================================================
# DenseFuse 网络结构
# ============================================================

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return self.layer(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16, num_layers=4):
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for _ in range(num_layers):
            self.layers.append(DenseLayer(current_channels, growth_rate))
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            fused_input = torch.cat(features, dim=1)
            new_feat = layer(fused_input)
            features.append(new_feat)

        return torch.cat(features, dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        growth_rate=16,
        num_layers=4,
    ):
        super().__init__()

        self.stem = ConvBNReLU(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.dense1 = DenseBlock(
            base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )

        self.compress1 = ConvBNReLU(
            self.dense1.out_channels,
            base_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.dense2 = DenseBlock(
            base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )

        self.out_channels = self.dense2.out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.compress1(x)
        x = self.dense2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=16,
        growth_rate=16,
        num_layers=4,
        out_channels=1,
    ):
        super().__init__()

        self.stem = ConvBNReLU(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.dense1 = DenseBlock(
            base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )

        self.compress1 = ConvBNReLU(
            self.dense1.out_channels,
            base_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.dense2 = DenseBlock(
            base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )

        self.reconstruct = nn.Conv2d(
            self.dense2.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.compress1(x)
        x = self.dense2(x)
        x = self.reconstruct(x)
        return torch.sigmoid(x)


class DenseFuseNet(nn.Module):
    """
    DenseFuse:
    - shared encoder for infrared and visible images
    - feature fusion in latent space
    - decoder reconstructs fused image
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        growth_rate=16,
        num_layers=4,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
        )

        self.decoder = Decoder(
            in_channels=self.encoder.out_channels,
            base_channels=base_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            out_channels=in_channels,
        )

    @staticmethod
    def fuse_add(feat1, feat2):
        return 0.5 * (feat1 + feat2)

    @staticmethod
    def fuse_l1(feat1, feat2, eps=1e-8):
        """
        L1 自适应融合：
        对两个特征图分别计算通道平均绝对响应，响应越强的位置权重越大。
        """
        w1 = torch.mean(torch.abs(feat1), dim=1, keepdim=True)
        w2 = torch.mean(torch.abs(feat2), dim=1, keepdim=True)

        weight_sum = w1 + w2 + eps
        w1 = w1 / weight_sum
        w2 = w2 / weight_sum

        return w1 * feat1 + w2 * feat2

    def encode(self, x):
        return self.encoder(x)

    def decode(self, feat):
        return self.decoder(feat)

    def forward(self, img1, img2, fusion_mode="l1"):
        feat1 = self.encode(img1)
        feat2 = self.encode(img2)

        if fusion_mode == "add":
            fused_feat = self.fuse_add(feat1, feat2)
        elif fusion_mode == "l1":
            fused_feat = self.fuse_l1(feat1, feat2)
        else:
            raise ValueError("fusion_mode must be 'add' or 'l1'")

        fused_img = self.decode(fused_feat)
        return fused_img, feat1, feat2, fused_feat

    def reconstruct(self, img):
        feat = self.encode(img)
        return self.decode(feat)


# ============================================================
# Sobel 梯度
# ============================================================

def sobel_grad(x):
    """
    Compute gradient magnitude for grayscale images.
    x: [B, C, H, W]
    """
    device = x.device
    dtype = x.dtype
    c = x.shape[1]

    kernel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=dtype,
        device=device,
    ).view(1, 1, 3, 3)

    kernel_x = kernel_x.repeat(c, 1, 1, 1)
    kernel_y = kernel_y.repeat(c, 1, 1, 1)

    gx = F.conv2d(x, kernel_x, padding=1, groups=c)
    gy = F.conv2d(x, kernel_y, padding=1, groups=c)

    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return grad


# ============================================================
# 融合损失
# ============================================================

class DenseFuseFusionLoss(nn.Module):
    """
    无监督图像融合损失：
    - intensity term: 保持输出接近输入强度
    - gradient term: 保留两幅图像中较强的边缘信息
    """

    def __init__(self, w_intensity=1.0, w_grad=1.0):
        super().__init__()
        self.w_intensity = w_intensity
        self.w_grad = w_grad

    def forward(self, fused, img1, img2):
        target_i = 0.5 * (img1 + img2)
        loss_intensity = F.l1_loss(fused, target_i)

        grad_fused = sobel_grad(fused)
        grad1 = sobel_grad(img1)
        grad2 = sobel_grad(img2)

        grad_target = torch.max(grad1, grad2)
        loss_grad = F.l1_loss(grad_fused, grad_target)

        loss = self.w_intensity * loss_intensity + self.w_grad * loss_grad
        return loss


# ============================================================
# 可选：单独预生成 RIFT 配准图像
# ============================================================

def build_registered_rgb_dataset(
    rgb_dir,
    ir_dir,
    registered_rgb_dir,
    rift_repo_dir=RIFT_REPO_DIR,
):
    """
    单独执行 RIFT 配准，并把配准后的 RGB 图像保存到 registered_rgb_dir。

    推荐先运行这个函数生成配准数据集，再训练 DenseFuse。
    这样训练阶段不会被配准过程拖慢。
    """
    rgb_paths = sorted(Path(rgb_dir).glob("*"))
    ir_paths = sorted(Path(ir_dir).glob("*"))

    assert len(rgb_paths) == len(ir_paths), \
        "RGB and IR datasets must have the same number of images"

    registered_rgb_dir = Path(registered_rgb_dir)
    registered_rgb_dir.mkdir(parents=True, exist_ok=True)

    registrar = RIFTRegistrar(
        rift_repo_dir=rift_repo_dir,
        min_matches=12,
        min_inliers=8,
        ransac_reproj_threshold=4.0,
    )

    for index, (rgb_path, ir_path) in enumerate(zip(rgb_paths, ir_paths)):
        save_path = registered_rgb_dir / rgb_path.name

        if save_path.exists():
            print(f"[{index + 1}/{len(rgb_paths)}] Exists: {save_path}")
            continue

        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img = Image.open(ir_path).convert("L")

        registered_rgb = registrar.register(rgb_img, ir_img)
        registered_rgb.save(save_path)

        print(f"[{index + 1}/{len(rgb_paths)}] Saved: {save_path}")


# ============================================================
# 训练函数
# ============================================================

def train(
    rgb_dir,
    ir_dir,
    epochs=20,
    batch_size=4,
    image_size=(256, 256),
    lr=1e-4,
    device=None,
    save_path=r"D:\Project\insulation_monitoring\config\dense\densefuse.pth",
    enable_registration=True,
    registered_rgb_dir=registered_rgb_dir,
    rift_repo_dir=RIFT_REPO_DIR,
    num_workers=0,
):
    """
    训练 DenseFuse。

    参数：
    - enable_registration:
        True  -> 使用 RIFT 先把 RGB 配准到 IR
        False -> 不配准，直接训练

    - registered_rgb_dir:
        配准结果缓存目录。
        如果目录中已经有对应图像，就直接读取，不重复配准。

    - num_workers:
        如果开启在线配准，建议先用 0。
        因为 RIFT + DataLoader 多进程可能会出现导入或资源冲突。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RGBIRDataset(
        rgb_dir=rgb_dir,
        ir_dir=ir_dir,
        image_size=image_size,
        enable_registration=enable_registration,
        registered_rgb_dir=registered_rgb_dir,
        rift_repo_dir=rift_repo_dir,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model = DenseFuseNet(
        in_channels=1,
        base_channels=16,
        growth_rate=16,
        num_layers=4,
    ).to(device)

    loss_fn = DenseFuseFusionLoss(
        w_intensity=1.0,
        w_grad=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for step, (rgb, ir) in enumerate(loader):
            rgb = rgb.to(device)  # [B, 3, H, W]
            ir = ir.to(device)    # [B, 1, H, W]

            rgb_gray = TF.rgb_to_grayscale(rgb)  # [B, 1, H, W]

            fused, feat_ir, feat_vi, fused_feat = model(
                ir,
                rgb_gray,
                fusion_mode="l1",
            )

            loss = loss_fn(
                fused,
                ir,
                rgb_gray,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Step [{step + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / len(loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Average Loss: {avg_loss:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "enable_registration": enable_registration,
                    "registration_method": "RIFT",
                },
                str(save_path),
            )

            print(f"Saved best model to: {save_path}")

    return model


if __name__ == "__main__":
    # 推荐方式 1：
    # 直接训练。第一次训练时会自动生成 rgb_registered 缓存。
    train(
        rgb_dir=rgb_dir,
        ir_dir=ir_dir,
        epochs=20,
        batch_size=4,
        image_size=(256, 256),
        lr=1e-4,
        device="cuda",
        enable_registration=True,
        registered_rgb_dir=registered_rgb_dir,
        rift_repo_dir=RIFT_REPO_DIR,
        num_workers=0,
    )

    # 推荐方式 2：
    # 如果你想先单独生成配准数据集，再训练，可以先运行这个函数：
    #
    # build_registered_rgb_dataset(
    #     rgb_dir=rgb_dir,
    #     ir_dir=ir_dir,
    #     registered_rgb_dir=registered_rgb_dir,
    #     rift_repo_dir=RIFT_REPO_DIR,
    # )
    #
    # 然后训练时关闭在线配准，直接把 rgb_dir 指向 registered_rgb_dir：
    #
    # train(
    #     rgb_dir=registered_rgb_dir,
    #     ir_dir=ir_dir,
    #     epochs=20,
    #     batch_size=4,
    #     image_size=(256, 256),
    #     lr=1e-4,
    #     device="cuda",
    #     enable_registration=False,
    #     num_workers=4,
    # )