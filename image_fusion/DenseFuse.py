import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

'''尺寸调整函数'''
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

rgb_dir = r'D:\Project\insulation_monitoring\dataset\images\rgb'
ir_dir = r'D:\Project\insulation_monitoring\dataset\images\ir'
class RGBIRDataset(Dataset):
    def __init__(self, rgb_dir, ir_dir, image_size=(256, 256)):

        self.rgb_paths = sorted(Path(rgb_dir).glob("*"))
        self.ir_paths = sorted(Path(ir_dir).glob("*"))
        self.image_size = image_size

        assert len(self.rgb_paths) == len(self.ir_paths), \
            "RGB and IR datasets must have the same number of images"

        height, width = image_size
        self.rgb_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        self.ir_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, index):
        rgb_img = Image.open(self.rgb_paths[index]).convert("RGB")
        ir_img = Image.open(self.ir_paths[index]).convert("L")

        rgb_tensor = self.rgb_transform(rgb_img)  # [3, H, W]
        ir_tensor = self.ir_transform(ir_img)     # [1, H, W]

        return rgb_tensor, ir_tensor

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
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
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
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



'''上面的是densenet的密集连接
输入 x
 ↓
第1层看 x，生成 f1
 ↓
第2层看 [x, f1]，生成 f2
 ↓
第3层看 [x, f1, f2]，生成 f3
 ↓
第4层看 [x, f1, f2, f3]，生成 f4
 ↓
输出 [x, f1, f2, f3, f4]

'''
class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, growth_rate=16, num_layers=4):
        super().__init__()
        '''特征提取'''
        self.stem = ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.dense1 = DenseBlock(base_channels, growth_rate=growth_rate, num_layers=num_layers)
        '''压缩层'''
        self.compress1 = ConvBNReLU(self.dense1.out_channels, base_channels, kernel_size=1, stride=1, padding=0)
        self.dense2 = DenseBlock(base_channels, growth_rate=growth_rate, num_layers=num_layers)

        self.out_channels = self.dense2.out_channels

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.compress1(x)
        x = self.dense2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, base_channels=16, growth_rate=16, num_layers=4, out_channels=1):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.dense1 = DenseBlock(base_channels, growth_rate=growth_rate, num_layers=num_layers)
        self.compress1 = ConvBNReLU(self.dense1.out_channels, base_channels, kernel_size=1, stride=1, padding=0)
        self.dense2 = DenseBlock(base_channels, growth_rate=growth_rate, num_layers=num_layers)
        '''特征转换'''
        self.reconstruct = nn.Conv2d(self.dense2.out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.dense1(x)
        x = self.compress1(x)
        x = self.dense2(x)
        x = self.reconstruct(x)
        return torch.sigmoid(x)


class DenseFuseNet(nn.Module):
    """
    DenseFuse reproduction:
    - shared encoder for infrared and visible images
    - feature fusion in latent space
    - decoder reconstructs fused image
    """
    def __init__(self, in_channels=1, base_channels=16, growth_rate=16, num_layers=4):
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
    '''简单加权'''
    @staticmethod
    def fuse_add(feat1, feat2):
        return 0.5 * (feat1 + feat2)
    '''L1自适应'''
    @staticmethod
    def fuse_l1(feat1, feat2, eps=1e-8):
        # Channel-wise L1 energy weighting
        w1 = torch.mean(torch.abs(feat1), dim=1, keepdim=True)
        w2 = torch.mean(torch.abs(feat2), dim=1, keepdim=True)
        '''
        .abs会把张量全部变成非负数
        对通道求平均值，使图像的H*W都有权重
        '''
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

'''Sobel算子：计算边缘强度'''
def sobel_grad(x):
    """
    Compute gradient magnitude for grayscale images.
    x: [B, C, H, W]
    """
    device = x.device
    dtype = x.dtype
    c = x.shape[1]

    kernel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=dtype,
        device=device
    ).view(1, 1, 3, 3)

    kernel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=dtype,
        device=device
    ).view(1, 1, 3, 3)

    kernel_x = kernel_x.repeat(c, 1, 1, 1)
    kernel_y = kernel_y.repeat(c, 1, 1, 1)
    '''groups分组卷积
    如果不用 groups=c，这个卷积核形状 [C, 1, 3, 3] 通常会和输入 [B, C, H, W] 对不上，
    因为普通卷积期望权重形状类似 [out_channels, C, 3, 3]'''
    gx = F.conv2d(x, kernel_x, padding=1, groups=c)
    gy = F.conv2d(x, kernel_y, padding=1, groups=c)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    '''生成边缘强度图'''
    return grad

'''无监督图像融合损失函数'''
class DenseFuseFusionLoss(nn.Module):
    """
    Simple unsupervised fusion loss:
    - intensity term: keep output close to stronger source intensity
    - gradient term: preserve edge information
    """
    def __init__(self, w_intensity=1.0, w_grad=1.0):
        super().__init__()
        self.w_intensity = w_intensity
        self.w_grad = w_grad

    def forward(self, fused, img1, img2):
        # Intensity target: average of inputs
        '''这一步还可以讨论学习'''
        target_i = 0.5 * (img1 + img2)
        loss_intensity = F.l1_loss(fused, target_i)

        # Edge target: stronger gradient from either source
        grad_fused = sobel_grad(fused)
        grad1 = sobel_grad(img1)
        grad2 = sobel_grad(img2)
        '''取最强的边缘'''
        grad_target = torch.max(grad1, grad2)
        loss_grad = F.l1_loss(grad_fused, grad_target)

        return self.w_intensity * loss_intensity + self.w_grad * loss_grad


def train(
    rgb_dir,
    ir_dir,
    epochs=20,
    batch_size=4,
    image_size=(256, 256),
    lr=1e-4,
    device="cuda",
    save_path="D:\Project\insulation_monitoring\config\dense\densefuse.pth",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RGBIRDataset(
        rgb_dir=rgb_dir,
        ir_dir=ir_dir,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
                },
                save_path,
            )


    return model
if __name__ == "__main__":
    train(rgb_dir,ir_dir,device="cuda")
