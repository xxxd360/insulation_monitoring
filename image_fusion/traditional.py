import cv2
import numpy as np
import pywt

def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img.astype(np.float32) / 255.0


def read_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_gray(path, img):
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def save_color(path, img):
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def resize_to_same(img1, img2):
    h, w = img1.shape[:2]
    img2 = cv2.resize(img2, (w, h))
    return img1, img2


#  拉普拉斯金字塔融合
def laplacian_pyramid_fusion(img1, img2, levels=4):
    img1, img2 = resize_to_same(img1, img2)

    gp1 = [img1]
    gp2 = [img2]

    for _ in range(levels):
        gp1.append(cv2.pyrDown(gp1[-1]))
        gp2.append(cv2.pyrDown(gp2[-1]))

    lp1 = []
    lp2 = []

    for i in range(levels):
        size1 = (gp1[i].shape[1], gp1[i].shape[0])
        size2 = (gp2[i].shape[1], gp2[i].shape[0])

        up1 = cv2.pyrUp(gp1[i + 1], dstsize=size1)
        up2 = cv2.pyrUp(gp2[i + 1], dstsize=size2)

        lp1.append(gp1[i] - up1)
        lp2.append(gp2[i] - up2)

    lp1.append(gp1[-1])
    lp2.append(gp2[-1])

    fused_pyramid = []

    for l1, l2 in zip(lp1, lp2):
        mask = np.abs(l1) >= np.abs(l2)
        fused = np.where(mask, l1, l2)
        fused_pyramid.append(fused)

    fused = fused_pyramid[-1]

    for i in range(levels - 1, -1, -1):
        size = (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0])
        fused = cv2.pyrUp(fused, dstsize=size)
        fused = fused + fused_pyramid[i]

    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    return fused


#  小波变换融合
def wavelet_fusion(img1, img2, wavelet="db2"):
    img1, img2 = resize_to_same(img1, img2)

    coeffs1 = pywt.dwt2(img1, wavelet)
    coeffs2 = pywt.dwt2(img2, wavelet)

    cA1, (cH1, cV1, cD1) = coeffs1
    cA2, (cH2, cV2, cD2) = coeffs2

    cA = (cA1 + cA2) / 2

    cH = np.where(np.abs(cH1) >= np.abs(cH2), cH1, cH2)
    cV = np.where(np.abs(cV1) >= np.abs(cV2), cV1, cV2)
    cD = np.where(np.abs(cD1) >= np.abs(cD2), cD1, cD2)

    fused = pywt.idwt2((cA, (cH, cV, cD)), wavelet)

    fused = cv2.resize(fused, (img1.shape[1], img1.shape[0]))
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

    return fused