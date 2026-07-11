from pathlib import Path

img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ir_dir = Path("../dataset/images/ir")
rgb_dir = Path("../dataset/images/rgb")

ir_imgs = [p for p in ir_dir.rglob("*") if p.suffix.lower() in img_exts]
rgb_imgs = [p for p in rgb_dir.rglob("*") if p.suffix.lower() in img_exts]

print(f"ir  图片数量: {len(ir_imgs)}")
print(f"rgb 图片数量: {len(rgb_imgs)}")

if len(ir_imgs) == len(rgb_imgs):
    print("数量一致")
else:
    print(f"数量不一致，相差 {abs(len(ir_imgs) - len(rgb_imgs))} 张")

