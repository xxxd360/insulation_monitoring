from pathlib import Path
import random
import shutil

# 输入文件夹
INPUT_DIR = Path("../dataset/images/ir")

# 输出文件夹
OUTPUT_DIR = Path("../dataset/images/ir")

# 支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

OUTPUT_DIR.mkdir(exist_ok=True)

images = sorted(
    [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS],
    key=lambda p: p.name
)

if len(images) < 2:
    raise ValueError("ir 文件夹中至少需要 2 张图片")

groups = [images[i:i + 2] for i in range(0, len(images), 2)]

output_index = 1

for group_index, group in enumerate(groups):
    # 复制当前小组的图片
    for img in group:
        new_name = f"{output_index:04d}_{img.name}"
        shutil.copy2(img, OUTPUT_DIR / new_name)
        output_index += 1

    # 如果后面还有小组，则随机复制当前小组中的一张，插入到小组之间
    if group_index < len(groups) - 1:
        chosen = random.choice(group)
        new_name = f"{output_index:04d}_random_{chosen.name}"
        shutil.copy2(chosen, OUTPUT_DIR / new_name)
        output_index += 1

print(f"完成，共输出 {output_index - 1} 张图片到 {OUTPUT_DIR}")
