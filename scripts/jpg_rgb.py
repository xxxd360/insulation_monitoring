from PIL import Image
import os

input_folder = r"D:\Project\insulation_monitoring\dataset\images\rgb2"  # 存放jpg的文件夹
output_folder = r"D:\Project\insulation_monitoring\dataset\images\rgb2"  # 输出png文件夹
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        in_path = os.path.join(input_folder, filename)
        out_name = os.path.splitext(filename)[0] + ".png"
        out_path = os.path.join(output_folder, out_name)

        img = Image.open(in_path)
        img.save(out_path)
        print(f"已转换：{filename} → {out_name}")