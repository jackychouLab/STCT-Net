import os
from PIL import Image



frame_duration = 30
root = "/home/jackychou/project/datasets/temp_visual_4dfft_wo_win"
save_path = "/home/jackychou/桌面/4dfft_wo_win.gif"
file_names = os.listdir(root)
file_names.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
frames = []
for file_name in file_names:
    file_path = os.path.join(root, file_name)
    img = Image.open(file_path)
    img = img.convert("RGB")
    img = img.resize((1024, 768))
    frames.append(img)

frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=frame_duration, loop=0)