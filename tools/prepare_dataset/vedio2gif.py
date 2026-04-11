from moviepy.editor import VideoFileClip

source_path = "/home/jackychou/project/datasets/wlhk/all/uav_seqs_1/15_23_30_30_wlhk_zy_azimuth.avi"
target_path = "/home/jackychou/桌面/test2.gif"
clip = VideoFileClip(source_path)
clip = clip.subclip(0, 30)
clip = clip.resize(height=480, width=640)
clip.write_gif(target_path, fps=10)