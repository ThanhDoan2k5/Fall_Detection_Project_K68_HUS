import os

video_folder = "dataset/raw_videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

print(f"Đã tìm thấy {len(video_files)} video để train:")
for v in video_files:
    print(f"- {v}")