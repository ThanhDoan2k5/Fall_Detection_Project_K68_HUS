import os
import cv2
import glob
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load model pose
model = YOLO('yolov8n-pose.pt')

def extract_and_autolabel(folder_path, is_fall_video):
    sequence_data = []
    
    # Quét đúng file .png
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if not image_paths: 
        print(f"   -> CẢNH BÁO: Không tìm thấy ảnh .png trong {folder_path}")
        return []

    print(f"   -> Đang xử lý: {os.path.basename(folder_path)} ({len(image_paths)} frames)")

    ratios = []
    raw_keypoints = []

    # BƯỚC 1: Thu thập dữ liệu
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
            
        results = model(img, verbose=False)
        kpts_flattened = None
        ratio = 0.0
        
        # Nếu detect được người
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
            kpts = results[0].keypoints.xyn[0].cpu().numpy()
            
            # Chỉ lấy nếu đủ 17 khớp (tránh detect lỗi)
            if len(kpts) == 17:
                hip_x = (kpts[11][0] + kpts[12][0]) / 2.0
                hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
                
                # Normalize tọa độ (Tương đối so với hông)
                kpts_norm = kpts.copy()
                kpts_norm[:, 0] -= hip_x
                kpts_norm[:, 1] -= hip_y
                kpts_flattened = kpts_norm.flatten()
                
                # Tính ratio BBox
                box = results[0].boxes.xywh[0].cpu().numpy()
                if box[3] > 0: ratio = box[2] / box[3]

        raw_keypoints.append(kpts_flattened)
        ratios.append(ratio)

    # BƯỚC 2: Gán nhãn động
    labels = [0] * len(raw_keypoints)
    if is_fall_video:
        # Tìm frame "chạm đất" (Ratio > 1.2)
        fall_idx = next((i for i, r in enumerate(ratios) if r > 1.2), -1)
        if fall_idx != -1:
            # 30 frames trước chạm đất = Ngã (2)
            start_fall = max(0, fall_idx - 30)
            for i in range(start_fall, fall_idx): labels[i] = 2
            # Sau chạm đất = Nằm (1)
            for i in range(fall_idx, len(labels)): labels[i] = 1
    else:
        # Video ADL: Nếu ratio cao (nằm) = (1), không thì (0)
        for i, r in enumerate(ratios): labels[i] = 1 if r > 1.2 else 0

    # BƯỚC 3: Đóng gói (Lọc bỏ khung hình không detect được)
    for i in range(len(raw_keypoints)):
        if raw_keypoints[i] is not None:
            row = list(raw_keypoints[i]) + [labels[i]]
            sequence_data.append(row)
        
    return sequence_data

# --- CẤU HÌNH ĐƯỜNG DẪN TUYỆT ĐỐI ---
BASE_DIR = r"C:\Du_An_Fall_Detection\dataset_raw"
OUTPUT_DIR = r"C:\Du_An_Fall_Detection\dataset\processed_csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_dataset.csv")

# Tự động tạo folder nếu chưa có
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

all_data = []

print("📂 Bắt đầu xử lý dữ liệu...")
# Xử lý Falls
fall_folders = glob.glob(os.path.join(BASE_DIR, "falls", "*"))
for folder in sorted(fall_folders):
    if os.path.isdir(folder): 
        all_data.extend(extract_and_autolabel(folder, is_fall_video=True))

# Xử lý ADL
adl_folders = glob.glob(os.path.join(BASE_DIR, "adl", "*"))
for folder in sorted(adl_folders):
    if os.path.isdir(folder): 
        all_data.extend(extract_and_autolabel(folder, is_fall_video=False))

# Xuất file
if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Xong! File đã lưu tại: {OUTPUT_FILE}")
else:
    print("\n⚠️ Không có dữ liệu nào được trích xuất!")