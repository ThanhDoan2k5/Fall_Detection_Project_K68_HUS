# ai_models.py
import numpy as np

def calculate_3d_spine_angle(landmarks):
    S = np.array([(landmarks[11].x + landmarks[12].x)/2.0, (landmarks[11].y + landmarks[12].y)/2.0, (landmarks[11].z + landmarks[12].z)/2.0])
    H = np.array([(landmarks[23].x + landmarks[24].x)/2.0, (landmarks[23].y + landmarks[24].y)/2.0, (landmarks[23].z + landmarks[24].z)/2.0])
    SH = S - H
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(SH, np.array([0,1,0])) / (np.linalg.norm(SH) + 1e-6), -1.0, 1.0)))
    return abs(angle_deg - 180)

def calculate_spine_angle_yolo(kpts):
    try:
        S = (kpts[5] + kpts[6]) / 2.0
        H = (kpts[11] + kpts[12]) / 2.0
        SH = S - H
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(SH, np.array([0,1])) / (np.linalg.norm(SH) + 1e-6), -1.0, 1.0)))
        return abs(angle_deg - 180)
    except: return 0.0

def normalize_keypoints(kpts, confs, min_conf=0.3):
    if confs[11] > min_conf and confs[12] > min_conf:
        hip_x = (kpts[11][0] + kpts[12][0]) / 2.0
        hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
        norm = kpts.copy()
        for i in range(17): 
            norm[i][0] -= hip_x; norm[i][1] -= hip_y
        return norm.flatten()
    return None