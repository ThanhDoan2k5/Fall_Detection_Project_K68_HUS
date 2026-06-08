# main.py - FINAL STABLE VERSION
import cv2, torch, time, winsound, csv, numpy as np 
from collections import deque
from ultralytics import YOLO
from train_lstm import FallDetectionLSTM
import mediapipe as mp

from config import CONFIG
from camera_utils import ThreadedCamera
from ai_models import calculate_3d_spine_angle, calculate_spine_angle_yolo, normalize_keypoints

# --- KHỞI TẠO ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8n-pose.pt')
lstm_model = FallDetectionLSTM(input_size=34, hidden_size=64, num_layers=2, num_classes=3)
lstm_model.load_state_dict(torch.load("fall_detection_lstm.pth", map_location=device))
lstm_model.to(device).eval()

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

# --- BIẾN TOÀN CỤC ---
frame_queue = deque(maxlen=CONFIG['WINDOW_SIZE'])
angle_history = deque(maxlen=15)
ratio_history = deque(maxlen=15)
fps_history = deque(maxlen=30)
cx_history, cy_history = deque(maxlen=15), deque(maxlen=15)

locked_id, lost_counter, MAX_LOST = None, 0, 60
fall_vote, suspected_fall, fall_detected, block_new_fall, lying_frame_count = 0, False, False, False, 0
prev_ang, prev_time, last_log_time, last_beep_time, alarm_start_time = 0.0, time.time(), time.time(), 0.0, 0.0
head_y_prev, head_y_prev_time, head_stuck_counter = None, None, 0

# --- SETUP LOGGING ---
log_file = open(CONFIG['CSV_LOG'], 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['timestamp', 'pred', 'p2', 'spine_angle', 'vote', 'state', 'lying_frames', 'block_status', 'fps', 'occlusion'])

cap = ThreadedCamera(CONFIG['IP_CAMERA'])
time.sleep(2)

print("🚀 v3.2 FINAL - Hệ thống đang hoạt động ổn định!")

while True:
    ret, frame = cap.read()
    if not ret or frame is None: continue
    frame = cv2.resize(frame, (640, 480))
    
    # --- TRACKING (ĐÃ FIX LỖI: Bỏ tham số tracker để dùng mặc định của thư viện) ---
    results = yolo_model.track(frame, conf=0.5, half=(device=='cuda'), persist=True, verbose=False)
    
    found_target = False
    curr_ang = prev_ang
    occlusion_case = 'none'
    
    if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # ID Locking: Bám theo ID khóa hoặc bắt thằng bự nhất
        if locked_id in track_ids:
            best_idx = track_ids.index(locked_id); lost_counter = 0; found_target = True
        else:
            boxes = results[0].boxes.xywh.cpu().numpy()
            best_idx = np.argmax(boxes[:, 2] * boxes[:, 3]); locked_id = track_ids[best_idx]; found_target = True
            
        box = results[0].boxes.xywh[best_idx].cpu().numpy()
        cx_history.append(box[0]/frame.shape[1]); cy_history.append(box[1]/frame.shape[0])
        ratio_history.append(box[2] / box[3] if box[3] > 0 else 0)
        
        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xyn[best_idx].cpu().numpy()
            curr_ang = calculate_spine_angle_yolo(kpts)
            norm = normalize_keypoints(kpts, results[0].keypoints.conf[best_idx].cpu().numpy())
            if norm is not None: frame_queue.append(norm)
        
        cv2.putText(frame, f"Locked ID: {locked_id}", (int(box[0]-20), int(box[1]-20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        lost_counter += 1
        # Gồng ID (Duy trì khóa nếu mất dấu dưới 20 frame)
        if lost_counter < 20: found_target = True 
        elif lost_counter > MAX_LOST: 
            locked_id = None; curr_ang = prev_ang

    angle_history.append(curr_ang)

    # --- FORCE RESET: Đứng thẳng là reset ---
    if curr_ang < 20.0 and (fall_detected or suspected_fall):
        fall_detected, suspected_fall, fall_vote, block_new_fall = False, False, 0, False

    # --- LOGIC HYBRID VETO (LSTM) ---
    avg_ang = np.mean(angle_history) if angle_history else 0.0
    if avg_ang < CONFIG['RESET_ANGLE_THRESH']: block_new_fall = False
    
    pred, p2 = 0, 0.0
    if len(frame_queue) == CONFIG['WINDOW_SIZE'] and not fall_detected:
        with torch.no_grad():
            tensor = torch.tensor(np.array(frame_queue), dtype=torch.float32).unsqueeze(0).to(device)
            preds = torch.softmax(lstm_model(tensor), dim=1)
            pred, p2 = torch.argmax(preds, dim=1).item(), preds[0][2].item()
            
            is_leaning = (curr_ang > CONFIG['ANGLE_FALL']) or ((np.mean(ratio_history) if ratio_history else 0) > CONFIG['RATIO_FALL'])
            if not suspected_fall:
                if not block_new_fall and pred == 2 and p2 > CONFIG['CONFIDENCE_FALL'] and is_leaning:
                    fall_vote += 1
                else: fall_vote = max(0, fall_vote - 1)
                if fall_vote >= CONFIG['VOTE_THRESH']: suspected_fall = True; lying_frame_count = 0
            else:
                lying_frame_count += 1
                if lying_frame_count >= CONFIG['LYING_FRAMES_THRESH']: fall_detected = True

    # --- CẢNH BÁO ---
    if fall_detected:
        cv2.rectangle(frame, (0,0), (640,480), (0, 0, 255), 8)
        cv2.putText(frame, "!!! NGA BAT TINH !!!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        if time.time() - alarm_start_time > CONFIG['AUTO_RESET_TIME']: fall_detected = False

    cv2.imshow("Main System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()