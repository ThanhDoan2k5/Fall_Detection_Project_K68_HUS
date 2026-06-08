import cv2
import torch
import numpy as np
import csv
import time
import winsound
import threading
from collections import deque
from ultralytics import YOLO
from train_lstm import FallDetectionLSTM
import mediapipe as mp

# --- CONFIGURATION (BẢN V2.9 - HYBRID VETO FINAL) ---
CONFIG = {
    'IP_CAMERA': "http://192.168.1.100:8080/video", # Sửa lại IP nếu điện thoại nhảy IP mới
    'WINDOW_SIZE': 30,
    'VOTE_THRESH': 4,                
    'CONFIDENCE_FALL': 0.60,         
    'RATIO_FALL': 0.7,               # Ngưỡng tỷ lệ Bounding Box (R = W/H). Dùng cứu hộ ngã trực diện
    'ANGLE_FALL': 38.0,              # GÓC NGƯỠNG: Dưới góc này tuyệt đối cấm báo ngã
    'MIN_KP_CONF': 0.3,
    'LYING_FRAMES_THRESH': 45,       
    'MOVE_WINDOW_THRESH': 0.20,      
    'ALARM_ANGLE_THRESH': 50.0,      
    'ALARM_RATIO_THRESH': 0.85,      
    'RESET_ANGLE_THRESH': 32.0,      
    'AUTO_RESET_TIME': 4.0,          
    'MAX_ANGULAR_VELOCITY': 25.0,    
    'CSV_LOG': 'test_results_v2.9_final.csv'
}

# --- CLASS THREADED CAMERA (TỐI ƯU FPS, CHỐNG GIẬT LAG VIDEO) ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.stopped = False
        
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                self.ret, self.frame = self.capture.read()
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame
        
    def isOpened(self):
        return self.capture.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

# --- HÀM HỖ TRỢ XỬ LÝ POSE ---
mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5)

def calculate_3d_spine_angle(landmarks):
    S = np.array([(landmarks[11].x + landmarks[12].x)/2.0, (landmarks[11].y + landmarks[12].y)/2.0, (landmarks[11].z + landmarks[12].z)/2.0])
    H = np.array([(landmarks[23].x + landmarks[24].x)/2.0, (landmarks[23].y + landmarks[24].y)/2.0, (landmarks[23].z + landmarks[24].z)/2.0])
    SH = S - H
    angle_deg = np.degrees(np.arccos(np.clip(np.dot(SH, np.array([0,1,0])) / (np.linalg.norm(SH) + 1e-6), -1.0, 1.0)))
    return abs(angle_deg - 180)

def normalize_keypoints(kpts, confs):
    if confs[11] > CONFIG['MIN_KP_CONF'] and confs[12] > CONFIG['MIN_KP_CONF']:
        hip_x = (kpts[11][0] + kpts[12][0]) / 2.0
        hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
        norm = kpts.copy()
        for i in range(17): 
            norm[i][0] -= hip_x
            norm[i][1] -= hip_y
        return norm.flatten()
    return None

# --- KHỞI TẠO AI MODELS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8n-pose.pt')
lstm_model = FallDetectionLSTM(input_size=34, hidden_size=64, num_layers=2, num_classes=3)
lstm_model.load_state_dict(torch.load("fall_detection_lstm.pth", map_location=device))
lstm_model.to(device).eval()

# --- KHAI BÁO BIẾN TOÀN CỤC ---
frame_queue = deque(maxlen=CONFIG['WINDOW_SIZE'])
angle_history = deque(maxlen=15)
ratio_history = deque(maxlen=15)
fps_history = deque(maxlen=30)
cx_history = deque(maxlen=15)
cy_history = deque(maxlen=15)

fall_vote = 0
suspected_fall = False
fall_detected = False
block_new_fall = False
lying_frame_count = 0  
prev_ang = 0.0

prev_time = time.time()
last_log_time = time.time()
last_beep_time = 0.0 
alarm_start_time = 0.0

log_file = open(CONFIG['CSV_LOG'], 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['timestamp', 'pred', 'p2', 'spine_angle', 'vote', 'state', 'lying_frames', 'block_status', 'fps'])

# --- BẮT ĐẦU KẾT NỐI CAMERA ---
print(f"📡 Đang kết nối tới IP Camera: {CONFIG['IP_CAMERA']} ...")
cap = ThreadedCamera(CONFIG['IP_CAMERA'])
time.sleep(2)

if not cap.isOpened():
    print("❌ LỖI: Không thể kết nối. Kiểm tra lại IP hoặc WiFi!")
    exit()

print("🚀 v2.9 FINAL HYBRID - Hoàn thiện 100%. Sẵn sàng bảo vệ đồ án!")

# --- VÒNG LẶP XỬ LÝ CHÍNH ---
while True:
    ret, frame = cap.read()
    if not ret or frame is None: 
        continue
    
    # Ép size ảnh nhỏ lại để AI chạy nhanh hơn
    frame = cv2.resize(frame, (640, 480))
    frame_h, frame_w = frame.shape[0], frame.shape[1]

    # 1. MediaPipe: Lấy góc cột sống
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = pose_estimator.process(frame_rgb)
    curr_ang = calculate_3d_spine_angle(mp_results.pose_world_landmarks.landmark) if mp_results.pose_world_landmarks else prev_ang
    angle_history.append(curr_ang)

    # 2. YOLOv8: Lấy Bounding Box & Keypoints (Chạy FP16 nếu có GPU)
    use_half = True if device == 'cuda' else False
    results = yolo_model(frame, conf=0.5, half=use_half, verbose=False)
    current_cx, current_cy = 0.0, 0.0

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes.xywh[0].cpu().numpy()
        current_cx = box[0] / frame_w
        current_cy = box[1] / frame_h
        ratio_history.append(box[2] / box[3] if box[3] > 0 else 0)
        
        if current_cx != 0:
            cx_history.append(current_cx)
            cy_history.append(current_cy)

        if results[0].keypoints is not None and results[0].keypoints.xyn.numel() > 0:
            kpts = results[0].keypoints.xyn[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()
            norm = normalize_keypoints(kpts, confs)
            if norm is not None: frame_queue.append(norm)

    # Giải phóng khóa block khi người dùng đã đứng thẳng dậy
    avg_ang = np.mean(angle_history) if angle_history else 0.0
    if avg_ang < CONFIG['RESET_ANGLE_THRESH']:
        block_new_fall = False

    # 3. MẠNG LSTM & THUẬT TOÁN LAI (HYBRID LOGIC)
    pred, p2 = 0, 0.0
    if len(frame_queue) == CONFIG['WINDOW_SIZE'] and not fall_detected:
        with torch.no_grad():
            tensor = torch.tensor(np.array(frame_queue), dtype=torch.float32).unsqueeze(0).to(device)
            probs = torch.softmax(lstm_model(tensor), dim=1)
            pred, p2 = torch.argmax(probs, dim=1).item(), probs[0][2].item()
            
            avg_ratio = np.mean(ratio_history) if ratio_history else 0
            is_rapid_motion = abs(curr_ang - prev_ang) > CONFIG['MAX_ANGULAR_VELOCITY']
            
            # --- VŨ KHÍ TỐI THƯỢNG: HYBRID VETO LỌC ĐIỂM MÙ ---
            # Cơ thể ngã ngang (Góc > 38) HOẶC ngã trực diện làm khung hình bẹp (Tỷ lệ R > 0.7)
            is_leaning = (curr_ang > CONFIG['ANGLE_FALL']) or (avg_ratio > CONFIG['RATIO_FALL'])
            
            if not suspected_fall:
                # ÉP ĐIỀU KIỆN: AI đoán ngã (pred=2) nhưng phải thỏa mãn luật Vật lý (is_leaning)
                if not block_new_fall and pred == 2 and p2 > CONFIG['CONFIDENCE_FALL'] and not is_rapid_motion and is_leaning:
                    fall_vote += 1
                else:
                    fall_vote = max(0, fall_vote - 1)
                
                if fall_vote >= CONFIG['VOTE_THRESH']:
                    suspected_fall = True
                    lying_frame_count = 0
                    cx_history.clear()
                    cy_history.clear()
            else:
                # Trạng thái nghi vấn (Đang xác minh)
                lying_frame_count += 1
                
                if time.time() - last_beep_time > 0.5:
                    winsound.Beep(1000, 80)
                    last_beep_time = time.time()
                
                is_actively_moving = False
                if len(cx_history) == cx_history.maxlen:
                    move_range_x = max(cx_history) - min(cx_history)
                    move_range_y = max(cy_history) - min(cy_history)
                    if move_range_x > CONFIG['MOVE_WINDOW_THRESH'] or move_range_y > CONFIG['MOVE_WINDOW_THRESH']:
                        is_actively_moving = True
                
                if is_actively_moving:
                    suspected_fall = False
                    fall_vote = 0
                    block_new_fall = True
                    lying_frame_count = 0
                elif lying_frame_count >= CONFIG['LYING_FRAMES_THRESH']:
                    if avg_ang > CONFIG['ALARM_ANGLE_THRESH'] or avg_ratio > CONFIG['ALARM_RATIO_THRESH']:
                        fall_detected = True
                        suspected_fall = False
                        alarm_start_time = time.time()
                    else:
                        suspected_fall = False
                        fall_vote = 0
                        block_new_fall = True
                        lying_frame_count = 0

    prev_ang = curr_ang

    # 4. CẢNH BÁO ALARM & UI
    if fall_detected:
        cv2.rectangle(frame, (0,0), (frame_w,frame_h), (0, 0, 255), 8)
        cv2.putText(frame, "!!! NGA BAT TINH !!!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
        
        if time.time() - last_beep_time > 0.3:
            winsound.Beep(2000, 250)
            last_beep_time = time.time()
        
        # Tự reset sau 4s
        if time.time() - alarm_start_time > CONFIG['AUTO_RESET_TIME']:
            fall_detected = False
            suspected_fall = False
            fall_vote = 0
            block_new_fall = True
            lying_frame_count = 0

    fps_history.append(1.0 / (time.time() - prev_time + 1e-6))
    prev_time = time.time()

    # Vẽ thông số lên màn hình
    cv2.putText(frame, f"FPS: {np.mean(fps_history):.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"3D Angle: {avg_ang:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Block Status: {'LOCKED' if block_new_fall else 'READY'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(frame, f"pred:{pred} p2:{p2:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if suspected_fall:
        cv2.rectangle(frame, (0,0), (frame_w,frame_h), (0, 165, 255), 4)
        cv2.putText(frame, f"Xac minh: {lying_frame_count}/{CONFIG['LYING_FRAMES_THRESH']} frames", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Ghi log CSV
    if time.time() - last_log_time >= 1.0:
        state_str = "ALARM" if fall_detected else ("SUSPECT" if suspected_fall else "NORMAL")
        csv_writer.writerow([time.time(), pred, p2, avg_ang, fall_vote, state_str, lying_frame_count, int(block_new_fall), np.mean(fps_history)])
        log_file.flush()
        last_log_time = time.time()

    cv2.imshow("v2.9 HYBRID VETO FINAL", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'): # Bấm 'r' để reset thủ công nếu cần
        fall_detected = False; suspected_fall = False; fall_vote = 0; block_new_fall = False; lying_frame_count = 0
        frame_queue.clear(); angle_history.clear(); ratio_history.clear()
    if key == ord('q'): # Bấm 'q' để thoát
        break

log_file.close()
if cap: cap.release()
cv2.destroyAllWindows()