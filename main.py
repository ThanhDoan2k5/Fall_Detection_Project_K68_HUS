# main.py - FINAL PRO VERSION + OCCLUSION RESTORED
import cv2, torch, time, winsound, csv, numpy as np 
import smtplib, threading
from email.mime.text import MIMEText
from collections import deque
from ultralytics import YOLO
from train_lstm import FallDetectionLSTM
import mediapipe as mp

from config import CONFIG
from camera_utils import ThreadedCamera
from ai_models import calculate_3d_spine_angle, calculate_spine_angle_yolo, normalize_keypoints

# ==========================================
# ⚙️ CẤU HÌNH EMAIL
# ==========================================
EMAIL_SENDER = "doanthanh2k5@gmail.com"        
EMAIL_PASSWORD = "xbko hvdd flix tjxy"    
EMAIL_RECEIVER = "quangthanhthanhthuy@gmail.com"
def send_email_async():
    msg = MIMEText("CẢNH BÁO KHẨN CẤP: Hệ thống camera vừa phát hiện có người ngã. Yêu cầu kiểm tra ngay lập tức!")
    msg['Subject'] = 'CẢNH BÁO: PHÁT HIỆN NGÃ (FALL DETECTED)'
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("📧 [THÀNH CÔNG] Đã gửi email cảnh báo!")
    except Exception as e:
        print(f"❌ [LỖI] Không thể gửi email: {e}")

# --- KHỞI TẠO AI ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8n-pose.pt')
lstm_model = FallDetectionLSTM(input_size=34, hidden_size=64, num_layers=2, num_classes=3)
lstm_model.load_state_dict(torch.load("fall_detection_lstm.pth", map_location=device))
lstm_model.to(device).eval()

# --- BIẾN TOÀN CỤC ---
frame_queue = deque(maxlen=CONFIG['WINDOW_SIZE'])
angle_history, fps_history = deque(maxlen=15), deque(maxlen=30)
cy_history = deque(maxlen=15) # KHÔI PHỤC LẠI BIẾN OCCLUSION NÀY

locked_id, lost_counter, MAX_LOST = None, 0, 60
fall_vote, suspected_fall, fall_detected, lying_frame_count = 0, False, False, 0
prev_ang, prev_time = 0.0, time.time()
last_beep_time, last_email_time = 0.0, 0.0

# File LOG
log_file = open(CONFIG['CSV_LOG'], 'a', newline='')
csv_writer = csv.writer(log_file)
if log_file.tell() == 0:
    csv_writer.writerow(['timestamp', 'pred', 'state', 'lying_frames', 'fps'])

cap = ThreadedCamera(CONFIG['IP_CAMERA'])
time.sleep(2)

print("🚀 Hệ thống PRO đã khôi phục Xử Lý Vật Cản (Occlusion)!")

while True:
    pred = 0 
    ret, frame = cap.read()
    if not ret or frame is None: continue
    
    # --- TRACKING ---
    results = yolo_model.track(frame, conf=0.5, persist=True, verbose=False)
    annotated_frame = results[0].plot()
    
    curr_ang = prev_ang
    occlusion_triggered = False # Cờ báo hiệu bị che khuất
    
    if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # ID Locking
        if locked_id in track_ids:
            best_idx = track_ids.index(locked_id); lost_counter = 0
        else:
            boxes = results[0].boxes.xywh.cpu().numpy()
            best_idx = np.argmax(boxes[:, 2] * boxes[:, 3]); locked_id = track_ids[best_idx]
            
        box = results[0].boxes.xywh[best_idx].cpu().numpy()
        
        # CẬP NHẬT TỌA ĐỘ Y LÊN CY_HISTORY (Cái này hồi nãy tao xóa nhầm)
        cy_history.append(box[1] / frame.shape[0])
        
        # Vẽ ID góc trái BBox
        x_tl, y_tl = int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)
        cv2.putText(annotated_frame, f"ID: {locked_id}", (x_tl, max(20, y_tl - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
        if results[0].keypoints is not None and results[0].keypoints.xyn.numel() > 0:
            kpts = results[0].keypoints.xyn[best_idx].cpu().numpy()
            curr_ang = calculate_spine_angle_yolo(kpts)
            norm = normalize_keypoints(kpts, results[0].keypoints.conf[best_idx].cpu().numpy())
            if norm is not None: frame_queue.append(norm)
        else:
            # Thấy Bounding box nhưng mất xương (Bị che nửa người)
            occlusion_triggered = True
    else:
        lost_counter += 1
        if lost_counter < 15: occlusion_triggered = True # Vừa mất dấu hoàn toàn
        if lost_counter > MAX_LOST: locked_id = None

    # --- KHÔI PHỤC LOGIC OCCLUSION FALLBACK ---
    if occlusion_triggered and len(cy_history) >= 5:
        v_y = cy_history[-1] - cy_history[-5] # Tính vận tốc rơi trục Y
        if v_y > 0.15 and not fall_detected:  # Rơi quá nhanh + bị che khuất
            fall_detected = True
            print("⚠️ [OCCLUSION] Phát hiện ngã sau vật cản!")

    # --- LOGIC NGÃ (LSTM) ---
    angle_history.append(curr_ang)
    if len(frame_queue) == CONFIG['WINDOW_SIZE'] and not fall_detected:
        with torch.no_grad():
            tensor = torch.tensor(np.array(frame_queue), dtype=torch.float32).unsqueeze(0).to(device)
            preds = torch.softmax(lstm_model(tensor), dim=1)
            pred = torch.argmax(preds, dim=1).item()
            
            if pred == 2: fall_vote = min(fall_vote + 2, 10)
            else: fall_vote = max(0, fall_vote - 1)
            
            if fall_vote >= CONFIG['VOTE_THRESH']: suspected_fall = True
            if suspected_fall:
                lying_frame_count += 1
                if lying_frame_count >= CONFIG['LYING_FRAMES_THRESH']: 
                    fall_detected = True
                    
    # --- RESET KHI ĐỨNG THẲNG ---
    if curr_ang < 20.0 and (fall_detected or suspected_fall):
        fall_detected, suspected_fall, fall_vote, lying_frame_count = False, False, 0, 0

    # --- HIỂN THỊ FPS ---
    fps = 1.0 / (time.time() - prev_time + 1e-6); prev_time = time.time()
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    # --- XỬ LÝ CẢNH BÁO (BEEP & EMAIL) ---
    if fall_detected:
        cv2.rectangle(annotated_frame, (0,0), (640,480), (0, 0, 255), 8)
        cv2.putText(annotated_frame, "!!! PHAT HIEN NGA !!!", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
        
        if time.time() - last_beep_time > 0.5:
            winsound.Beep(2000, 200)
            last_beep_time = time.time()
            
        if time.time() - last_email_time > 60.0:
            print("⏳ Đang gửi email cảnh báo...")
            threading.Thread(target=send_email_async, daemon=True).start()
            last_email_time = time.time()

    cv2.imshow("Main System", annotated_frame)
    csv_writer.writerow([time.time(), pred, ('ALARM' if fall_detected else 'NORMAL'), lying_frame_count, fps])
    log_file.flush()

    if cv2.waitKey(1) & 0xFF == ord('q'): break

log_file.close(); cap.release(); cv2.destroyAllWindows()