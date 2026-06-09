# main.py - BẢN MULTI-VIDEO TỰ ĐỘNG XÓA LOG CŨ
import cv2, torch, time, winsound, csv, numpy as np 
import smtplib, threading
from email.mime.text import MIMEText
from collections import deque
from ultralytics import YOLO
from train_lstm import FallDetectionLSTM
import mediapipe as mp

from config import CONFIG
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
    except Exception as e:
        pass 

# --- KHỞI TẠO AI ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO('yolov8n-pose.pt')
lstm_model = FallDetectionLSTM(input_size=34, hidden_size=64, num_layers=2, num_classes=3)
lstm_model.load_state_dict(torch.load("fall_detection_lstm.pth", map_location=device))
lstm_model.to(device).eval()

# --- FILE LOG (ĐÃ CHUYỂN SANG 'w' ĐỂ TỰ ĐỘNG XÓA FILE CŨ MỖI KHI CHẠY) ---
log_file = open(CONFIG['CSV_LOG'], 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['video_name', 'timestamp', 'pred', 'state', 'lying_frames', 'fps']) # Ghi ngay header chuẩn

# ==========================================
# 🎥 DANH SÁCH VIDEO TEST
# ==========================================
video_paths = [
    "normal_01.mp4", 
    "fall_nghieng_01.mp4"
]

print(f"🚀 Bắt đầu chạy chế độ ĐÁNH GIÁ theo chuỗi ({len(video_paths)} video)...")
alarm_triggered_count = 0
total_fps_history = []

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ BỎ QUA: Không tìm thấy file '{video_path}'")
        continue
        
    print(f"\n▶️ ĐANG CHẠY VIDEO: {video_path}")
    
    frame_queue = deque(maxlen=CONFIG['WINDOW_SIZE'])
    angle_history, fps_history = deque(maxlen=15), deque(maxlen=30)
    cy_history = deque(maxlen=15) 
    locked_id, lost_counter = None, 0
    fall_vote, suspected_fall, fall_detected, lying_frame_count = 0, False, False, 0
    prev_ang, prev_time = 0.0, time.time()
    last_beep_time, last_email_time = 0.0, 0.0
    MAX_LOST = 60

    while True:
        pred = 0 
        ret, frame = cap.read()
        
        if not ret or frame is None: 
            break
            
        frame = cv2.resize(frame, (640, 480))
        results = yolo_model.track(frame, conf=0.5, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        
        curr_ang = prev_ang
        occlusion_triggered = False 
        
        if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            if locked_id in track_ids:
                best_idx = track_ids.index(locked_id); lost_counter = 0
            else:
                boxes = results[0].boxes.xywh.cpu().numpy()
                best_idx = np.argmax(boxes[:, 2] * boxes[:, 3]); locked_id = track_ids[best_idx]
                
            box = results[0].boxes.xywh[best_idx].cpu().numpy()
            cy_history.append(box[1] / frame.shape[0])
            
            x_tl, y_tl = int(box[0] - box[2] / 2), int(box[1] - box[3] / 2)
            cv2.putText(annotated_frame, f"ID: {locked_id}", (x_tl, max(20, y_tl - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
            if results[0].keypoints is not None and results[0].keypoints.xyn.numel() > 0:
                kpts = results[0].keypoints.xyn[best_idx].cpu().numpy()
                curr_ang = calculate_spine_angle_yolo(kpts)
                norm = normalize_keypoints(kpts, results[0].keypoints.conf[best_idx].cpu().numpy())
                if norm is not None: frame_queue.append(norm)
            else:
                occlusion_triggered = True
        else:
            lost_counter += 1
            if lost_counter < 15: occlusion_triggered = True 
            if lost_counter > MAX_LOST: locked_id = None

        if occlusion_triggered and len(cy_history) >= 5:
            v_y = cy_history[-1] - cy_history[-5] 
            if v_y > 0.15 and not fall_detected:  
                fall_detected = True

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
                        alarm_triggered_count += 1
                        
        if curr_ang < 20.0 and (fall_detected or suspected_fall):
            fall_detected, suspected_fall, fall_vote, lying_frame_count = False, False, 0, 0

        fps = 1.0 / (time.time() - prev_time + 1e-6); prev_time = time.time()
        fps_history.append(fps); total_fps_history.append(fps)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"File: {video_path}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if fall_detected:
            cv2.rectangle(annotated_frame, (0,0), (640,480), (0, 0, 255), 8)
            cv2.putText(annotated_frame, "!!! PHAT HIEN NGA !!!", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            if time.time() - last_beep_time > 0.5:
                winsound.Beep(2000, 200)
                last_beep_time = time.time()
            if time.time() - last_email_time > 60.0:
                threading.Thread(target=send_email_async, daemon=True).start()
                last_email_time = time.time()

        cv2.imshow("Main System", annotated_frame)
        
        # LOG GHI CẢ TÊN FILE
        csv_writer.writerow([video_path, time.time(), pred, ('ALARM' if fall_detected else 'NORMAL'), lying_frame_count, fps])
        log_file.flush()

        if cv2.waitKey(30) & 0xFF == ord('q'): 
            break

    cap.release()

print(f"\n📊 SUMMARY TẤT CẢ VIDEO: {alarm_triggered_count} lần báo ngã")
log_file.close()
cv2.destroyAllWindows()