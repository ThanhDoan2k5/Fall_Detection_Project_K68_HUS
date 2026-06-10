Dataset: https://drive.google.com/drive/folders/1BLSNlh_P592D79rmNt_TbWAb6OyapPFJ?usp=sharing
được tham khảo ở: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
# Hệ thống Phát hiện Té ngã dựa trên Thị giác Máy tính (Real-time Fall Detection System)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLOv8%20%7C%20MediaPipe-orange)](https://github.com/ultralytics/ultralytics)
[![Academic Project](https://img.shields.io/badge/HUS--VNU-Introduction%20to%20AI-red)](https://hus.vnu.edu.vn/)

[cite_start]Sản phẩm nghiên cứu khoa học thuộc môn học **Nhập môn Trí tuệ Nhân tạo** - Nhóm 9 - Khoa Vật lý, Trường Đại học Khoa học Tự nhiên, Đại học Quốc gia Hà Nội (VNU-HUS)[cite: 1].

## 👥 Thành viên thực hiện
* [cite_start]**Đoàn Quang Thành** (Mã SV: 23001628) - Phát triển Code, Thuyết trình, Nội dung[cite: 2, 12].
* [cite_start]**Vũ Thanh Tâm** (Mã SV: 23001626) - Phát triển Code, Viết báo cáo[cite: 2, 12].
* [cite_start]**Nguyễn Tuấn Vũ** (Mã SV: 23001639) - Nghiên cứu nội dung, Viết báo cáo[cite: 2, 12].
* [cite_start]**Lưu Minh Tuân** (Mã SV: 23001637) - Nghiên cứu nội dung, Thiết kế Slide, Viết báo cáo[cite: 2, 12].

**Giáo viên hướng dẫn:** TS. Nguyễn Tiến Cường & CN. [cite_start]Vi Anh Quân[cite: 1, 2].

---

## 📌 Giới thiệu đề tài
[cite_start]Té ngã là một trong những tai nạn phổ biến và nguy hiểm nhất đối với người cao tuổi hoặc bệnh nhân sống một mình[cite: 16, 18]. [cite_start]Đề tài này xây dựng một giải pháp toàn diện cho bài toán **Phát hiện té ngã thời gian thực (Real-time Fall Detection)** sử dụng duy nhất **một camera RGB** thông thường[cite: 28, 29]. 

[cite_start]Hệ thống kết hợp sức mạnh của các mô hình học sâu hiện đại phục vụ việc trích xuất khung xương, định danh đối tượng và mạng nơ-ron chuỗi thời gian nhằm nhận diện hành vi động học, đồng thời tích hợp các thuật toán chốt chặn hình học giúp tối ưu hóa tốc độ xử lý và giảm thiểu báo động giả[cite: 28, 220].


## 🛠️ Kiến trúc hệ thống (Data Pipeline)
[cite_start]Hệ thống vận hành theo một luồng dữ liệu tuần tự khép kín (Sequential Data Pipeline) bao gồm các giai đoạn[cite: 219]:
1.  [cite_start]**Thu nhận & Chuẩn hóa đầu vào:** Sử dụng kỹ thuật đa luồng `ThreadedCamera` để tối ưu FPS, tự động giải phóng bộ đệm hình ảnh và chuẩn hóa kích thước khung hình về $640 \times 480$[cite: 223, 225].
2.  [cite_start]**Phát hiện & Trích xuất Khung xương 2D:** Sử dụng mô hình một giai đoạn **YOLOv8-Pose** để trích xuất 17 điểm mốc cơ thể theo chuẩn COCO[cite: 74, 495, 501].
3.  [cite_start]**Định danh & Theo dõi đối tượng:** Thuật toán **ByteTrack** giúp duy trì ID nhất quán cho từng người, cô lập dữ liệu và chống nhiễu đa đối tượng[cite: 75, 510, 516].
4.  [cite_start]**Bổ sung thông tin không gian 3D:** Sử dụng **MediaPipe Pose 3D** để tính toán góc nghiêng cột sống trong không gian 3 chiều, giúp hệ thống không phụ thuộc vào góc đặt camera (đặc biệt là lỗi ngã trực diện)[cite: 76, 77, 137].
5.  [cite_start]**Phân loại hành vi chuỗi thời gian:** Sử dụng mạng **LSTM** kết hợp cơ chế cửa sổ trượt (Sliding Window 30 frames) để phân tích hành vi động học của cú ngã[cite: 78, 272].
6.  [cite_start]**Bộ lọc chống báo động giả đa tầng:** Xác minh trạng thái nằm bất động thông qua bộ đếm thời gian trễ quyết định (`lying_frame_count >= 45 frames`, tương đương ~1.5 giây bất động thực tế)[cite: 201].

---

## ⚡ Các tính năng cốt lõi
* [cite_start]**Xử lý thời gian thực vượt trội:** Pipeline tối ưu hóa giải phóng CPU/GPU, cho tốc độ xử lý mượt mà trên camera RGB đơn[cite: 29, 497].
* [cite_start]**Thích ứng hiện tượng che khuất (Occlusion):** Cơ chế lai (Hybrid Architecture) kết hợp bộ nhớ đệm đóng băng trạng thái của ByteTrack giúp hệ thống giữ vững luồng phân tích khi nạn nhân bị đồ đạc, bàn ghế che khuất một phần hoặc mất dấu ngắn hạn (< 2 giây)[cite: 123, 142].
* [cite_start]**Cảnh báo đa kênh thông minh:** Khi phát hiện té ngã bất tỉnh thực sự, hệ thống kích hoạt đồng thời[cite: 244, 245]:
    * **Visual Alert:** Hiển thị viền đỏ bao quanh khung giao diện kèm dòng chữ cảnh báo trực quan `!!! [cite_start]NGA BAT TINH !!!`[cite: 245].
    * [cite_start]**Audio Alert:** Phát còi hú báo động y tế tần số cao (2000Hz) ra loa máy tính thông qua thư viện `winsound`.
    * [cite_start]**Remote Notification:** Tự động gửi Email thông báo khẩn cấp (kèm ID, thời gian tai nạn) trực tiếp tới người thân hoặc bác sĩ trực ca thông qua giao thức SMTP[cite: 248].
* [cite_start]**Cơ chế tự động giải trừ (Auto-Reset):** Hệ thống sẽ tự khôi phục về trạng thái giám sát an toàn nếu nạn nhân tự đứng dậy và góc cột sống phục hồi về mức tiêu chuẩn[cite: 250, 280].

---

## 📊 Kết quả thực nghiệm
[cite_start]Mô hình phân loại hành vi té ngã đạt được độ chính xác cao dựa trên các chỉ số kiểm thử định lượng tổng thể[cite: 218, 282]:

| Chỉ số đánh giá (Metrics) | Kết quả thực nghiệm (%) |
| :--- | :---: |
| **Độ chính xác tổng thể (Accuracy)** | [cite_start]**84.57%** [cite: 282] |
| **Độ chính xác dự đoán (Precision)** | [cite_start]**85.77%** [cite: 282] |
| **Khả năng phát hiện / Tránh bỏ sót (Recall)** | [cite_start]**84.23%** [cite: 282] |
| **F1-Score** | [cite_start]**84.99%** [cite: 282] |

---

## 📂 Cấu trúc thư mục mã nguồn tham khảo
```text
├── models/
│   ├── yolov8n-pose.pt          # Trọng số mô hình YOLOv8-Pose đã pre-train [cite: 227]
│   └── fall_detection_lstm.pt   # Trọng số mô hình mạng LSTM phân loại chuỗi thời gian
├── src/
│   ├── config.py                # Cấu hình ngưỡng góc nghiêng, tỷ lệ hộp bao, frame trễ [cite: 198, 199]
│   ├── utils_3d.py              # Hàm tính toán vector cột sống và góc nghiêng 3D [cite: 137, 163]
│   ├── mail_notifier.py         # Mô-đun cấu hình SMTP tự động gửi thư cảnh báo [cite: 248]
│   └── tracker.py               # Thuật toán ByteTrack định danh đối tượng [cite: 75]
├── main.py                      # File khởi chạy chính tích hợp Data Pipeline thời gian thực [cite: 141, 283]
├── test_results_final.csv       # Nhật ký hệ thống tự động ghi lại dữ liệu (Log data) [cite: 279]
├── NhapmonAI_BaoCao.pdf         # Báo cáo chi tiết đề tài dự án [cite: 1]
└── README.md                    # Hướng dẫn chi tiết hệ thống
