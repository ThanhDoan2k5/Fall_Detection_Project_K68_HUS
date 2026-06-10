Dataset: https://drive.google.com/drive/folders/1BLSNlh_P592D79rmNt_TbWAb6OyapPFJ?usp=sharing
được tham khảo ở: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
# Hệ Thống AI Phát Hiện Té Ngã Dựa Trên Thị Giác Máy Tính (Computer Vision Based Fall Detection System)

- Hệ thống phát hiện té ngã theo thời gian thực (Real-time Fall Detection) sử dụng trí tuệ nhân tạo và thị giác máy tính từ một camera RGB duy nhất. Dự án xây dựng một giải pháp lai nâng cao (Hybrid Model) kết hợp giữa các mô hình học sâu hiện đại và các quy tắc hình học không gian  nhằm tối ưu hóa độ chính xác, xử lý che khuất và triệt tiêu tối đa tỷ lệ báo động giả trong môi trường giám sát thực tế.

---

##  Tính Năng Chính

* **Phát hiện & Ước lượng tư thế một giai đoạn:** Sử dụng mô hình cấu trúc **YOLOv8-Pose** để nhận diện thực thể người và trích xuất đồng thời ma trận tọa độ khung xương 17 điểm khớp chuẩn COCO trong một lần suy luận duy nhất
* **Định danh và bám vết đa đối tượng (Multi-Object Tracking):** Tích hợp thuật toán **ByteTrack** để duy trì mã số định danh (ID) liên tục cho từng người , cô lập nhiễu động học từ xung quanh và loại bỏ hoàn toàn nguy cơ hoán đổi dữ liệu đa mục tiêu.
* **Nội suy không gian 3D độc lập góc nhìn:** Khai thác mô hình **MediaPipe Pose 3D** để bổ sung thông tin chiều sâu (Z), tính toán góc nghiêng trục cột sống nhằm khắc phục nhược điểm mất dấu hình thái của các phương pháp 2D truyền thống khi đối tượng ngã trực diện.
* **Khối xử lý "Gồng ID" khi bị che khuất (Occlusion Handling):** Cơ chế xử lý thông minh sử dụng bộ đệm lịch sử kết hợp mạng **LSTM** và cửa sổ trượt , cho phép hệ thống "gồng gánh" dữ liệu liên tục lên tới 20 khung hình khi đối tượng bị khuất sau bàn ghế, giường ngủ hoặc vật cản.
* **Chống báo động giả đa tầng (Multi-stage False Alarm Filtering):** Hệ thống chốt chặn nghiêm ngặt qua tỷ lệ khung bao cơ thể (Body Aspect Ratio) , thuật toán tích lũy phiếu bầu (Voting Accumulation) và phân tích trạng thái nằm im bất động sau va chạm.
* **Cảnh báo đa kênh tức thời:** Kích hoạt đồng thời viền đỏ giao diện (GUI visual alert), còi hú hệ thốngvà **tự động gửi email cảnh báo (Gmail API)** đính kèm thông tin sự kiện đến người thân/nhân viên trực ca ngay trong "thời gian vàng".

---

##  Công Nghệ Sử Dụng

* **Ngôn ngữ chính:** Python 3.x
* **Thị giác máy tính & Đồ họa:** `OpenCV` (cv2) 
* **Mô hình AI cốt lõi:** `YOLOv8-Pose` (Ultralytics) , `MediaPipe` (Google) , `ByteTrack`
* **Học sâu chuỗi thời gian:** `PyTorch` hoặc `TensorFlow/Keras` (Mạng LSTM) 
* **Xử lý dữ liệu & Lưu trữ nhật ký:** `numpy`, `pandas`, `csv` 

---

##  Nguyên Lý Hoạt Động

Hệ thống hoạt động theo mô hình luồng dữ liệu tuần tự tích hợp (**Sequential Data Pipeline Hybrid Model**) qua các giai đoạn chính:
- Trích xuất đặc trưng & chuẩn hoá dữ liệu
- Bộ lọc chốt chặ hình học và phân loại hành vi



---

##  Chỉ Số Đánh Giá Mô Hình

Hiệu suất tổng thể của hệ thống phân loại hành vi lai được nghiệm thu định lượng trên tập dữ liệu thử nghiệm thực tế đạt kết quả ấn tượng:

| Chỉ Số Đánh Giá (Metrics) | Kết Quả Thực Nghiệm (%) |
| :--- | :---: |
| **Độ chính xác tổng thể (Accuracy)** | **84.57%**  |
| **Độ chính xác dự đoán (Precision - Hạn chế báo giả)** | **85.77%**  |
| **Khả năng phát hiện (Recall - Tránh bỏ sót nạn nhân)** | **84.23%**  |
| **F1-Score (Cân bằng động)** | **84.99%**  |

---

##  Hướng Dẫn Cài Đặt & Sử Dụng

###  Chuẩn bị môi trường
Cài đặt các thư viện phụ thuộc cần thiết cho luồng xử lý dữ liệu và mô hình AI. Mở terminal và chạy lệnh:

```bash
pip install ultralytics torch mediapipe numpy opencv-python pandas scikit-learn
