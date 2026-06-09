# evaluate.py - BẢN XUẤT RA FILE CSV
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 1. ĐỌC FILE KẾT QUẢ CSV
csv_file = "test_results_v3.0_final.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"❌ Không tìm thấy file {csv_file}. Chạy main.py để tạo file trước nhé!")
    exit()

# Kiểm tra xem file có cột video_name không
if 'video_name' not in df.columns:
    print("❌ File CSV cũ quá, không có cột tên video. Chạy lại main.py bản mới nhất nhé!")
    exit()

# 2. TẠO MẢNG DỰ ĐOÁN (y_pred) TỪ AI
df['y_pred'] = df['state'].apply(lambda x: 1 if x == 'ALARM' else 0)

# 3. TẠO MẢNG THỰC TẾ (y_true)
def get_ground_truth(row):
    vid_name = str(row['video_name']).lower()
    if 'normal' in vid_name:
        return 0
    elif 'fall' in vid_name:
        return 1
    return 0

df['y_true'] = df.apply(get_ground_truth, axis=1)

y_true = df['y_true'].tolist()
y_pred = df['y_pred'].tolist()

# 4. TÍNH TOÁN CÁC CHỈ SỐ
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# 5. IN BÁO CÁO RA MÀN HÌNH (Terminal)
print("="*40)
print("🎯 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (EVALUATION)")
print("="*40)
print(f"✔️ Accuracy  (Độ chính xác) : {acc * 100:.2f} %")
print(f"🎯 Precision (Độ chuẩn xác) : {prec * 100:.2f} %")
print(f"🔍 Recall    (Độ bao phủ)   : {rec * 100:.2f} %")
print(f"🔥 F1-Score  (Điểm tổng hợp): {f1 * 100:.2f} %")
print("-" * 40)

# 6. LƯU BÁO CÁO RA FILE CSV (Phần mày yêu cầu)
report_data = {
    "Chỉ số Đánh giá (Metrics)": [
        "Accuracy (Độ chính xác tổng thể)", 
        "Precision (Độ chuẩn xác - Ít báo giả)", 
        "Recall (Độ bao phủ - Ít bỏ sót)", 
        "F1-Score (Điểm trung bình hài hòa)"
    ],
    "Kết quả (%)": [
        round(acc * 100, 2), 
        round(prec * 100, 2), 
        round(rec * 100, 2), 
        round(f1 * 100, 2)
    ]
}

report_df = pd.DataFrame(report_data)

# Dùng utf-8-sig để Excel mở ra không bị lỗi font tiếng Việt
report_filename = "evaluation_report.csv"
report_df.to_csv(report_filename, index=False, encoding='utf-8-sig')

print(f"\n✅ Đã lưu file báo cáo điểm số vào: {report_filename}")
print("   👉 Mày mở file này bằng Excel để copy vào báo cáo đồ án nhé!")
print("="*40)