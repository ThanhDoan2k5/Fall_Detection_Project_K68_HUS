import pandas as pd
import numpy as np
import os

def create_sequences(csv_path, window_size=30):
    df = pd.read_csv(csv_path)
    data = df.values # Mảng tọa độ + label
    
    X, y = [], []
    for i in range(len(data) - window_size):
        # 30 frames làm input
        sequence = data[i:i+window_size, :-1]
        # Label của frame cuối cùng làm target
        label = data[i+window_size-1, -1]
        
        X.append(sequence)
        y.append(label)
        
    return np.array(X), np.array(y)

# Chạy tạo sequence
X, y = create_sequences(r"C:\Du_An_Fall_Detection\dataset\processed_csv\final_dataset.csv")

# Lưu lại để train
np.save(r"C:\Du_An_Fall_Detection\dataset\X_train.npy", X)
np.save(r"C:\Du_An_Fall_Detection\dataset\y_train.npy", y)

print(f"✅ Đã tạo xong dataset: X {X.shape}, y {y.shape}")