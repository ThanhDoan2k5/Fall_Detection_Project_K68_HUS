import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

# --- 1. CẤU HÌNH ---
DATASET_PATH = r"C:\Du_An_Fall_Detection\dataset"
MODEL_PATH = "fall_detection_lstm.pth"
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 2. MODEL LSTM ---
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(FallDetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: [batch, 30, 34]
        out, _ = self.lstm(x)
        # Chỉ lấy output của frame cuối cùng trong chuỗi 30 frames
        out = self.fc(out[:, -1, :])
        return out

# --- 3. CHUẨN BỊ DỮ LIỆU ---
X = np.load(os.path.join(DATASET_PATH, "X_train.npy"))
y = np.load(os.path.join(DATASET_PATH, "y_train.npy"))

# Chuyển thành Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Khởi tạo model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FallDetectionLSTM(input_size=34, hidden_size=64, num_layers=2, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. VÒNG LẶP TRAINING ---
print(f"🚀 Bắt đầu train trên {device}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # In thông tin mỗi 10 epoch
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# --- 5. LƯU MODEL ---
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n💾 Đã lưu model thành công tại: {MODEL_PATH}")