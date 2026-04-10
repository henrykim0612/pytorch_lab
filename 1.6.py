import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Model
model = nn.Sequential(nn.Linear(1, 1))

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 데이터
xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)

# 훈련
for _ in range(500):
    optimizer.zero_grad()
    outputs = model(xs)
    loss = criterion(outputs, ys)
    loss.backward()
    optimizer.step()
    print(loss)

# 예측
with torch.no_grad():
    print(model(torch.tensor([[10.0]], dtype=torch.float32)))

# 모델의 첫 번째(그리고 유일한) 층을 얻습니다.
layer = model[0]
# 가중치와 절편을 추출합니다.
weight = layer.weight.data.numpy()
bias = layer.bias.data.numpy()
print(f"Weight: {weight}, Bias: {bias}") # 기대한 y = 2x - 1 과 유사한 값이 출력됨