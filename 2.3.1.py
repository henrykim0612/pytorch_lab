import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 데이터셋 로드
transform = transforms.Compose([transforms.ToTensor()])

# datasets.FashionMNIST 은 60,000개의 샘플을 갖고 있지만, 60,000개를 신경망 훈련에 사용하고 남은 10,000개는 테스트에 사용할 수 있다(train=True 또는 False 로).
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

# 28x28 이미지여서 64로
# batch_size는 컴퓨터 성능에 따라 타협점을 찾아야 함.
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 모델 정의
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10), # out_features 가 10인 이유는 이 데이터의 레이블이 10개여서.
            nn.LogSoftmax(dim=1) # 각 이미지마다 10개 클래스 중 하나를 예측해야 하므로 dim=1
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = FashionMNISTModel()

# 손실 함수와 옵티마이저 정의
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

# 정확도 계산 함수
def get_accuracy(pred, labels):
    _, predictions = torch.max(pred, 1)
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy

# 모델 훈련 함수
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, total_accuracy = 0, 0

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # 예측과 손실 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = get_accuracy(pred, y)

        # 역전파: 시험을 봤는데 틀린 문제만 보는 게 아니라, 어떤 개념 이해가 부족해서 틀렸는지를 거슬러 올라가 찾아내는 과정
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        if batch % 100 == 0:
            current = batch * len(X)
            avg_loss = total_loss / (batch + 1)
            avg_accuracy = total_accuracy / (batch + 1) * 100
            print(f"Batch {batch}, Loss: {avg_loss:>7f}, Accuracy: {avg_accuracy:>0.2f}% [{current:>5d}/{size:>5d}]")

        # 조기 종료 조건
        if avg_accuracy >= 95:
            print(f"95% 정확도에 도달했으므로 훈련을 중지합니다.")
            return True

# Training process
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    if train(train_loader, model, loss_function, optimizer):  # Check for the early stopping signal
        print("Early stopping triggered.")
        break
print("Done!")

# 모델 테스트 함수
def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 모델을 평가 모드(추론 모드)로 지정
    model.eval()

    test_loss, correct = 0, 0
    # 학습 시: gradient 필요 → torch.no_grad() 쓰면 안 됨
    # 검증/테스트 시: gradient 불필요 → torch.no_grad() 사용
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"테스트 오차: \n 정확도: {(100*correct):>0.1f}%, 평균 손실: {test_loss:>8f} \n")

# 모델 평가
test(test_loader, model)

# 이미지 하나에 대한 예측을 만드는 함수 정의
def predict_single_image(image, label, model):
    # 모델을 평가 모드로 전환
    model.eval()

    # 모델이 기대하는 배치 차원을 추가
    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model(image)
        print(prediction)
        predicted_label = prediction.argmax(1).item()

    # 이미지와 예측 결과를 출력
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Predicted: {predicted_label}, Actual: {label}")
    plt.show()

    return predicted_label

# 테스트 세트에 있는 이미지 하나를 선택
image, label = test_dataset[0] # 인덱스를 바꾸어 다른 이미지로 테스트 가능

# 선택한 이미지의 레이블을 예측
predicted_label = predict_single_image(image, label, model)
print(f"모델의 예측: {predicted_label}, 실제 레이블: {label}")


