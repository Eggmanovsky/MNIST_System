import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 第一层全连接层，输入展平的28*28图像
        self.fc2 = nn.Linear(128, 64)     # 第二层全连接层
        self.fc3 = nn.Linear(64, 10)      # 输出层，包含10类

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将图像展平为一维向量
        x = F.relu(self.fc1(x))  # 通过第一层全连接并应用ReLU激活函数
        x = F.relu(self.fc2(x))  # 通过第二层全连接并应用ReLU激活函数
        x = self.fc3(x)          # 通过输出层
        return F.log_softmax(x, dim=1)

def get_data_loader(is_train):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST("", is_train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=15, shuffle=True)

def evaluate(test_data, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_data:
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def main():
    train_data_loader = get_data_loader(True)
    test_data_loader = get_data_loader(False)
    net = MLP()

    print("Initial accuracy:", evaluate(test_data_loader, net))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    for epoch in range(2):
        for data, targets in train_data_loader:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        print("Epoch", epoch, "Accuracy:", evaluate(test_data_loader, net))

    # Demo prediction
    for i, (data, _) in enumerate(test_data_loader):
        if i > 3:
            break
        plt.imshow(data[0][0], cmap='gray')
        plt.title(f"Prediction: {torch.argmax(net(data[0].unsqueeze(0))).item()}")
        plt.show()

if __name__ == "__main__":
    main()