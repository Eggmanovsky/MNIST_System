import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1).to(device)
        self.conv2 = nn.Conv2d(16, 32, 3, 1).to(device)
        self.fc1 = nn.Linear(32 * 5 * 5, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)  

    def forward(self, x):
        x = x.to(device)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=16, shuffle=True)

def evaluate(test_data, net):
    net = net.to(device)
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)  # 加载到设备

def evaluate_single_image(model, image_path):
    model = model.to(device)
    img_tensor = load_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predict = torch.argmax(output, dim=1)
    plt.imshow(img_tensor.cpu()[0][0], cmap='gray')  # 确保数据回到cpu
    plt.title(f"Prediction: {predict.item()}")
    plt.show()

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    cnn = CNN().to(device)  # 模型初始化到设备

    print("initial accuracy:", evaluate(test_data, cnn))
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    for epoch in range(10):
        p0=time.process_time()
        for (x, y) in train_data:
            x, y = x.to(device), y.to(device)
            cnn.zero_grad()
            output = cnn(x)
            loss = F.nll_loss(F.log_softmax(output, dim=1), y)
            loss.backward()
            optimizer.step()
        p1=time.process_time()
        print("epoch", epoch, "accuracy:", evaluate(test_data, cnn), "coast:", p1-p0)

    # for (n, (x, _)) in enumerate(test_data):
    #     if n > 3:
    #         break
    #     x = x.to(device)
    #     predict = torch.argmax(cnn(x[0].unsqueeze(0)))
    #     plt.figure(n)
    #     plt.imshow(x.cpu()[0][0], cmap='gray')
    #     plt.title("Prediction: {}".format(predict.item()))
    # plt.show()

    image_path = r"D:\.VSCode\Python\AI\test\test1.jpg"
    evaluate_single_image(cnn, image_path)

    image_path = r"D:\.VSCode\Python\AI\test\test2.jpg"
    evaluate_single_image(cnn, image_path)


if __name__ == "__main__":
    main()