import torch
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
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # 输入通道=1, 输出通道=16, 卷积核大小=3x3, 步长=1
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # 输入通道=16, 输出通道=32, 卷积核大小=3x3, 步长=1
        # 计算卷积处理后的图像尺寸: W' = (W - F + 2P) / S + 1
        # 输出尺寸由卷积核大小、步长和填充决定，由于没有填充，我们假设此处结果很小
        # 第一层: (28 - 3) / 1 + 1 = 26 -> 池化后 26 / 2 = 13
        # 第二层: (13 - 3) / 1 + 1 = 11 -> 池化后 11 / 2 = 5.5 (不可能，所以取整为5)
        self.fc1 = nn.Linear(32 * 5 * 5, 128) # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 第一层卷积后接池化层
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 第二层卷积后接池化层
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=16, shuffle=True)

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net(x)  # 不需要 view
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def load_image(image_path):
    """加载并转换测试用例"""
    transform = transforms.Compose([
        transforms.Grayscale(),  # 确保图像是灰度的
        transforms.Resize((28, 28)),  # 调整大小以匹配MNIST数据的尺寸
        transforms.ToTensor()  # 转换为Tensor
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)  # 添加一个批次维度

def evaluate_single_image(model, image_path):
    """评估单个图像"""
    img_tensor = load_image(image_path)
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(img_tensor)
        predict = torch.argmax(output, dim=1)
    plt.imshow(img_tensor[0][0], cmap='gray')
    plt.title(f"Prediction: {predict.item()}")
    plt.show()

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    cnn = CNN()

    print("initial accuracy:", evaluate(test_data, cnn))
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    for epoch in range(10):
        for (x, y) in train_data:
            cnn.zero_grad()
            output = cnn(x)  # 不需要 view
            loss = F.nll_loss(F.log_softmax(output, dim=1), y)  # 增加 log_softmax
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, cnn))

    # 以下可保持原样：
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(cnn(x[0].unsqueeze(0)))  # 加上 unsqueeze 来增加批次维度
        plt.figure(n)
        plt.imshow(x[0][0], cmap='gray')  # 调整以显示第一个图像通道
        plt.title("Prediction: {}".format(predict.item()))
    plt.show()

    image_path = r"D:\.VSCode\Python\AI\test\test1.jpg"  # 示例路径和文件名
    evaluate_single_image(cnn, image_path)

    image_path = r"D:\.VSCode\Python\AI\test\test2.jpg"  # 示例路径和文件名
    evaluate_single_image(cnn, image_path)


if __name__ == "__main__":
    main()