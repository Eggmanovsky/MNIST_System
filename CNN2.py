import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image  # 导入PIL库来处理图像

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
    cnn = CNN()
    image_path = r"D:\.VSCode\Python\AI\test\test1.jpg"  # 示例路径和文件名
    evaluate_single_image(cnn, image_path)

if __name__ == "__main__":
    main()