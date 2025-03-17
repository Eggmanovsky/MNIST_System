import torch
import os
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# 设定超参数
batch_size = 64
learning_rate = 0.01
epochs = 5

# 1. 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载训练数据和测试数据
trainset = datasets.MNIST('', download=True, train=True, transform=transform)
testset = datasets.MNIST('', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# 2. 定义卷积神经网络结构
class CNN(nn.Module):
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
    

# 初始化模型
model = CNN()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 4. 训练模型
for epoch in range(epochs):
    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 5. 模型评估
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')


#***************************************************************************************

class LocalMNISTDataset(Dataset):
    def __init__(self, root_path, transform=None):
        """
        初始化数据集
        Args:
            root_path (str): 包含图像的目录的路径
            transform (callable, optional): 应用于每个图像的变换
        """
        self.root_path = root_path
        self.transform = transform
        self.filenames = []  # 存储图像文件的列表
        self.labels = []  # 存储标签的列表

        # 读取图像文件及其标签
        for filename in os.listdir(root_path):
            if filename.endswith('.png'):
                # Assume labels are part of the filename format: "image_label.png"
                label = int(filename.split('_')[-1].split('.')[0])
                self.filenames.append(filename)
                self.labels.append(label)

    def __len__(self):
        """返回数据集中的图像数"""
        return len(self.filenames)

    def __getitem__(self, index):
        """按照给定索引获取图像及其标签"""
        # 图像路径
        img_path = os.path.join(self.root_path, self.filenames[index])
        # 读取图像
        image = Image.open(img_path).convert('L')  # 确保图像是灰度格式
        
        label = self.labels[index]

        # 如果提供了转换函数，则应用它
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

# 定义用于从图像到PyTorch张量的转换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 此处参数根据实际情况调整
])




model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 加载测试数据
test_dir = 'D:\\.VSCode\\Python\\MNIST\\test'
test_dataset = LocalMNISTDataset(root_path=test_dir, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=False)

# 测试函数，并保存图像及其识别结果
def test_and_save_images(model, test_loader, save_dir):
    with torch.no_grad():
        batch = next(iter(test_loader))  # 获取一个批次的数据
        images, labels = batch
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        for idx in range(images.shape[0]):
            # 将张量图像转换为PIL图像保存
            pil_image = TF.to_pil_image(images[idx])
            prediction = predicted[idx].item()
            img_label = labels[idx].item()
            filename = f"img_{idx}_label_{img_label}_pred_{prediction}.png"
            full_path = os.path.join(save_dir, filename)
            
            pil_image.save(full_path)
            
        print(f"Images and predictions saved to {save_dir}")

# 执行测试并保存图像和结果
save_dir = 'D:\\.VSCode\\Python\\MNIST\\test\\results'
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
test_and_save_images(model, test_loader, save_dir)