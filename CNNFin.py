import os
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", train=is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=16, shuffle=True)

def evaluate(test_data, net):
    net.eval()
    net.to(device)
    n_correct = 0
    n_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
    
    # 计算评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
    # 返回准确率
    return n_correct / n_total

class InvertTransform:
    def __call__(self, tensor):
        return 1 - tensor

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        InvertTransform()  # Adding the invert transform here
    ])
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0).to(device)

def evaluate_single_image(model, image_path):
    img_tensor = load_image(image_path)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predict = torch.argmax(output, dim=1)
    plt.imshow(img_tensor.cpu()[0][0], cmap='gray')
    plt.title(f"Prediction: {predict.item()}")
    plt.show()

def main():
    model_path = r"D:\.VSCode\Python\AI\model\cnn_model_16Test.pth"  # 模型读取路径
    model_spath =r"D:\.VSCode\Python\AI\model\cnn_model_16Test.pth" # 模型保存路径

    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    
    cnn = CNN().to(device)

    # 检查是否存在已保存的模型
    if os.path.exists(model_path):
        cnn.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded saved model.")
    else:
        print("No saved model found, starting training.")
    
    if not os.path.exists(model_path):  # 没有模型文件时进行训练
        print("Initial accuracy:", evaluate(test_data, cnn))
        optimizer = optim.Adam(cnn.parameters(), lr=0.001)
        
        for epoch in range(10):
            cnn.train()
            p0 = time.process_time()
            for (x, y) in train_data:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = cnn(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
            p1 = time.process_time()
            print(f"Epoch {epoch}, Accuracy: {evaluate(test_data, cnn)}, Time: {p1-p0:.2f}s")
        
        # 保存模型
        torch.save(cnn.state_dict(), model_spath)
        print("Model saved to", model_spath)

    # 最终评估
    print("Final accuracy and evaluation metrics on test set:")
    evaluate(test_data, cnn)

    # 测试现有模型
    image_paths = [
        r"D:\.VSCode\Python\AI\test\test0.jpg",
        r"D:\.VSCode\Python\AI\test\test1.jpg",
        r"D:\.VSCode\Python\AI\test\test2.jpg",
        r"D:\.VSCode\Python\AI\test\test3.jpg",
        r"D:\.VSCode\Python\AI\test\test4.jpg",
        r"D:\.VSCode\Python\AI\test\test5.jpg",
        r"D:\.VSCode\Python\AI\test\test6.jpg",
        r"D:\.VSCode\Python\AI\test\test7.jpg",
        r"D:\.VSCode\Python\AI\test\test8.jpg",
        r"D:\.VSCode\Python\AI\test\test9.jpg",
        r"D:\.VSCode\Python\AI\test\test10.jpg",
        r"D:\.VSCode\Python\AI\test\test11.jpg",
        r"D:\.VSCode\Python\AI\test\test12.jpg",
        r"D:\.VSCode\Python\AI\test\test13.jpg",
        r"D:\.VSCode\Python\AI\test\test14.jpg",
        r"D:\.VSCode\Python\AI\test\test15.jpg",
        r"D:\.VSCode\Python\AI\test\test16.jpg",
        r"D:\.VSCode\Python\AI\test\test17.jpg",
        r"D:\.VSCode\Python\AI\test\test18.jpg",
    ]
    
    for image_path in image_paths:
        evaluate_single_image(cnn, image_path)

if __name__ == "__main__":
    main()