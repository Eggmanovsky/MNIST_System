import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# 检查是否有可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def get_data_loader(is_train):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = MNIST("", train=is_train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=128, shuffle=True)

def evaluate_metrics(test_data, net):
    net.eval()
    net.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return precision, recall, f1, cm

class InvertTransform:
    def __call__(self, tensor):
        return 1 - tensor

def load_and_preprocess_image(image_path):
    """加载并预处理单个图像"""
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        InvertTransform(),  # Adding the invert transform here
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0).to(device)  # 加载到设备

def predict_single_image(model, image_path):
    """加载图像并进行预测"""
    img_tensor = load_and_preprocess_image(image_path).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        predict = torch.argmax(output, dim=1)
    plt.imshow(img_tensor.cpu().squeeze(), cmap='gray')  # 从GPU转到CPU
    plt.title(f"Prediction: {predict.item()}")
    plt.show()

def main():
    model_path = r"D:\.VSCode\Python\AI\model\mlp_model_16Test.pth"  # 模型读取路径
    model_spath = r"D:\.VSCode\Python\AI\model\mlp_model_16Test.pth"  # 模型保存路径

    train_data_loader = get_data_loader(True)
    test_data_loader = get_data_loader(False)
    net = MLP().to(device)

    # 检查是否存在已保存的模型
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        net.to(device)
        print("Loaded saved model")
    else:
        print("No saved model found, starting training")

    if not os.path.exists(model_path):  # 仅在需要训练时进行训练
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.NLLLoss()

        for epoch in range(10):
            net.train()
            p0 = time.process_time()
            for data, targets in train_data_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
            p1 = time.process_time()
            precision, recall, f1, cm = evaluate_metrics(test_data_loader, net)
            print(f"Epoch {epoch}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Time: {p1-p0:.2f}s")

        torch.save(net.state_dict(), model_spath)
        print(f"Model saved to {model_spath}")

##################        
    
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # criterion = nn.NLLLoss()
    # for epoch in range(10):
    #     net.train()
    #     p0 = time.process_time()
    #     for data, targets in train_data_loader:
    #         data, targets = data.to(device), targets.to(device)
    #         optimizer.zero_grad()
    #         output = net(data)
    #         loss = criterion(output, targets)
    #         loss.backward()
    #         optimizer.step()
    #     p1 = time.process_time()
    #     precision, recall, f1, cm = evaluate_metrics(test_data_loader, net)
    #     print(f"Epoch {epoch}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, Time: {p1-p0:.2f}s")

    # torch.save(net.state_dict(), model_spath)
    # print(f"Model saved to {model_spath}")
    
    
################


    # 最终评估并显示混淆矩阵
    precision, recall, f1, cm = evaluate_metrics(test_data_loader, net)
    print(f"Final Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    # 演示预测使用测试数据集
    net.eval()
    for i, (data, _) in enumerate(test_data_loader):
        if i > 3:
            break
        data = data.to(device)
        plt.imshow(data[0].cpu().squeeze(), cmap='gray')
        prediction = torch.argmax(net(data[0].unsqueeze(0)), dim=1).item()
        plt.title(f"Prediction: {prediction}")
        plt.show()

    # 预测单个图像
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
        predict_single_image(net, image_path)

if __name__ == "__main__":
    main()