import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision.models.resnet
import torch.nn.parallel
import os
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

from model import resnet34

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 10

# 模型存储路径
model_save_path = './weights/3.pth'

transform_test = transforms.Compose([
    transforms.Resize([150,150]),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_test = datasets.ImageFolder('./dataset', transform_test)
print(dataset_test.class_to_idx)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = resnet34()
net = torch.load(model_save_path)
net.to(DEVICE)
net.eval()

def val(model, device, test_loader):
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predict_y = torch.max(output, dim=1)[1]
            acc += torch.eq(predict_y, target.to(device)).sum().item()

            # test_loss += loss_function(output, target).item()
            # pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).to(device)
            # correct += pred.eq(target.long()).sum().item()

        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
             acc, len(test_loader.dataset),
            100. * acc / len(test_loader.dataset)))


val(net, DEVICE, test_loader)