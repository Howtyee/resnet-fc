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
from model import resnet34

BATCH_SIZE = 10
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Resize([150,150]),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.RandomResizedCrop(150),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 读取数据
whole_set = datasets.ImageFolder('./dataset', transform)
length = len(whole_set)
train_size, validate_size = int(0.9*length), int(0.1*length)
print(train_size, validate_size)
dataset_train,dataset_test=torch.utils.data.random_split(whole_set,[1630,182])

# dataset_train = datasets.ImageFolder('./dataset', transform)
# print(dataset_train.imgs)
# print(dataset_train.class_to_idx)
# dataset_test = datasets.ImageFolder('./dataset', transform)
# print(dataset_test.class_to_idx)

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)




modellr = 1e-4
net = resnet34()
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
model_weight_path = "./resnet34-b627a593.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
# for param in net.parameters():
#     param.requires_grad = False

# change fc layer structure
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 2)
net.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=modellr)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 5))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.binary_cross_entropy(output, target)
        # target = target.squeeze()
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
# 定义测试过程
def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
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


# 训练
for epoch in range(1, EPOCHS + 1):
    adjust_learning_rate(optimizer, epoch)
    train(net, DEVICE, train_loader, optimizer, epoch)
    val(net, DEVICE, test_loader)

    torch.save(net, './weights/' + str(epoch) + '.pth')