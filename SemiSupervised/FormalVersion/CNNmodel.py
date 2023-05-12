import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("C:\\Users\\user\\Desktop\\alg\\mid\\SVM\\train\\train_200\\", transform)
test_dataset = datasets.ImageFolder("C:\\Users\\user\\Desktop\\alg\\mid\\SVM\\train\\test_100\\", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
                                
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
                                        
LR = 0.0003
entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LR)


def train(LossBuffer):
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        loss = entropy_loss(out, labels)
        LossBuffer.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
    print(f"Test acc:{(correct.item()/len(test_dataset))*100}%", end="\t\t")

    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
    print(f"Train acc:{(correct.item()/len(train_dataset))*100}%")


import datetime
for epoch in range(0, 100):
    start=datetime.datetime.now()
    LossBuffer = []
    print("epoch", epoch)
    train(LossBuffer)
    test()
    end  =datetime.datetime.now()
    diff = end - start
    print(f"time cost : {diff.microseconds}") # 單位微秒
    print("loss value : ")
    for lossvalue in LossBuffer : 
        print(lossvalue)

torch.save(model.state_dict(), "./cat_dog100.pth")