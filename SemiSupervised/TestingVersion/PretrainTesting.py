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

train_dataset = datasets.ImageFolder("path", transform)
test_dataset = datasets.ImageFolder("path", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
                                
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
                                        
LR = 0.0003
entropy_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), LR)


def train():
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        labels = torch.Tensor([[0.0, 1.0] if l==1 else [1.0, 0.0] for l in labels])
        loss = entropy_loss(out, labels)
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


for epoch in range(0, 100):
    print("epoch", epoch)
    train()
    test()

torch.save(model.state_dict(), "./cat_dog100.pth")