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
hinge = nn.MarginRankingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


def train():
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        outs = model(inputs)
        idx = 0
        Labels = []
        for l in labels : 
            if l == 0 : Labels.append(torch.tensor(-3.0 , requires_grad=True))
            else : Labels.append(torch.tensor(3.0 , requires_grad=True))
        for out in outs : 
            x1 = torch.tensor(out[0].item() , requires_grad=True)
            x2 = torch.tensor(out[1].item() , requires_grad=True)
            y = Labels[idx]
            idx += 1
            # input(x1)
            # input(x2)
            # input(y)
            loss = hinge(x1 , x2 , y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # for l in range(len(labels)) :
        #     if labels[l]==0 : labels[l] = -1.0
        #     else : labels[l] = 1.0
        # x1 = []
        # x2 = []
        # y = []
        # for out in outs : 
        #     x1.append(out[0])
        #     x2.append(out[1])
        # for l in range(len(labels)) : 
        #     y.append(labels[l])
        # input(torch.tensor(x1))
        # input(torch.tensor(x2))
        # input(y)
        # input(torch.tensor(y))
        

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