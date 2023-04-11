import numpy as np
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

train_dataset = datasets.ImageFolder("..\\image\\trainingData", transform)
test_dataset = datasets.ImageFolder("..\\image\\testingData", transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = models.resnet18(pretrained=True)
                               
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
                                       
LR = 0.0003
entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LR)


def RBFunction(dataA , dataB , gamma) : 
    diff = dataA - dataB
    dist = (diff ** 2).sum().detach().numpy()
    return np.exp(-gamma * dist)

def Sorting(neighbors , out , gamma) : 
    idx = len(neighbors)-1
    while RBFunction(neighbors[idx] , out , gamma) < RBFunction(neighbors[idx-1] , out , gamma) : 
        temp = neighbors[idx]
        neighbors[idx] = neighbors[idx-1]
        neighbors[idx-1] = temp
        idx -= 1
        if idx == 0 : return 0

def KNearistNeighbor(out , OutputDataset , K , gamma , startbatch) :
    if len(OutputDataset) < startbatch : return out
    neighbors = []
    outs = out
    for data in OutputDataset : 
        if len(neighbors) < K : 
            neighbors.append(data)
            Sorting(neighbors , out , gamma) 
        else : 
            for i in neighbors : 
                if RBFunction(out , data , gamma) > RBFunction(out , i , gamma) : 
                    neighbors.remove(neighbors[0])
                    neighbors.append(data)
                    Sorting(neighbors , out , gamma) 
                    continue
    CatNeighbors = [] 
    DogNeighbors = []
    for neighbor in neighbors : 
        if neighbor[0][0]<neighbor[0][1] : DogNeighbors.append(neighbor)
        else : CatNeighbors.append(neighbor)
    if len(DogNeighbors) > len(CatNeighbors) : outs = torch.tensor([[0.0 , 3.0]] , requires_grad=True)
    else : outs = torch.tensor([[3.0 , 0.0]] , requires_grad=True)   
    return outs           

def train():
    model.train()
    OutputDataset = []
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        out = KNearistNeighbor(out , OutputDataset , K=10 , gamma=0.9 , startbatch=150)
        OutputDataset.append(out)
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


for epoch in range(0, 10):
    print("epoch", epoch)
    train()
    test()

torch.save(model.state_dict(), "./cat_dog.pth")