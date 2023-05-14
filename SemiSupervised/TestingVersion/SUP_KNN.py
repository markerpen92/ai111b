import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import sys

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("D:\\NQU\\Algrithm\\image\\trainingData", transform)
test_dataset = datasets.ImageFolder("D:\\NQU\\Algrithm\\image\\testingData", transform)

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

def ConvertFeature(feature) : 
    if feature >= 7 : return 7000
    elif feature < 7 and feature > -7 : return int(feature*1000)
    elif feature <= -7 : return -7000

def CreateMemoryBuffer(MemoryBuffer , outputs , labels) : 
    for output , label in zip(outputs , labels) : 
        DogFeature = ConvertFeature(output[0].item())
        CatFeature = ConvertFeature(output[1].item())
        OutputIsAccurate = True
        if DogFeature > CatFeature and label.item() != 0 or CatFeature > DogFeature and label.item() != 1 : OutputIsAccurate=False
        MemoryBuffer.append([DogFeature , CatFeature , OutputIsAccurate])
    return MemoryBuffer

def RBFunction(output , t) : 
    x1 = output[0].item()
    x2 = output[1].item()
    TwoNorm = (x1-t[0])**2 + (x2-t[1])**2
    return np.exp(-0.9*TwoNorm)

def KNearistNeighbor(K , MemoryBuffer , outputs) : 
    answers = []
    for output in outputs : 
        neighbor = []
        OutputFeature = torch.tensor([ConvertFeature(output[0].item()) , ConvertFeature(output[1].item())])
        for i in range(K) : neighbor.append(MemoryBuffer[i])
        for i in range(len(MemoryBuffer)-K) : 
            for j in range(K) : 
                if RBFunction(OutputFeature , neighbor[j]) < RBFunction(OutputFeature , MemoryBuffer[i]) : 
                    MemoryBuffer[i] = neighbor[j]
                    break
        FalseCount = TrueCount = 0
        for i in MemoryBuffer : 
            if i[2] == False : FalseCount += 1
            else : TrueCount += 1
        if FalseCount > TrueCount : 
            answers.append([output[1].item(), output[0].item()])
        else : answers.append([output[0].item(), output[1].item()])
    return torch.tensor(answers , requires_grad=True)

def train(MemoryBuffer , LossBuffer):
    model.train()
    epoch = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        loss = entropy_loss(out, labels)
        LossBuffer.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i , data in enumerate(train_loader) : 
        inputs , labels = data
        out = model(inputs)
        MemoryBuffer = CreateMemoryBuffer(MemoryBuffer , out , labels)

def test(MemoryBuffer):
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        out = KNearistNeighbor(10 , MemoryBuffer , out)
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
    MemoryBuffer = []
    LossBuffer = []
    print("epoch", epoch)
    start=datetime.datetime.now()
    train(MemoryBuffer , LossBuffer)
    test(MemoryBuffer)
    end  =datetime.datetime.now()
    diff = end - start
    print(f"======epoch{epoch}======\ntime cost : {diff.microseconds}\n") # 單位微秒
    print("loss value : ")
    for lossvalue in LossBuffer : 
        print(lossvalue)
    print("==================\n")

torch.save(model.state_dict(), "./cat_dog100.pth")