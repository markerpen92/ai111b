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
    if feature >= 7 : return 700
    elif feature < 7 and feature > -7 : return int(feature*100)
    elif feature <= -7 : return -700

def CreateMemoryBuffer(MemoryBuffer , outputs , labels) : 
    for output , label in zip(outputs , labels) : 
        DogFeature = ConvertFeature(output[0].item())
        CatFeature = ConvertFeature(output[1].item())
        OutputIsAccurate = True
        if DogFeature > CatFeature and label.item() != 0 or CatFeature > DogFeature and label.item() != 1 : OutputIsAccurate=False
        MemoryBuffer.append([DogFeature , CatFeature , OutputIsAccurate])
    return MemoryBuffer

def EuclideanDistance(output , t) : 
    # if state == "n" : print("n     " , end='\n')
    # else : print("m     " , end='\n')
    x1 = output[0].item()
    x2 = output[1].item()
    # print(f"x1 : {x1} , x2 : {x2}")
    # print(f"t1 : {t[0]} , t2 : {t[1]}")
    TwoNorm = (x1-t[0])**2 + (x2-t[1])**2
    # print(TwoNorm)
    return TwoNorm

def KNearistNeighbor(K , MemoryBuffer , outputs) : 
    answers = []
    for output in outputs : 
        neighbor = []
        OutputFeature = torch.tensor([ConvertFeature(output[0].item()) , ConvertFeature(output[1].item())])
        # print(OutputFeature)
        for i in range(K) :  
            neighbor.append(MemoryBuffer[i])
        for i in range(K) : 
            for j in range(K) : 
                if EuclideanDistance(OutputFeature , neighbor[i]) < EuclideanDistance(OutputFeature , neighbor[j]) : 
                    temp = neighbor[i]
                    neighbor[i] = neighbor[j]
                    neighbor[j] = temp
        # input(neighbor)
        for i in range(K , len(MemoryBuffer)) : 
            # for j in range(len(neighbor)) : 
            #     print(MemoryBuffer[i])
            #     if EuclideanDistance(OutputFeature , neighbor[j] , "n") > EuclideanDistance(OutputFeature , MemoryBuffer[i] , "m") : 
            #         neighbor[j] = MemoryBuffer[i] 
            #         break
            #     input(neighbor)
            idx = 0
            IsIn = False
            # print(MemoryBuffer[i])
            while EuclideanDistance(OutputFeature , neighbor[idx]) > EuclideanDistance(OutputFeature , MemoryBuffer[i]) : 
                idx += 1
                IsIn = True
                if idx >= len(neighbor) :
                    idx -= 1
                    break
            if IsIn : 
                neighbor[idx] = MemoryBuffer[i]

        FalseCount = TrueCount = 0
        # print("=============================")
        # for n in neighbor : 
        #     print(n)
        for i in neighbor : 
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
    K = 9
    model.eval()
    correct = 0
    knn_correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        knn_out = KNearistNeighbor(K , MemoryBuffer , out)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
        _, predicted = torch.max(knn_out, 1)
        knn_correct += (predicted == labels).sum()
    print(f"Test acc:{(correct.item()/len(test_dataset))*100}%", end="\t\t")
    print(f"KNN Test acc:{(knn_correct.item()/len(test_dataset))*100}%", end="\t\t")

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
    # print("loss value : ")
    # for lossvalue in LossBuffer : 
    #     print(lossvalue)
    # print("==================\n")

torch.save(model.state_dict(), "./cat_dog100.pth")