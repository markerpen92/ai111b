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

train_dataset = datasets.ImageFolder("D:\\NQU\\Algrithm\\image\\trainingData", transform)
test_dataset = datasets.ImageFolder("D:\\NQU\\Algrithm\\image\\testingData", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = models.resnet18(pretrained=True)
                        
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

                                            
LR = 0.0003

entropy_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), LR)

def ConvertFeature(FeatureValue) : 
    if FeatureValue >= 7 : return 7000
    elif FeatureValue < 7 and FeatureValue > -7 : return int(FeatureValue * 1000) 
    elif FeatureValue <= -7 : return -7000

def MemoryBufferCreate(outputs , labels , MemoryBuffer) : 
    for output , label in zip(outputs , labels) : 
        DogFeature = ConvertFeature(output[0])
        CatFeature = ConvertFeature(output[1])
        OutputIsAccurate= True
        if output[0] > output[1] and label.item() != 0 or output[1] > output[0] and label.item() != 1 : OutputIsAccurate = False
        MemoryBuffer.append([DogFeature , CatFeature , OutputIsAccurate])
    return MemoryBuffer

def NaiveBayes(outputs , MemoryBuffer) : 
    answers = []
    for output in outputs : 
        outputFeature = torch.tensor([ConvertFeature(output[0].item()) , ConvertFeature(output[1].item())])
        TrueAmount = FalseAmount = XBelongstoTrue = XBelongstoFalse = 0
        for memory in MemoryBuffer : 
            if memory[2] == True : 
                TrueAmount += 1
                if memory[0] == outputFeature[0].item() and memory[1] == outputFeature[1].item() : XBelongstoTrue += 1
            else :
                FalseAmount += 1
                if memory[0] == outputFeature[0].item() and memory[1] == outputFeature[1].item() : XBelongstoFalse += 1
        Pa = (XBelongstoTrue/TrueAmount+0.0003)*(TrueAmount/len(MemoryBuffer))
        Pb = (XBelongstoFalse/FalseAmount+0.0003)*(FalseAmount/len(MemoryBuffer))
        PofTrue = Pa/(Pa+Pb+0.0003)
        PofFalse = Pb/(Pa+Pb+0.0003)
        if PofTrue < PofFalse : answers.append([output[1].item() , output[0].item()])
        else : answers.append([output[0].item() , output[1].item()])
    return torch.tensor(answers)


def train(MemoryBuffer , LossBuffer):
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        # NaiveBayse(out , OutputDataset , labels , 8 , 0.003 , 35 , idx)
        loss = entropy_loss(out, labels)
        LossBuffer.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for i , data in enumerate(train_loader) : 
        inputs , labels = data
        out = model(inputs)
        MemoryBuffer = MemoryBufferCreate(out , labels , MemoryBuffer)

def test(MemoryBuffer):
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        out = NaiveBayes(out , MemoryBuffer)
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
        

torch.save(model.state_dict(), "./cat_dog.pth")