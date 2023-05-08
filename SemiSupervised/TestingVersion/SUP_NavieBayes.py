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
                        
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

                                            
LR = 0.0003

entropy_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), LR)

def Ranked(featureValue) : #+-5之間
    if featureValue>=5.0 : return 50000
    elif featureValue<5.0 and featureValue>-5.0 : 
        return int(featureValue*1000)
    else : return -50000

def ReadLabels(labels , batchSize) : 
    label = []
    for n in labels :
        if n == 1 : label.append("dog")
        else : label.append("cat")
    return label

def RankedData(out , labels , OutputDataset , batchsize) :
    label = ReadLabels(labels , batchsize) 
    outlabel = " "
    for i in range(len(out)) : 
        if out[i][0] < out[i][1] : outlabel = "dog"
        else : outlabel = "cat"
        catFeature = Ranked(out[i][0].item())
        dogFeature = Ranked(out[i][1].item())
        if outlabel != label[i] : OutputDataset.append([catFeature , dogFeature , "false"])
        else : OutputDataset.append([catFeature , dogFeature , "true"])

# def NaiveBayse(out , OutputDataset , labels , batchsize , eps , startbatch , buffersize) : 
#     if buffersize < startbatch : 
#         # RankedData(out , labels , OutputDataset , 8)
#         return 
#     DataFeature = []
#     DataAmount = len(OutputDataset) * batchsize
#     for outs in out :
#         catFeature = Ranked(outs[0].item())
#         dogFeature = Ranked(outs[1].item())
#         DataFeature.append([catFeature , dogFeature])
#     curraset = []
#     falset = []
#     for data in OutputDataset :
#         if data[2] == "true" : curraset.append([data[0] , data[1]])
#         else : falset.append([data[0], data[1]]) 
#     idx = 0   
#     \

def ConvertFeature(FeatureValue) : 
    if FeatureValue >= 5 : return 5000
    elif FeatureValue < 5 and FeatureValue > -5 : return FeatureValue * 1000 
    elif FeatureValue <= -5 : return -5000

def MemoryBufferCreate(outputs , labels , MemoryBuffer) : 
    for output , label in zip(outputs , labels) : 
        DogFeature = ConvertFeature(output[0])
        CatFeature = ConvertFeature(output[1])
        OutputIsAccurate= True
        if output[0] > output[1] and label.item() != 1 or output[1] > output[0] and label.item() != 0 : OutputIsAccurate = False
        MemoryBuffer.append([DogFeature , CatFeature , OutputIsAccurate])
    return MemoryBuffer

def NaiveBayes(outputs , MemoryBuffer) : 
    for output in outputs : 
        

def train(MemoryBuffer):
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        # NaiveBayse(out , OutputDataset , labels , 8 , 0.003 , 35 , idx)
        loss = entropy_loss(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for i , data in enumerate(train_loader) : 
        inputs , labels = data
        out = model(inputs)
        MemoryBuffer = MemoryBufferCreate(out , labels , MemoryBuffer)

def test(OutputDataset):
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        out = NaiveBayes(out , labels , OutputDataset , "test")
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
    MemotyBuffer = []
    print("epoch", epoch)
    train(MemotyBuffer)
    test(MemotyBuffer)
    if (epoch+1)%10 == 0:
        if aa := input("save? (y/n): ") == "y":
            torch.save(model.state_dict(), f"./cat_dog{epoch}.pth")
        if aa == "123": break
        

torch.save(model.state_dict(), "./cat_dog.pth")