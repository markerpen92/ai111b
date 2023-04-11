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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
      
LR = 0.0003
entropy_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LR)

def RBFunction(dataA , dataB , gamma) : 
    diff = torch.sub(dataA, dataB)
    dist = (diff ** 2).sum().detach().numpy()
    return np.exp(-gamma * dist)

def Sorting(neighbors , out , gamma) : 
    idx = len(neighbors)-1
    while RBFunction(neighbors[idx][0] , out , gamma) < RBFunction(neighbors[idx-1][0] , out , gamma) : 
        temp = neighbors[idx]
        neighbors[idx] = neighbors[idx-1]
        neighbors[idx-1] = temp
        idx -= 1
        if idx == 0 : return 

def KNearistNeighbor(outs , OutputDataset , K , gamma , startbatch) :
    neighbors = []
    answers = []
    for out in outs : 
        for data in OutputDataset : 
            if len(neighbors) < K : 
                neighbors.append(data)       
                Sorting(neighbors , out , gamma) 
            else : 
                for i in neighbors : 
                    if RBFunction(out , data[0] , gamma) > RBFunction(out , i[0] , gamma) : 
                        neighbors = neighbors[1:]
                        neighbors.append(data)
                        Sorting(neighbors , out , gamma) 
                        continue

        ans = torch.tensor([[0.0 , 0.0]] , requires_grad=True)
        for neighbor in neighbors :
            ans = torch.add(ans , RBFunction(out , neighbor[0] , gamma)*neighbor[0])
        answers.append(ans)
    outs = answers
               

def train(OutputDataset):
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        KNearistNeighbor(out , OutputDataset , K=15 , gamma=0.9 , startbatch=320)
        """如果將上面的KNN使用就會實現在半監督是上面"""

        for i in range(len(out)) : 
            OutputDataset.append([out[i] , labels[i]])
            
        loss = entropy_loss(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(OutputDataset):
    model.eval()
    correct = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        out = model(inputs)
        # KNearistNeighbor(out , OutputDataset , K=15 , gamma=0.9 , startbatch=320)
        """如果使勇在這裡的話就會是監督式的regression"""
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
    print(f"Test acc:{(correct.item()/len(test_dataset))*100}%", end="\t\t")
    t1 = (correct.item()/len(test_dataset))*100

    correct = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        _, predicted = torch.max(out, 1)
        correct += (predicted == labels).sum()
    print(f"Train acc:{(correct.item()/len(train_dataset))*100}%")
    t2 = (correct.item()/len(train_dataset))*100
    return [t1 , t2]

acc = [0 , 0]
lastacc = [0 , 0]
epoch = 0
while acc[0]/10.0 <= 92.0 : 
    OutputDataset = []
    print("epoch", epoch)
    train(OutputDataset)

    ans = test(OutputDataset)
    acc[0] += ans[0]
    acc[1] += ans[1]

    if epoch%10 == 9 : 
        print("testing improve : " + str((acc[0] - lastacc[0])/10))
        print("training improve : " + str((acc[1] - lastacc[1])/10))
        print("testing : " + str(acc[0]/10) + "    training : " + str(acc[1]/10))
        lastacc = acc
        acc = [0 , 0]      
    epoch += 1

torch.save(model.state_dict(), "./cat_dog.pth")