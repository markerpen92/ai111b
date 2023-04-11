"""
其他監督實作 : [knearist] [decision tree] [naive base] [random forest]

1. knearist : 
    1. 指定k的數值以及data之間的權重計算方式(RBMfunction)
    2. 透過投票的方式來決定在範圍內屬於該class最多的,這筆unlabled data就會變成該class

2. decision tree : 

3. naive base : 

4. random forest :
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys

transform = transforms.Compose([
    # transforms.RandomResizedCrop(64),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("..\\image\\trainingData", transform)
test_dataset = datasets.ImageFolder("..\\image\\testingData", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# classes = train_dataset.classes
# classes_index = train_dataset.class_to_idx

model = models.resnet18(pretrained=True)
# # print(model)

# for param in model.parameters():
#     param.requires_grad = False

# model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 100),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Dropout(p=0.5),
#                                         torch.nn.Linear(100, 2))
                                    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

                                            
LR = 0.0003

entropy_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), LR)

def RBFunction(dataA , dataB , gamma) : 
    # print(dataA , dataB)
    # print(type(dataA), type(dataB))
    diff = torch.sub(dataA, dataB)
    #dist = np.dot(diff , diff)
    dist = (diff ** 2).sum().detach().numpy()
    # input(a)
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
    # if len(OutputDataset) < startbatch/8 : return 
    neighbors = []
    answers = []
    #input(OutputDataset)
    for out in outs : 
        # print("~~~~~~~")
        # print(out)
        # print("~~~~~~~")
        for data in OutputDataset : 
            # print(data)  
            # print(out)
            # print("==============")
            # input(data)
            # input(neighbors)
            if len(neighbors) < K : 
                neighbors.append(data)
                # input(neighbors)
                Sorting(neighbors , out , gamma) 
                # input(neighbors)
            else : 
                for i in neighbors : 
                    # input(data)
                    # input(out)
                    # print(out , data)
                    # print(type(out), type(data[0]))
                    if RBFunction(out , data[0] , gamma) > RBFunction(out , i[0] , gamma) : 
                        # input(neighbors)
                        # input(neighbors[0])
                        neighbors = neighbors[1:]
                        neighbors.append(data)
                        Sorting(neighbors , out , gamma) 
                        continue
        # CatNeighbors = 0
        # DogNeighbors = 0
        ans = torch.tensor([[0.0 , 0.0]] , requires_grad=True)
        for neighbor in neighbors :
            ans = torch.add(ans , RBFunction(out , neighbor[0] , gamma)*neighbor[0])
            # input(neighbor[0][0]) 
            # input(neighbor[0][1])
            # input(neighbor)
            # ans = ans + neighbor[0]
            # input(ans)
            # input(neighbors)
            #input(neighbor)
        #     if neighbor[1] == "dog" : DogNeighbors += 1
        #     else : CatNeighbors += 1
        # if DogNeighbors > CatNeighbors : answers.append("dog")
        # else : answers.append("cat")
        # input(ans)
        # input(out)
        # answers.append(ans/K)
        answers.append(ans)
    outs = answers
    #print(outs)                 

def train(OutputDataset):
    model.train()
    # time = 0
    for i, data in enumerate(train_loader):
        # print(i, end=" ")
        inputs, labels = data
        out = model(inputs)
        # print("train")
        # input(out)
        KNearistNeighbor(out , OutputDataset , K=15 , gamma=0.9 , startbatch=320)
        """如果將上面的KNN使用就會實現在半監督是上面"""
        # print("Knn time : " + str(time))
        # time += 1
        # input(out)
        for i in range(len(out)) : 
            OutputDataset.append([out[i] , labels[i]])
        # input(out)
        # print(OutputDataset)
        # for a in OutputDataset : 
        #     input(a[0])
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
    # input(OutputDataset)
    ans = test(OutputDataset)
    acc[0] += ans[0]
    acc[1] += ans[1]
    # if input("save? (y/n): ") == "y":
    #     torch.save(model.state_dict(), f"./cat_dog{epoch}.pth")
    if epoch%10 == 9 : 
        print("testing improve : " + str((acc[0] - lastacc[0])/10))
        print("training improve : " + str((acc[1] - lastacc[1])/10))
        print("testing : " + str(acc[0]/10) + "    training : " + str(acc[1]/10))
        lastacc = acc
        acc = [0 , 0]      
    epoch += 1
# for epoch in range(0, 10):
#     print("epoch", epoch)
#     train()
#     test()
    # if input("save? (y/n): ") == "y":
    #     torch.save(model.state_dict(), f"./cat_dog{epoch}.pth")

torch.save(model.state_dict(), "./cat_dog.pth")