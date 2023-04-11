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
                        
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

                                            
LR = 0.0003

entropy_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), LR)

def Ranked(featureValue) : #+-5之間
    if featureValue>=5.0 : return 1
    elif featureValue<5.0 and featureValue>=4.0 : return 2
    elif featureValue<3.9 and featureValue>=3.5 : return 3
    elif featureValue<3.5 and featureValue>=3.0 : return 4
    elif featureValue<3.0 and featureValue>=2.5 : return 5
    elif featureValue<2.5 and featureValue>=2.0 : return 6
    elif featureValue<2.0 and featureValue>=1.6 : return 7
    elif featureValue<1.6 and featureValue>=1.2 : return 8
    elif featureValue<1.2 and featureValue>=1.0 : return 9
    elif featureValue<1.0 and featureValue>=0.8 : return 10
    elif featureValue<0.8 and featureValue>=0.7 : return 11
    elif featureValue<0.7 and featureValue>=0.6 : return 12
    elif featureValue<0.6 and featureValue>=0.5 : return 13
    elif featureValue<0.5 and featureValue>=0.4 : return 14
    elif featureValue<0.4 and featureValue>=0.3 : return 15
    elif featureValue<0.3 and featureValue>=0.2 : return 16
    elif featureValue<0.2 and featureValue>=0.1 : return 17
    elif featureValue<0.1 and featureValue>=0.0 : return 18
    elif featureValue<0.0 and featureValue>=-0.1 : return 19
    elif featureValue<-0.1 and featureValue>=-0.2 : return 20
    elif featureValue<-0.2 and featureValue>=-0.3 : return 21
    elif featureValue<-0.3 and featureValue>=-0.4 : return 22
    elif featureValue<-0.4 and featureValue>=-0.5 : return 23
    elif featureValue<-0.5 and featureValue>=-0.6 : return 24
    elif featureValue<-0.6 and featureValue>=-0.7 : return 25
    elif featureValue<-0.7 and featureValue>=-0.8 : return 26
    elif featureValue<-0.8 and featureValue>=-0.9 : return 27
    elif featureValue<-0.9 and featureValue>=-1.0 : return 28
    elif featureValue<-1.0 and featureValue>=-1.2 : return 29
    elif featureValue<-1.2 and featureValue>=-1.6 : return 30
    elif featureValue<-1.6 and featureValue>=-2.0 : return 31
    elif featureValue<-2.0 and featureValue>=-2.5 : return 32
    elif featureValue<-2.5 and featureValue>=-3.0 : return 33
    elif featureValue<-3.0 and featureValue>=-4.0 : return 34
    elif featureValue<-4.0 and featureValue>=-5.0 : return 35
    else : return 36

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

def NaiveBayse(out , OutputDataset , batchsize , eps) : 
    DataFeature = []
    DataAmount = len(OutputDataset) * batchsize
    for outs in out :
        catFeature = Ranked(outs[0].item())
        dogFeature = Ranked(outs[1].item())
        DataFeature.append([catFeature , dogFeature])
    curraset = []
    falset = []
    for data in OutputDataset :
        if data[2] == "true" : curraset.append([data[0] , data[1]])
        else : falset.append([data[0], data[1]]) 
    idx = 0   
    for features in DataFeature : 
        currasetCatFeatures = falsetCatFeatures = currasetDogFeatures = falsetDogFeatures = 0
        for f in falset : 
            if features[0] == f[0] : falsetCatFeatures += 1
            if features[1] == f[1] : falsetDogFeatures += 1
        for c in curraset : 
            if features[0] == c[0] : currasetCatFeatures += 1
            if features[1] == c[1] : currasetDogFeatures += 1
        result = True
        label = "cat"
        if out[idx][1] > out[idx][0] : label = "dog"
        a = (len(falset)/DataAmount)*(falsetCatFeatures/(len(falset)+eps))*(falsetDogFeatures/(len(falset)+eps))
        b = (len(curraset)/DataAmount)*(currasetCatFeatures/(len(curraset)+eps))*(currasetDogFeatures/(len(curraset)+eps))
        curracy = b/(a+b+eps)
        failure = a/(a+b+eps)
        if curracy < failure : result = False
        if not result : 
            if label == "dog" : out[idx] = torch.tensor([[3.0 , 0.0]] , requires_grad=True)
            else : out[idx] = torch.tensor([[0.0 , 3.0]] , requires_grad=True)
        idx += 1
   
def train():
    model.train()
    OutputDataset = []
    idx = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        if idx <= 35 : RankedData(out , labels , OutputDataset , batchsize=8)
        else : 
            NaiveBayse(out , OutputDataset , batchsize=8 , eps=0.0003)
            RankedData(out , labels , OutputDataset , batchsize=8)
        idx += 1
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
    if (epoch+1)%10 == 0:
        if aa := input("save? (y/n): ") == "y":
            torch.save(model.state_dict(), f"./cat_dog{epoch}.pth")
        if aa == "123": break
        

torch.save(model.state_dict(), "./cat_dog.pth")