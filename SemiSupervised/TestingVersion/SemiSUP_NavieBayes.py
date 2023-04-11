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

def Ranked(featureValue) : 
    if featureValue>=5.0 : return 100
    elif featureValue<5.0 and featureValue>-5.0 : 
        return int(featureValue*100)
    else : return -100

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
    counts = 0
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
        if curracy < failure : 
            result = False
            counts += 1
        if not result : 
            if label == "dog" : out[idx] = torch.tensor([[1.0 , 0.0]] , requires_grad=True)
            else : out[idx] = torch.tensor([[0.0 , 1.0]] , requires_grad=True)
        idx += 1

   
def train(OutputDataset):
    model.train()
    
    idx = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        out = model(inputs)
        RankedData(out , labels , OutputDataset , batchsize=8)
        idx += 1
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
        NaiveBayse(out , OutputDataset , batchsize=8 , eps=0.0003)
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
    t2 = (correct.item()/len(train_dataset))*100
    print(f"Train acc:{(correct.item()/len(train_dataset))*100}%")
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