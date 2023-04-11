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

def Ranked(featureValue) : #+-5之間
    if featureValue>=5.0 : return 100
    elif featureValue<5.0 and featureValue>-5.0 : 
        return int(featureValue*100)
    else : return -100
    # if featureValue>=5.0 : return 1
    # elif featureValue<5.0 and featureValue>=4.0 : return 2
    # elif featureValue<3.9 and featureValue>=3.5 : return 3
    # elif featureValue<3.5 and featureValue>=3.0 : return 4
    # elif featureValue<3.0 and featureValue>=2.5 : return 5
    # elif featureValue<2.5 and featureValue>=2.0 : return 6
    # elif featureValue<2.0 and featureValue>=1.6 : return 7
    # elif featureValue<1.6 and featureValue>=1.2 : return 8
    # elif featureValue<1.2 and featureValue>=1.0 : return 9
    # elif featureValue<1.0 and featureValue>=0.8 : return 10
    # elif featureValue<0.8 and featureValue>=0.7 : return 11
    # elif featureValue<0.7 and featureValue>=0.6 : return 12
    # elif featureValue<0.6 and featureValue>=0.5 : return 13
    # elif featureValue<0.5 and featureValue>=0.4 : return 14
    # elif featureValue<0.4 and featureValue>=0.3 : return 15
    # elif featureValue<0.3 and featureValue>=0.2 : return 16
    # elif featureValue<0.2 and featureValue>=0.1 : return 17
    # elif featureValue<0.1 and featureValue>=0.0 : return 18
    # elif featureValue<0.0 and featureValue>=-0.1 : return 19
    # elif featureValue<-0.1 and featureValue>=-0.2 : return 20
    # elif featureValue<-0.2 and featureValue>=-0.3 : return 21
    # elif featureValue<-0.3 and featureValue>=-0.4 : return 22
    # elif featureValue<-0.4 and featureValue>=-0.5 : return 23
    # elif featureValue<-0.5 and featureValue>=-0.6 : return 24
    # elif featureValue<-0.6 and featureValue>=-0.7 : return 25
    # elif featureValue<-0.7 and featureValue>=-0.8 : return 26
    # elif featureValue<-0.8 and featureValue>=-0.9 : return 27
    # elif featureValue<-0.9 and featureValue>=-1.0 : return 28
    # elif featureValue<-1.0 and featureValue>=-1.2 : return 29
    # elif featureValue<-1.2 and featureValue>=-1.6 : return 30
    # elif featureValue<-1.6 and featureValue>=-2.0 : return 31
    # elif featureValue<-2.0 and featureValue>=-2.5 : return 32
    # elif featureValue<-2.5 and featureValue>=-3.0 : return 33
    # elif featureValue<-3.0 and featureValue>=-4.0 : return 34
    # elif featureValue<-4.0 and featureValue>=-5.0 : return 35
    # else : return 36

    # if featureValue>1.5 or featureValue==1.5 : return 1
    # elif featureValue<1.5 and featureValue>0.8 or featureValue==0.8 : return 2
    # elif featureValue<0.8 and featureValue>0 or featureValue==0 : return 3
    # elif featureValue<0 and featureValue>-0.8 and featureValue==-0.8 : return 4
    # elif featureValue<-0.8 and featureValue>-1.5 and featureValue==-1.5 : return 5
    # else : return 6

def ReadLabels(labels , batchSize) : 
    label = []
    for n in labels :
        if n == 1 : label.append("dog")
        else : label.append("cat")
    return label

def RankedData(out , labels , OutputDataset , batchsize) :
    #input(out)
    label = ReadLabels(labels , batchsize) 
    outlabel = " "
    for i in range(len(out)) : 
        # print(out)
        # print(out[i])
        # print("===========================")
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
    #input(OutputDataset)
    for data in OutputDataset :
        #input(data) 
        if data[2] == "true" : curraset.append([data[0] , data[1]])
        else : falset.append([data[0], data[1]]) 
    idx = 0   
    for features in DataFeature : 
        # input(DataFeature)
        currasetCatFeatures = falsetCatFeatures = currasetDogFeatures = falsetDogFeatures = 0
        # input(features)
        # input(falset)
        for f in falset : 
            if features[0] == f[0] : falsetCatFeatures += 1
            if features[1] == f[1] : falsetDogFeatures += 1
        for c in curraset : 
            if features[0] == c[0] : currasetCatFeatures += 1
            if features[1] == c[1] : currasetDogFeatures += 1
        # print("Falset : ")
        # print(falset)
        # print("Curraset : ")
        # print(curraset)
        # print("=======================")
        result = True
        label = "cat"
        if out[idx][1] > out[idx][0] : label = "dog"
        # input(curraset)
        # input(currasetDogFeatures)
        # input(features)
        # input(falsetCatFeatures)
        # input(falsetDogFeatures)
        a = (len(falset)/DataAmount)*(falsetCatFeatures/(len(falset)+eps))*(falsetDogFeatures/(len(falset)+eps))
        b = (len(curraset)/DataAmount)*(currasetCatFeatures/(len(curraset)+eps))*(currasetDogFeatures/(len(curraset)+eps))
        curracy = b/(a+b+eps)
        failure = a/(a+b+eps)
        # input((len(falset)/DataAmount))
        # input((falsetCatFeatures/(len(falset)+eps)))
        # input((falsetDogFeatures/(len(falset)+eps)))
        # print("a : " + str(a))
        # print("b : " + str(b))
        # print("curracy : " + str(curracy))
        # print("failure" + str(failure))
        if curracy < failure : 
            result = False
            counts += 1
        # print("accury : " + str(curracy) + " failure" + str(failure))
        if not result : 
            if label == "dog" : out[idx] = torch.tensor([[1.0 , 0.0]] , requires_grad=True)
            else : out[idx] = torch.tensor([[0.0 , 1.0]] , requires_grad=True)
            # t1 = out[idx][0]
            # t2 = out[idx][1]
            # average = (t1+t2)/2.0
            # out[idx] = torch.tensor([[0.0 , 0.0]] , requires_grad=True)
            # print("before : " + str(out[idx]))
            # out[idx] = torch.tensor([[t2 , t1]] , requires_grad=True)
            # print("after : " + str(out[idx]))
        idx += 1
    #print(counts)
   
def train(OutputDataset):
    model.train()
    
    idx = 0
    for i, data in enumerate(train_loader):

        # print(i, end=" ")
        inputs, labels = data
        out = model(inputs)
        RankedData(out , labels , OutputDataset , batchsize=8)
        # if idx <= 50 : RankedData(out , labels , OutputDataset , batchsize=8)
        # else : 
        #     NaiveBayse(out , OutputDataset , batchsize=8 , eps=0.0003)
        #     RankedData(out , labels , OutputDataset , batchsize=8)
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

torch.save(model.state_dict(), "./cat_dog.pth")