import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys
#import timm

transform = transforms.Compose([
    # transforms.RandomResizedCrop(64),
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("C:\\Users\\user\\Desktop\\alg\\mid\\SVM\\train\\train_200\\", transform)
test_dataset = datasets.ImageFolder("C:\\Users\\user\\Desktop\\alg\\mid\\SVM\\train\\test_100\\", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# classes = train_dataset.classes
# classes_index = train_dataset.class_to_idx

# model = models.vgg16(pretrained = True)
model = models.resnet18(pretrained=True)
# # print(model)

for param in model.parameters():
    param.requires_grad = False

# model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 100),
#                                         torch.nn.ReLU(),
#                                         torch.nn.Dropout(p=0.5),
#                                         torch.nn.Linear(100, 2))
                                    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)


                                            
LR = 0.0003

entropy_loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), LR)


def train():
    model.train()
    for i, data in enumerate(train_loader):
        # print(i, end=" ")
        inputs, labels = data
        out = model(inputs)
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
    # if input("save? (y/n): ") == "y":
    #     torch.save(model.state_dict(), f"./cat_dog{epoch}.pth")

torch.save(model.state_dict(), "./cat_dog100.pth")