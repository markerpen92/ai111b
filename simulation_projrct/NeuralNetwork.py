from sklearn.datasets import make_moons, make_blobs
from simuNN import MutiLayersPerceptron
from LossFunction import HingueLoss
from LearningRate import learning_rate

model = MutiLayersPerceptron(2 , [5,5,1])

inputs, y = make_moons(n_samples=100, noise=0.1)
labels = y*2 - 1 # make y be -1 or 1

print("first model")
input(model)
print("=============================================")

lr = learning_rate()
for i in range(100) : 
    model.InitGrade()
    losses = HingueLoss(model , inputs , labels , 1e-4)
    totaloss = losses.loss()
    print("the losses : ")
    input(totaloss)
    print("~~~~~~~~~~~~~~~~~~~")
    totaloss.backwardpropagation()
    print(len(model.parameters()))
    for param in model.parameters() : 
        # print(param.grade)
        param.data -= lr.SGD(i , 100 , 0.9) * param.grade
    input(model)
    if i!=0 : print(totaloss.data)
print(model.parameters())