from simuNN import MutiLayersPerceptron
from simuEgine import Value

class HingueLoss : 
    def __init__(self , model , inputs , labels , epsilon) : 
        output = []
        for i in inputs : 
            output.append(model(i))
        losses = Value(0)
        for i in range(len(labels)) : 
            yhat = labels[i]
            y = output[i][0]
            t = y*yhat
            t = t*-1
            t = t+1
            losses = losses+t
        self.average_loss = losses.relu()*(1/len(inputs))
        self.Regu2Loss = Value(0)
        for param in model.parameters() : 
            t = param*param
            self.Regu2Loss = self.Regu2Loss + t
        self.Regu2Loss *= (1/2*epsilon)

        # totaloss = average_loss + Regu2Loss
        # return totaloss

    def loss(self) :
        totaloss = self.average_loss + self.Regu2Loss
        return totaloss