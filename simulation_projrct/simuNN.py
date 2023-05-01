from simuEgine import Value
import random as rd

class Neuron() : 
    def __init__(self , input) : 
        self.weights = []
        self.bias = Value(0)
        for i in range(input) : 
            self.weights.append(Value(rd.uniform(-1 , 1)))

    def __call__(self , input , nonLinear) : 
        out = Value(0)
        for i in range(self.weights) : 
            out += self.weights[i] * input[i]
        out += self.bias
        if nonLinear : out.relu()
        return out
    
    def parameters(self) : 
        parameters = []
        for w in self.weights : 
            parameters.append(w)
        parameters.append(self.bias)
        return parameters
    
    def __repr__(self) : 
        return f"Neuron => weights({self.weights}) | bias({self.bias})"

class Layer : 
    def __init__(self , input , output ) : 
        self.neuronMap = []
        for n in range(output) : 
            self.neuronMap.append(Neuron(input))

    def __call__(self , input , nonLinear) : 
        out = []
        for n in self.neuronMap : 
            out.append(n(input , nonLinear))
        return out
    
    def parameters(self) : 
        _parameters = []
        for n in self.neuronMap : 
            for param in n.parameters() : 
                _parameters.append(param)
        return _parameters
    
    def __repr__(self) : 
        return f"Layer => layer({self.neuronMap})"

class MutiLayersPerceptron : 
    def __init__(self , input , output) : 
        structure = [input] + output
        self.layerMap = []
        for l in range(len(structure)-1) : 
            self.layerMap.append(Layer(structure[l] , structure[l+1]))

    def __call__(self , input) :
        x = input
        for L in self.layerMap : 
            x = L(x)
        return x
    
    def parameters(self) : 
        _parameters = []
        for layer in self.layerMap : 
            for param in layer.parameters() : 
                _parameters.append(param)
        return _parameters
    
    def __repr__(self) : 
        return f"MLP : Layers({self.layerMap})\n"