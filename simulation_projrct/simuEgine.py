class Value : 
    def __init__(self , data , children=()) : 
        self.data = data
        self.grade = 0
        self.backward = lambda : None
        self.children = set()
    
    def __add__(self , other) : 
        if not isinstance(other , Value) : other = Value(other)
        out = Value(self.data+other.data , (self , other))
        def backward() : 
            self.grade += out.data
            other.grade += out.data
        out.backward = backward()
    
    def __mul__(self , other) : 
        if not isinstance(other , Value) : other = Value(other)
        out = Value(self.data*other.data , (self , other))
        def backward() : 
            self.grade += out.grade*other.data
            other.grade += out.grade*self.data
        out.backward = backward()

    def relu(self) : 
        out = Value(self.data , (self))
        if self.data < 0 : out = Value(0 , (self))
        def backward() : 
            self.grade += int(self.data>0)*out.data
        out.backward = backward()
    
    def backwardpropagation(self) : 
        visited = set()
        map = []

        def order(node) : 
            if node in visited : return
            else : 
                visited.add(node)
                for child in node.children : 
                    order(child)
                    map.append(child)
        
        for node in reversed(map) : 
            node.backward()
    
    def __repr__(self) :
        return f"Value => data({self.data}) | grade({self.grade})\n"