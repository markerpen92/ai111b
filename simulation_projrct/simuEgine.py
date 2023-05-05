class Value : 
    def __init__(self , data , children=()) : 
        self.data = data
        self.grade = 0
        self.backward = lambda : None
        self.children = set(children)
    
    def __add__(self , other) : 
        if not isinstance(other , Value) : other = Value(other)
        out = Value(self.data+other.data , (self , other))
        def backward() : 
            # print("====+====")
            # print(out)
            # print(self)
            # print(other)
            # input()
            self.grade += out.grade
            other.grade += out.grade
            # print("after----------------")
            # print(out)
            # print(self)
            # print(other)
            # input()
        out.backward = backward
        return out
    
    def __mul__(self , other) : 
        if not isinstance(other , Value) : other = Value(other)
        out = Value(self.data*other.data , (self , other))
        # input(out)
        def backward() : 
            # print("====*====")
            # print(out)
            # print(self)
            # print(other)
            # input()
            self.grade += out.grade*other.data
            other.grade += out.grade*self.data
            # print("after----------------")
            # print(out)
            # print(self)
            # print(other)
            # input()
        out.backward = backward
        return out

    def relu(self) : 
        out = Value(self.data , (self,))
        if self.data < 0 : out.data = 0
        def backward() : 
            # print("====relu====")
            # print(out)
            # print(self)
            # input()
            if self.data > 0 : self.grade = self.grade + 1
            else : self.grade = self.grade + 0 
            # print("after----------------")
            # print(out)
            # print(self)
            # input()
        out.backward = backward
        return out
    
    def backwardpropagation(self) : 
        visited = set()
        map = []

        def order(node) : 
            if node in visited : return
            else : 
                visited.add(node)
                for child in node.children : 
                    order(child)
                map.append(node)
        order(self)
        self.grade = 1
        for node in reversed(map) : 
            # print(node.backward)
            # input(type(node))
            node.backward()
    
    def __repr__(self) : 
        return f"Value => data({self.data}) | grade({self.grade})\n"