import random as rd
import math

Solution = [0,1,2,3,4,5,6,7,8,9,10,11]
rd.shuffle(Solution)
print(Solution)

def Swap(S , pointA , pointB) :
    temp = S[pointA]
    S[pointA] = S[pointB]
    S[pointB] = temp
    return S 

def Distance(pointA , pointB) :
    columnA = int(pointA/4) 
    rowA = int(pointA-columnA*4)
    columnB = int(pointB/4)
    rowB = int(pointB-columnB*4)
    distance = ((columnA-columnB)**2 + (rowA-rowB)**2)**(1/2)
    return distance

def GraphDistance(S) :
    sum = 0 
    for i in range(0, len(S)-1) :
        sum+=Distance(S[i] , S[i+1])
    sum += Distance(S[len(S)-1] , S[0])
    return sum

def optimize(S) : 
    MaxEpisodes = 10000
    FailCount = 0
    while FailCount < MaxEpisodes:
        pointA = rd.randint(0,11)
        pointB = rd.randint(0,11)
        while pointA == pointB : 
            pointB = rd.randint(0,11)
        nextSolution = S.copy()
        nextSolution = Swap(nextSolution , pointA , pointB)
        if GraphDistance(S)>GraphDistance(nextSolution) : 
            S = nextSolution
            FailCount = 0
        else : 
            FailCount += 1
    return S

def currections(S , BestDB):
    SuccessCount = 0
    for i in range(100) : 
        rd.shuffle(S)
        S = optimize(S)
        print(str(S) + str(GraphDistance(S)))
        if(GraphDistance(S) < 12.5) : 
            SuccessCount += 1
            BestDB.append(S)
        elif(GraphDistance(S)<13 and GraphDistance(S)>=12.5) : SuccessCount += 0.8
        elif(GraphDistance(S)<14 and GraphDistance(S)>=12.3) : SuccessCount += 0.4
    print(SuccessCount/100)
    return BestDB

BestDB = []
BestDB = currections(Solution , BestDB)
print(BestDB)
            
    