#每次mouse移動時會更新graph , INTgraph , Vistedgraph
#根據INTgraph以及Vistedgraph的資訊老鼠會朝尚未拜訪的非牆壁位置走過去
#INTgraph主要用來判斷要走的位置是否有牆壁
#Visitgraph判斷走的位置是否已經被拜訪過
#如果附近有沒拜拜訪過並且不是牆壁地方就走過去並把vistgraph轉成已拜訪
#如果附近都已經被拜訪過就退回去上一步

#透過RBM function計算出老鼠與終點之間的權重
#將權重當作分數，老鼠會往分數大的地方移動
#老鼠只會走 非牆壁區域 未被拜訪過 分數最高的 地方移動
#先判斷是否會超出地圖 計算前後左右的RBM

#問題 : 無法解決遇到死路後，能夠挑選未拜訪的最佳路徑

import os
import time
import math

graph = ["********", 
         "** * ***",
         "E    ***",
         "* ******",
         "*     **",
         "*****m**"]

def PrintGraph(graph) : 
    time.sleep(0.8)
    os.system("cls")
    print("==========")
    for row in graph : 
        for INT in row : 
            if INT == -1 : print("*" , end="")
            elif INT == 1 : print("M" , end="")
            elif INT == 10 : print("E" , end="")
            else : print(" " , end="")
        print("  |" , end="\n")
    print("==========")

def RBMfunction(mX , mY , dX=2 , dY=0) : 
    gamma = -0.009
    return 1/math.exp(-gamma * ((mX - dX)+(mY - dY))**2)

def InitGraph(graph) : 
    for i in range(6) : 
        graph.append([0 , 0 , 0 , 0 , 0 , 0 , 0 , 0])
    return graph

def ConvertINTgraph(graph) :
    INTgraph = []
    INTgraph = InitGraph(INTgraph)
    x = 0
    y = 0
    for row in graph : 
        for char in row : 
            if char == "*" : 
                INTgraph[x][y] = -1
            elif char == " " : 
                INTgraph[x][y] = 0
            elif char == "m" : 
                INTgraph[x][y] = 1
            elif char == "E" : 
                INTgraph[x][y] = 10
            y += 1
        x += 1
        y = 0
    return INTgraph

def ConvertVistedGraph(graph):
    VG = []
    VG = InitGraph(VG)
    x = 0
    y = 0
    for row in graph : 
        for char in row : 
            if char == "*" : 
                VG[x][y] = -1
            else :  
                VG[x][y] = 0
            y += 1
        x += 1
        y = 0
    return VG

def FindMouse(g) : 
    x = 0
    y = 0
    for row in g : 
        for loc in row : 
            if loc == 1 : return x,y
            y += 1
        x += 1
        y = 0

def DFS(mX , mY , ig , vg) : 
    if ig[mX][mY] == 10 : 
        ig[mX][mY] = 1
        vg[mX][mY] = 1
        PrintGraph(ig)
        print("EEEEEEEEEEEE")
        return
    
    ig[mX][mY] = 1
    vg[mX][mY] = 1
    failcount = 0
    PrintGraph(ig)

    mouseUP = mouseDown = mouseRight = mouseLeft = 0

    if mX+1 <= 5 : 
        if ig[mX+1][mY] != -1 and vg[mX+1][mY] != 1 :  
            mouseDown = RBMfunction(mX+1 , mY)
            print("D" + str(mouseDown))
    if mX-1 >= 0 : 
        if ig[mX-1][mY] != -1 and vg[mX-1][mY] != 1 : 
            mouseUP = RBMfunction(mX-1 , mY)
            print("U" + str(mouseUP))
    if mY+1 <= 7 : 
        if ig[mX][mY+1] != -1 and vg[mX][mY+1] != 1 : 
            mouseRight = RBMfunction(mX , mY+1)
            print("R" + str(mouseRight))
    if mY-1 >= 0 : 
        if ig[mX][mY-1] != -1 and vg[mX][mY-1] != 1 : 
            mouseLeft = RBMfunction(mX , mY-1)
            print("L" + str(mouseLeft))

    maxRBM = max(mouseUP , mouseDown , mouseLeft , mouseRight)
    print(maxRBM)
    print(mX , mY)
    if mX-1>=0 : print("YYYYYY")

    if maxRBM == mouseDown and mX+1 <= 5 : 
        if ig[mX+1][mY] != -1 and vg[mX+1][mY] != 1 : 
            ig[mX][mY] = 0
            DFS(mX+1 , mY , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1
    if maxRBM == mouseUP and mX-1 >= 0 : 
        if ig[mX-1][mY] != -1 and vg[mX-1][mY] != 1 : 
            ig[mX][mY] = 0
            DFS(mX-1 , mY , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1
    if maxRBM == mouseRight and mY+1 <= 7 : 
        if ig[mX][mY+1] != -1 and vg[mX][mY+1] != 1 : 
            ig[mX][mY] = 0
            DFS(mX , mY+1 , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1
    if maxRBM == mouseLeft and mY-1 >= 0 : 
        if ig[mX][mY-1] != -1 and vg[mX][mY-1] != 1 : 
            ig[mX][mY] = 0
            DFS(mX , mY-1 , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1

    if failcount >= 4 :
        ig[mX][mY] = 0
        return


def MouseMoving(graph) : 
    ig = ConvertINTgraph(graph)
    vg = ConvertVistedGraph(graph)
    mX , mY = FindMouse(ig)
    #print(mX, mY)
    DFS(mX , mY , ig , vg)

MouseMoving(graph)