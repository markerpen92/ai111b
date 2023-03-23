#每次mouse移動時會更新graph , INTgraph , Vistedgraph
#根據INTgraph以及Vistedgraph的資訊老鼠會朝尚未拜訪的非牆壁位置走過去
#INTgraph主要用來判斷要走的位置是否有牆壁
#Visitgraph判斷走的位置是否已經被拜訪過
#如果附近有沒拜拜訪過並且不是牆壁地方就走過去並把vistgraph轉成已拜訪
#如果附近都已經被拜訪過就退回去上一步

import os
import time

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

    if mX+1 <= 5 : 
        if ig[mX+1][mY] != -1 and vg[mX+1][mY] != 1 : 
            ig[mX][mY] = 0
            DFS(mX+1 , mY , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1
    if mX-1 >= 0 : 
        if ig[mX-1][mY] != -1 and vg[mX-1][mY] != 1 : 
            ig[mX][mY] = 0
            DFS(mX-1 , mY , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1
    if mY+1 <= 7 : 
        if ig[mX][mY+1] != -1 and vg[mX][mY+1] != 1 : 
            ig[mX][mY] = 0
            DFS(mX , mY+1 , ig , vg)
            ig[mX][mY] = 1
            PrintGraph(ig)
            ig[mX][mY] = 0
        else : failcount += 1
    if mY-1 >= 0 : 
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