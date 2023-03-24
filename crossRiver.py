#用一個陣列紀錄出發點有哪些人----羊+狼+菜+人
#用一個陣列紀錄終點有哪些人------沒人

#規則----羊與狼在人不在的情況下不能共存(除非都已經到達目的地)
#規則----羊與菜在人不在的情況下不能共存(除非都已經到達目的地)
#規則每次只能載一個東西到終點

#透過Ready將human以及隨機一種拿去搭船Crossing
#搭船到目的地後發現不符合規則，人跟那一種東西返回
#如果剩餘的兩種加入後都會不符合規則就return回第一個物種

import random as rd
import os
import time

start = ["sheet" , "wolf" , "food" , "human"]
destination = []

def printSituation(start , destination) :
    time.sleep(0.8)
    os.system("cls")
    print(start)
    print("~~~~~~~~~~~~~~~~~~~~~~")
    print(destination)

def Crossing(start , destination , sp1 , sp2) :
    if sp2 == "sheet" : 
        if "wolf" not in destination and "food" not in destination : 
            start.remove(sp2)
            destination.append(sp1 , sp2)
            printSituation(start , destination)
            destination.remove(sp1)
            start.append(sp1)
        else : 
            destination.append(sp1 , sp2)
            printSituation(start , destination)
            destination.remove(sp1 , sp2)
            start.append(sp1 , sp2)
            printSituation(start , destination)

def Ready(start , destination) : 
    sp1 = "human"
    start.remove(sp1)
    for sp2 in start : 
        Crossing(start, destination , sp1 , sp2)