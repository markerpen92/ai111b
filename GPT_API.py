#自動翻譯器
#如果翻譯字數在50以內就會直翻，如果不會的話就會將原文以及翻譯過後的文章放在一個新檔案裏面

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def getResponse(question) : 
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "user", "content": question},
        ]
    ) 
    return response['choices'][0]['message']['content'] 
    
def ShowInFile(question , response) : 
    FILEIDX = 1
    file = open(str(FILEIDX) + ".txt" , 'w+')
    file.write("user : " + str(question) + "\n================================\n")
    file.write("Gpt : " + str(response))
    FILEIDX += 1

cmd_list = [["tl" , "translate"] , ["exit" , "quite the process"] , 
            ["cmdls" , "show all commands"]]

while True:
    cmdline = input("Enter your commnad : ")
    cmd = cmdline.split()[0]
    if cmd == "tl" : 
        op = cmdline.split()[1]
        content = "".join(cmdline.split()[2:])
        op_list = [["-en" , "convert to English"] , ["-ch" , "convert to Chinese"] ,
                   ["-h"  , "show operations"]    , ["-ls" , "show the dataList"]]
        if op == "-en" : 
            #convert to English
            question = '請幫我把這段話"' + content + '"翻譯成英文'
            response = getResponse(question)
            if len(response)>60 : ShowInFile(question , response)
            else : print(response)
        elif op == "-ch" : 
            #convert to chinese
            question = '請幫我把這段話"' + content + '"翻譯成中文'
            response = getResponse(question)
            if len(response)>60 : ShowInFile(question , response)
            else : print(response)
        elif op == "-h" : 
            #show operations
            for i in op_list : print(i[0] + " : " + i[1])
        elif op == "-ls" : 
            #show the dataList
            print("this operation implement yet")
        else : 
            print("=====Wrong operation=====")
            for i in op_list : print(i[0] + " : " + i[1])
    elif cmd == "cmdls" : 
        print("=====cmd list=====")
        for cmdlist in cmd_list : print(cmdlist[0] + " : " + cmdlist[1])
        print("==================")
    elif cmd == "exit" or cmd == "exit()" : 
        break
    else : 
        print("this cmd implement yet")
        print("=====cmd list=====")
        for cmdlist in cmd_list : print(cmdlist[0] + " : " + cmdlist[1])
        print("==================")