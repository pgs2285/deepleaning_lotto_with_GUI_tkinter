
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from time import sleep
import requests
from bs4 import BeautifulSoup
import datetime as dt
import sys
from tkinter import *
import re
import lxml

def numbers2ohbin(numbers):

    ohbin = np.zeros(45) #45개의 빈 칸을 만듬

    for i in range(6): #여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i])-1] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌
    
    return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):

    numbers = []
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0: # 1.0으로 설정되어 있으면 해당 번호를 반환값에 추가한다.
            numbers.append(i+1)
    
    return numbers
def get_bestNum(nums_prob):
    ball_box=[]
    nums_prob = nums_prob.tolist()
    for i in range(6):
        maxNum = nums_prob.index(max(nums_prob))
        nums_prob[maxNum] = -1
        ball_box.append(maxNum+1)
    return ball_box
def gen_numbers_from_probability(nums_prob): #확률 값을 받고 표시하기

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls

def registNum(number):
    num = PhotoImage(file = f"./ball_photo/{number}.gif")
    label = Label(root, image = num)
    numberLabel.append(label)
    return label

def howManyShow():
    return showList.get()

def showNumber(list):
    for i in range(int(howManyShow())):
        for j in range(6):
            label = registNum(list[j])
            label.place(x=200+50*j, y=50*i+250)
            print(label)



def process():
    numberList = []
    baseURL = 'https://superkts.com/lotto/list/?pg='
    tmpList = []
    count = 1
    bar.set(40)
    progressBar.update()
    for i in range(110):
            r = requests.get(url=baseURL+str(i+1))
            soup = BeautifulSoup(r.content, "lxml")
            table = soup.findAll('span',{'class':re.compile('n[0-9]')})
            for j in table:   
                j = j.get_text()
                if count % 7 != 0:
                    tmpList.append(int(j))

                else: 
                    numberList.append(tmpList.copy());
                    tmpList.clear()
                    pass
                count += 1   

    lottoNumbers = numberList
    ohbins = list(map(numbers2ohbin, lottoNumbers)) # 원-핫 인코딩하기
    train_idx = (0,800)
    val_idx = (801,900)
    test_idx = (900,len(lottoNumbers))
    bar.set(80)
    progressBar.update()
    model = keras.Sequential([
    keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True),
    keras.layers.Dense(45, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   
    x_samples = ohbins[0:len(ohbins)-1]
    y_samples = ohbins[1:len(ohbins)] 
    batch_train_loss = []
    batch_train_acc = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    import os
    if os.path.exists('./base_model/saved_model.pb'):
        model= models.load_model("./base_model/")
    bar.set(160)
    progressBar.update()
    for epoch in range(1):

        model.reset_states() # 중요! 매 에포크마다 1회부터 다시 훈련하므로 상태 초기화 필요

        for i in range(len(x_samples)):
        
            xs = x_samples[i].reshape(1, 1, 45)
            ys = y_samples[i].reshape(1, 45)
            
            loss, acc = model.train_on_batch(xs, ys) 
            # #배치만큼 모델에 학습시킴
            # loss, acc = model.fit(xs,ys)
            batch_train_loss.append(loss)
            batch_train_acc.append(acc)

        train_loss.append(np.mean(batch_train_loss))
        train_acc.append(np.mean(batch_train_acc))

        # print('epoch {0:4d} train acc {1:0.3f} loss {2:0.3f}'.format(epoch, np.mean(self.batch_train_acc), np.mean(self.batch_train_loss)))
        # model.save('./lotto_model/model'+str(dt.date.today()))
    
    model.save("base_model")        
    bar.set(240)
    progressBar.update()
    xs = x_samples[-1].reshape(1, 1, 45)
    ys_pred = model.predict_on_batch(xs)
    print(ys_pred)
    count = 0
    print(labelList)
    for i in tmp:
        i.destroy()

    for i in tmp2:
        i.destroy()
    tmp.clear()
    tmp2.clear()
    loop = int(howManyShow())
    for i in range(loop):
        # a = numList[:]
        # labelList[i].place(x=20, y=50*i+250)
        tmp2.append(Label(root,image=labelList[i]))
        tmp2[i].place(x=50, y=50*i+280)
        numbers = gen_numbers_from_probability(ys_pred[0])
        numbers.sort()
        for j in range(6):
            # print(numbers ,imgList[numbers[j]])
            print(numbers[j])
            tmp.append(Label(root, image = imgList[numbers[j]-1]))
            tmp[j+i*6].place(x=200+50*j, y=50*i+280)
            
            # a[numbers[j]].place(x=200+50*j, y=50*i+250)
            # print(200+50*j, 30*i+250)
            # print(numbers[j])
    bar.set(260)
    progressBar.update()

        

root = Tk() #객체 인스턴스 생성
tmp = []
tmp2 = []
root.title("LOTTO") # 타이틀 설정
root.geometry("602x804") # 크기 설정
root.resizable(False,False) # 크기변경 불가

# Canvas.(root,width-602,height=580).place(x=0,y=)
titleImg = PhotoImage(file="./photo/main.png") # 메인타이틀 이미지
titleLabel = Label(root, image = titleImg) # 라벨에 이미지 삽입
titleLabel.pack() # root 에 배치함

startButtonPhoto = PhotoImage(file="./photo/button.png")
startButton = Button(root, image=startButtonPhoto, command=process)
startButton.pack(side = BOTTOM, anchor='se')

from tkinter import ttk
showList = ttk.Combobox(root, values=[1,2,3,4,5,6,7,8], state="readonly")
showList.current(0)
showList.place(x= 130, y = 720) #place 위치설정

showListCharImg = PhotoImage(file = "./photo/output.png") #label 삽입
showListChar = Label(root, image = showListCharImg)
showListChar.place(x=20,y=700)

bar = DoubleVar()
progressBar = ttk.Progressbar(root, maximum=260, length = 260,variable=bar) #maximum 100짜리 progress bar // indeterminate : 표시기가 처음부터 끝까지 반복 이동
progressBar.place(x=20, y = 770)

labelList = []
imgList = []
for i in range(8):
    labelList.append(PhotoImage(file="./photo/num{}.png".format(i+1)))
for i in range(45):
    imgList.append(PhotoImage(file="./ball_photo/{}.gif".format(i+1)))
print(len(imgList))    

root.mainloop()