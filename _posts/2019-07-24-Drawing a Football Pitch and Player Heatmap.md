---
layout: post
title:  "Drawing a Football Pitch and Player Heatmap"
date: 2019-07-24
author: YoungHwan Seol
categories: Football
---

모든 내용은 https://fcpython.com을 참고하였음을 먼저 밝힙니다.

아래와 같은 히트맵(Heatmap)을 그리기 이전에 피치(Pitch)를 먼저 그릴 수 있어야 한다.

![Pitch Draw](/images/heatmap_son.jpg)

피치를 그리는 과정은 다음과 같이 골라인(Goal Line)과 터치 라인(Touch Line)을 그리는 것에서 시작한다. 순차적으로 1. 골라인과 터치라인 그리기 2. 센터 서클(Centre Circle) 그리기 3. 페널티 박스(Penalty Box)와 페널티 아크(Penalty Arc) 그리기 4. 모든 과정을 하나의 함수로 만들기 과정으로 진행된다.

~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import seaborn as sns
%matplotlib inline
~~~

먼저 FIFA의 경기장 국제 규격에 따라서 터치라인 길이를 120m로 골라인 길이를 90m로 하여 피치를 그리기로 한다.
~~~
fig=plt.figure(figsize=[8,6])
ax=fig.add_subplot(1,1,1)

plt.plot([0,0],[0,90], color="blue")
plt.plot([0,120],[90,90], color="orange")
plt.plot([120,120],[90,0], color="green")
plt.plot([120,0],[0,0], color="red")
plt.plot([60,60],[0,90], color="pink")
plt.axis('off')
plt.savefig(r'C:\seolbluewings.github.io\images\picth_image1.jpg')
~~~

![Pitch Draw](/images/picth_image1.jpg)

다음은 센터서클을 그리는 과정이다. centreCircle, centreSpot으로 표기된 부분이 센터서클과 센터스팟을 찍는 코드이다.

~~~
fig=plt.figure(figsize=[8,6])
ax=fig.add_subplot(1,1,1)

plt.plot([0,0],[0,90], color="black")
plt.plot([0,120],[90,90], color="black")
plt.plot([120,120],[90,0], color="black")
plt.plot([120,0],[0,0], color="black")
plt.plot([60,60],[0,90], color="black")

centreCircle = plt.Circle((60,45),9.15,color="red",fill=False)
centreSpot = plt.Circle((60,45),0.8,color="blue")

ax.add_patch(centreCircle)
ax.add_patch(centreSpot)
plt.axis('off')
plt.savefig(r'C:\seolbluewings.github.io\images\picth_image2.jpg')
~~~

![Pitch Draw](/images/picth_image2.jpg)

다음으로는 페널티박스와 페널티아크를 그려야 한다.

~~~
fig=plt.figure(figsize=[8,6])
ax=fig.add_subplot(1,1,1)

plt.plot([0,0],[0,90], color="black")
plt.plot([0,120],[90,90], color="black")
plt.plot([120,120],[90,0], color="black")
plt.plot([120,0],[0,0], color="black")
plt.plot([60,60],[0,90], color="black")

#Penalty Area 그리기
plt.plot([16.5,16.5],[65,25],color="black")
plt.plot([0,16.5],[65,65],color="black")
plt.plot([16.5,0],[25,25],color="black")

centreCircle = plt.Circle((60,45),9.15,fill=False)
centreSpot = plt.Circle((60,45),0.8)
ax.add_patch(centreCircle)
ax.add_patch(centreSpot)

#Penalty Arc 그리기
leftArc = Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="red")

ax.add_patch(leftArc)
plt.axis('off')
plt.savefig(r'C:\seolbluewings.github.io\images\picth_image3.jpg')
~~~

![Pitch Draw](/images/picth_image3.jpg)

지금까지의 과정에 반대편 페널티 박스와 6-yard 박스를 추가하여 하나의 함수로 만들면 다음과 같다.

~~~
def createPitch():
    
    #Create figure
    fig=plt.figure(figsize=[8,6])
    ax=fig.add_subplot(1,1,1)

    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,90], color="black")
    plt.plot([0,120],[90,90], color="black")
    plt.plot([120,120],[90,0], color="black")
    plt.plot([120,0],[0,0], color="black")
    plt.plot([60,60],[0,90], color="black")
    
    #Left Penalty Area
    plt.plot([16.5,16.5],[65,25],color="black")
    plt.plot([0,16.5],[65,65],color="black")
    plt.plot([16.5,0],[25,25],color="black")
    
    #Right Penalty Area
    plt.plot([120,103.5],[65,65],color="black")
    plt.plot([103.5,103.5],[65,25],color="black")
    plt.plot([103.5,120],[25,25],color="black")
    
    #Left 6-yard Box
    plt.plot([0,5.5],[54,54],color="black")
    plt.plot([5.5,5.5],[54,36],color="black")
    plt.plot([5.5,0.5],[36,36],color="black")
    
    #Right 6-yard Box
    plt.plot([120,114.5],[54,54],color="black")
    plt.plot([114.5,114.5],[54,36],color="black")
    plt.plot([114.5,120],[36,36],color="black")
    
    #Prepare Circles
    centreCircle = plt.Circle((60,45),9.15,color="black",fill=False)
    centreSpot = plt.Circle((60,45),0.8,color="black")
    leftPenSpot = plt.Circle((11,45),0.8,color="black")
    rightPenSpot = plt.Circle((109,45),0.8,color="black")
    
    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    leftArc = Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="black")
    rightArc = Arc((109,45),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
    
    plt.axis('off')
    
    plt.show()
    
createPitch()
~~~

![Pitch Draw](/images/picth_image4.jpg)

이제 지난 2018 FIFA 러시아 월드컵 대한민국 vs 독일 경기의 데이터를 가져와 앞서 확인한 손흥민의 히트맵을 그린다. 데이터는 Statsbomb open resource를 활용하여 구하였고 경기 이벤트 데이터에서 손흥민에 해당하는 X좌표와 Y좌표를 이용하여 히트맵을 그린다.

~~~
son = data.loc[data["player.id"]==3083,:]
son = son[["period","timestamp","player.id","player.name","type.id","type.name","location.x","location.y"]]
~~~

우선 손흥민에 해당하는 데이터만 불러온 다음 위에서 만든 createPitch함수 안에 sns.kdeplot을 활용하여 히트맵을 그린다. 

~~~
def createPitch():
    
    #Create figure
    fig=plt.figure(figsize=[8,6])
    ax=fig.add_subplot(1,1,1)

    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,90], color="black")
    plt.plot([0,120],[90,90], color="black")
    plt.plot([120,120],[90,0], color="black")
    plt.plot([120,0],[0,0], color="black")
    plt.plot([60,60],[0,90], color="black")
    
    #Left Penalty Area
    plt.plot([16.5,16.5],[65,25],color="black")
    plt.plot([0,16.5],[65,65],color="black")
    plt.plot([16.5,0],[25,25],color="black")
    
    #Right Penalty Area
    plt.plot([120,103.5],[65,65],color="black")
    plt.plot([103.5,103.5],[65,25],color="black")
    plt.plot([103.5,120],[25,25],color="black")
    
    #Left 6-yard Box
    plt.plot([0,5.5],[54,54],color="black")
    plt.plot([5.5,5.5],[54,36],color="black")
    plt.plot([5.5,0.5],[36,36],color="black")
    
    #Right 6-yard Box
    plt.plot([120,114.5],[54,54],color="black")
    plt.plot([114.5,114.5],[54,36],color="black")
    plt.plot([114.5,120],[36,36],color="black")
    
    #Prepare Circles
    centreCircle = plt.Circle((60,45),9.15,color="black",fill=False)
    centreSpot = plt.Circle((60,45),0.8,color="black")
    leftPenSpot = plt.Circle((11,45),0.8,color="black")
    rightPenSpot = plt.Circle((109,45),0.8,color="black")
    
    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)
    
    leftArc = Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="black")
    rightArc = Arc((109,45),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
    
    plt.axis('off')
    sns.kdeplot(son["location.x"],son["location.y"],cmap="magma_r",shade=True,n_levels=30)
    plt.ylim(0,90)
    plt.xlim(0,120)
    plt.title("Son-HeungMin Heatmap")
    
    plt.savefig(r'C:\seolbluewings.github.io\images\heatmap_son.jpg')
    
createPitch()
~~~

이렇게 하면 다음과 같이 손흥민의 히트맵을 얻을 수 있고 다른 선수들의 경우도 마찬가지 방식으로 진행하여 얻을 수 있다.

![Pitch Draw](/images/heatmap_son.jpg)

