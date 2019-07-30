---
layout: post
title:  "Drawing a Player Pass Map"
date: 2019-07-29
author: YoungHwan Seol
categories: Football
---


2018 FIFA 러시아 월드컵 대한민국 vs 독일 경기의 데이터를 가져와 개별 선수의 패스맵과 히트맵을 그리고자 한다. 데이터는 [Statsbomb open resource](https://github.com/statsbomb/open-data)를 활용하였다. 해당 링크에서는 다른 월드컵 경기 데이터 역시 구할 수 있다. 대한민국 vs 독일 경기는 7567번 파일이므로 해당 파일을 주소에서 다운 받아 불러온다.

먼저 필요한 라이브러리를 불러온다.
~~~
%matplotlib inline
import json
from pandas.io.json import json_normalize
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle, ConnectionPatch
from matplotlib.offsetbox import  OffsetImage
#import squarify
from functools import reduce
~~~
json 파일을 다음과 같은 과정을 통해 열어준다.
~~~
with open('E://7567.json') as data_file:
	data = json.load(data_file)
~~~

우선 데이터가 json 파일이므로 보기가 불편해 우리에게 익숙한 데이터 프레임으로 바꾸는 과정이 필요하다.

~~~
df = json_normalize(data, sep = "_")
~~~

이 경기는 대한민국이 의도적으로 라인을 내려서 독일에게 지배권을 내준 경기이기 때문에 독일의 패스 시행횟수가 압도적으로 많다. 따라서 독일 선수들의 데이터를 살펴보는 것이 바람직해 보인다. 따라서 독일 빌드업의 시작점인 2명의 선수, 마츠 후멜스와 토니 크로스의 패스 데이터를 시각화 해보려고 한다.

먼저 다음과 같이 두 선수의 데이터를 불러온다. 후멜스의 player_id는 5572이며 크로스의 player_id는 5574이다. 번호로 불러오는 니클라스 쥘레, 메수트 외질 같은 선수들의 이름 표기가 깨지는 경우가 있기 때문에 player_id로 불러오는 것이 더 낫다.

각 선수에 대해 불러오는 데이터는 period(전,후반 구분표기), timestamp(해당 event가 발생한 경기 시간) location(해당 event에서의 선수 위치), pass_end_location(패스가 끝난 지점), pass_recipient_name(패스를 받는 선수),pass_outcome_name(패스 결과) 이다. 패스미스가 발생한 경우, 패스를 받는 선수는 원래 선수가 의도한 것으로 추측되는 선수의 이름이 적혀있으며, 완전히 벗어난 경우에는 입력이 되어있지 않다. 패스 결과는 성공일 때는 NaN값이며, 실패한 경우 Incomplete, 완전히 벗어나버린 경우 Out 등으로 표기되어 있다.

~~~
hummels_pass = df[(df['type_name'] == "Pass") & (df['player_id']==5572)]
kross_pass = df[(df['type_name'] == "Pass") & (df['player_id']==5574)]
hummels_pass = hummels_pass[["period","player_name", "timestamp", "location", "pass_end_location", "pass_recipient_name","pass_outcome_name"]]
kross_pass = kross_pass[["period","player_name", "timestamp", "location", "pass_end_location", "pass_recipient_name","pass_outcome_name"]]
~~~

앞서 pass_outcome_name의 경우 패스가 성공일 때 NaN이라 하였다. 따라서 이 NaN값을 다른 내용으로 채우기 위한 작업을 진행한다. 여기서는 Success라 표현했는데 Complete라 표기해도 좋다.
~~~
hummels_pass["pass_outcome_name"] = hummels_pass["pass_outcome_name"].fillna("Success")
hummels_pass.head()

kross_pass["pass_outcome_name"] = kross_pass["pass_outcome_name"].fillna("Success")
kross_pass.head()
~~~




![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/heatmap_son.jpg?raw=true)

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

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/picth_image1.jpg?raw=true)

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

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/picth_image2.jpg?raw=true)

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

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/picth_image3.jpg?raw=true)

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

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/picth_image4.jpg?raw=true)

이제 지난 2018 FIFA 러시아 월드컵 대한민국 vs 독일 경기의 데이터를 가져와 앞서 확인한 손흥민의 히트맵을 그린다. 데이터는 Statsbomb open resource[Statsbomb](https://github.com/statsbomb/open-data)를 활용하여 구하였고 경기 이벤트 데이터에서 손흥민에 해당하는 X좌표와 Y좌표를 이용하여 히트맵을 그린다.


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

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/heatmap_son.jpg?raw=true)

