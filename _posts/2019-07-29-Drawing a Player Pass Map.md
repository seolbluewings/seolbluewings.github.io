---
layout: post
title:  "Drawing a Player Pass Map"
date: 2019-07-29
author: seolbluewings
categories: ETC
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

head값을 출력했을 때 보이는 결과는 다음과 같다. 앞서 언급하였듯이 외질, 쥘레의 이름이 깨져있는걸 볼 수 있다.

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/hummels_table.JPG?raw=true)

이제 패스 기록을 가지고 plot을 그려보려 한다. 패스의 시작점은 location을 사용하고 패스 종점은 pass_end_location 데이터를 활용한다. 성공한 패스는 파란색 선으로 성공하지 못한 패스, 즉 동료 선수들(독일 선수)에게 연결되지 못한 패스는 빨간색으로 표기한다.

~~~
fig, ax = plt.subplots()
fig.set_size_inches(9,6)
ax.set_xlim([0,120])
ax.set_ylim([0,80])
for i in range(len(hummels_pass)):
    if hummels_pass.iloc[i]['pass_outcome_name'] == "Success":
        color = "blue"
    else:
        color = "red"
    ax.annotate("", xy = (hummels_pass.iloc[i]['pass_end_location'][0], hummels_pass.iloc[i]['pass_end_location'][1]), 
                xycoords = 'data', xytext = (hummels_pass.iloc[i]['location'][0], hummels_pass.iloc[i]['location'][1]), 
                textcoords = 'data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color=color),)
plt.show()
~~~

이렇게 처리하면 다음과 같은 그림을 얻을 수 있다.

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/pass.jpg?raw=true)

이제는 히트맵을 그리기위한 정보를 불러온다. 앞서 구한 것은 패스를 시도한 경우에 대한 위치 정보만 담겨있어 히트맵을 그리는데 불충분하다. 현재 불러오는 데이터는 공을 받는 이벤트, 상대 선수와 경합하는 상황에 대해서도 모두 포함하고 있어 히트맵을 그리기 적합한 데이터라 할 수 있다.

~~~
hummels_action = df[(df['player_id']==5572)][["player_name","type_name","period", "timestamp", "location"]]

fig, ax = plt.subplots()
fig.set_size_inches(9, 6)

x_coord = [i[0] for i in hummels_action["location"]]
y_coord = [i[1] for i in hummels_action["location"]]

sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)
plt.show()
~~~

이 코드를 시행하면 다음과 같은 결과를 확인할 수 있다.

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/heatmap.jpg?raw=true)

이제 마츠 후멜스의 패스 그림과 히트맵 그림을 얻었는데 이를 경기장 그림 위에 덧칠하는 것이 바람직하다. 경기장을 그리는 function은 다음과 같다.

~~~
def draw_pitch(ax):
    # size of the pitch is 120, 80
    #Create figure

    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,80], color="black")
    plt.plot([0,120],[80,80], color="black")
    plt.plot([120,120],[80,0], color="black")
    plt.plot([120,0],[0,0], color="black")
    plt.plot([60,60],[0,80], color="black")

    #Left Penalty Area
    plt.plot([14.6,14.6],[57.8,22.2],color="black")
    plt.plot([0,14.6],[57.8,57.8],color="black")
    plt.plot([0,14.6],[22.2,22.2],color="black")

    #Right Penalty Area
    plt.plot([120,105.4],[57.8,57.8],color="black")
    plt.plot([105.4,105.4],[57.8,22.5],color="black")
    plt.plot([120, 105.4],[22.5,22.5],color="black")

    #Left 6-yard Box
    plt.plot([0,4.9],[48,48],color="black")
    plt.plot([4.9,4.9],[48,32],color="black")
    plt.plot([0,4.9],[32,32],color="black")

    #Right 6-yard Box
    plt.plot([120,115.1],[48,48],color="black")
    plt.plot([115.1,115.1],[48,32],color="black")
    plt.plot([120,115.1],[32,32],color="black")

    #Prepare Circles
    centreCircle = plt.Circle((60,40),8.1,color="black",fill=False)
    centreSpot = plt.Circle((60,40),0.71,color="black")
    leftPenSpot = plt.Circle((9.7,40),0.71,color="black")
    rightPenSpot = plt.Circle((110.3,40),0.71,color="black")

    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    leftArc = Arc((9.7,40),height=16.2,width=16.2,angle=0,theta1=310,theta2=50,color="black")
    rightArc = Arc((110.3,40),height=16.2,width=16.2,angle=0,theta1=130,theta2=230,color="black")

    #Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)
~~~
~~~
fig=plt.figure()
fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
draw_pitch(ax)
plt.show()
~~~

다음과 같은 결과를 얻을 수 있다.

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/picth_image4.jpg?raw=true)

이제 모든 과정을 합쳐 경기 데이터와 선수의 player_id를 입력했을 때, 선수의 히트맵과 패스맵이 동시에 그려지는 함수를 만든다. 함수는 다음과 같다. 굳이 GER을 붙여 독일 선수들에 대해서만 그릴 수 있다는 듯이 표기한 이유는 실제 경기장에서는 독일과 대한민국의 공격방향이 서로 반대이나 주어진 데이터에서 공격방향이 서로 같게 설정 되어있어 이에 대한 차이를 주기 위함이다.

~~~
def GER_heat_pass_map(data, player_id):
    pass_data = data[(data['type_name'] == "Pass") & (data['player_id'] == player_id)]
    pass_data = pass_data[["period","player_name", "timestamp", "location", "pass_end_location", "pass_recipient_name","pass_outcome_name"]]
    pass_data["pass_outcome_name"] = pass_data["pass_outcome_name"].fillna("Success")
    heatmap_data = data[(data['player_id'] == player_id)][["player_name","type_name","period", "timestamp", "location"]]
    
    fig=plt.figure()
    fig.set_size_inches(9, 6)
    ax=fig.add_subplot(1,1,1)
    draw_pitch(ax)
    plt.axis('off')

    for i in range(len(pass_data)):
        if pass_data.iloc[i]['pass_outcome_name'] == "Success":
            color="blue"
        else:
            color="red"
        ax.annotate("", xy = (pass_data.iloc[i]['pass_end_location'][0], pass_data.iloc[i]['pass_end_location'][1]), 
                xycoords = 'data', xytext = (pass_data.iloc[i]['location'][0], pass_data.iloc[i]['location'][1]), 
                textcoords = 'data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color=color),)
        
    x_coord = [i[0] for i in heatmap_data["location"]]
    y_coord = [i[1] for i in heatmap_data["location"]]
    sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)
    plt.ylim(0, 80) 
    plt.xlim(0, 120)
    plt.show()
~~~

이제 아래의 코드를 시행하면 마츠 후멜스의 히트맵과 패스맵이 같이 그려져있는 그림이 나온다.

~~~
GER_heat_pass_map(df, 5572)
~~~

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/hummels.jpg?raw=true)

마찬가지 방식으로 토니 크로스에 대한 그림도 얻을 수 있다. 

~~~
GER_heat_pass_map(df, 5574)
~~~

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/kroos.jpg?raw=true)

센터백 후멜스의 히트맵이 센터서클에 가까운 점, 후방 플레이메이커인 토니 크로스의 히트맵이 센터서클과 대한민국 페널티박스 사이에 굉장히 집중되어 있다는 사실에 비추어 독일이 주도권을 잡고 경기를 이끌어갔음을 확인할 수 있다. 더불어 독일 공격의 시발점인 크로스의 패스들 중에서 정작 페널티박스 안으로 들어가는 것이 별로 없다는 것을 통해 대한민국 수비가 본인들의 박스를 중심으로 조밀한 수비 라인을 형성했음을 알 수 있다.

대한민국의 경우 다음과 같이 x좌표를 수정해주어야 한다.

~~~
def KOR_heat_pass_map(data, player_id):
    pass_data = data[(data['type_name'] == "Pass") & (data['player_id'] == player_id)]
    pass_data = pass_data[["period","player_name", "timestamp", "location", "pass_end_location", "pass_recipient_name","pass_outcome_name"]]
    pass_data["pass_outcome_name"] = pass_data["pass_outcome_name"].fillna("Success")
    heatmap_data = data[(data['player_id'] == player_id)][["player_name","type_name","period", "timestamp", "location"]]
    
    fig=plt.figure()
    fig.set_size_inches(9, 6)
    ax=fig.add_subplot(1,1,1)
    draw_pitch(ax)
    plt.axis('off')

    for i in range(len(pass_data)):
        if pass_data.iloc[i]['pass_outcome_name'] == "Success":
            color="blue"
        else:
            color="red"
        ax.annotate("", xy = (120-pass_data.iloc[i]['pass_end_location'][0], pass_data.iloc[i]['pass_end_location'][1]), 
                xycoords = 'data', xytext = (120-pass_data.iloc[i]['location'][0], pass_data.iloc[i]['location'][1]), 
                textcoords = 'data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color=color),)
        
    x_coord = [i[0] for i in heatmap_data["location"]]
    y_coord = [i[1] for i in heatmap_data["location"]]
    x_coord = np.repeat(120,len(x_coord))-x_coord
    sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)
    plt.ylim(0, 80) 
    plt.xlim(0, 120)
    plt.show()
~~~

대한민국 레프트백 홍철의 패스맵과 히트맵은 다음과 같다. 피치를 2등분했을 때, 대한민국 진영에 머물러있는 것으로 확인된다. 본래 홍철이 굉장히 공격적인 유형의 선수임을 고려하면 이 경기에서 대한민국이 의도적으로 라인을 내리고 점유율을 포기하면서 독일을 상대했음을 확인할 수 있다.

~~~
KOR_heat_pass_map(df, 5986)
~~~

![Pitch Draw](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/chul.jpg?raw=true)



##### 모든 내용은 다음의 [링크](https://github.com/tuangauss/Various-projects/blob/master/Python/football_visual.ipynb?source=post_page---------------------------)를 참고하였음을 밝힙니다.