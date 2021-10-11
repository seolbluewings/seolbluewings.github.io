---
layout: post
title:  "Change Point Detection"
date: 2021-10-10
author: seolbluewings
categories: Statistics
---

[작성중...]

Change Point Detection(CPD) 은 확률 과정(Stochastic Process) 데이터 또는 시계열 데이터의 상태 변화를 탐지하는데 사용하는 알고리즘이며, 이는 데이터의 확률 분포가 급격하게 변화하는 시점을 파악하기 위해 사용하는 모델이라 할 수 있다.

CPD는 앞으로 데이터 변화가 일어날지 탐지하기 위한 예측 목적으로 사용될 수 있고(online method) 이미 데이터가 다 주어진 상황에서 수차례 Change Point가 존재한다고 했을 때, 과연 어느 지점을 Change Point로 보아야하는가?란 답을 얻기 위한 목적(사후분석, offline method)으로 활용할 수 있다.

CPD의 목적은 아래와 같은 데이터가 주어졌을 때, 250이란 지점에서 데이터의 분포가 급격하게 변하게 되었음을 탐지하기 위함이다.

![CPD](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/CPD.png?raw=true){:width="70%" height="70%"}{: .aligncenter}

CPD에 대한 논의를 이어가기 위해서 몇가지 notation에 대한 정의를 내려야 한다.

관측된 데이터 집합을 $$\mathbf{y} = \left\{y_{t}  \right\}^{T}_{t=1}$$ 로 표현하고 관측 데이터의 subset 을 $$y_{a:b} = (y_{a},y_{a+1},...,y_{b})$$ 로 표현한다.

데이터가 총 K개의 point에서 급격하게(abruptly) 변화한다고 가정하자. 이러한 change point의 시점을 표현하는 데이터는 $$\mathcal{T} = \{t_{1},...,t_{K}\}$$ 로 표현하는데 CPD는 바로 이 $$t_{k}$$ 를 찾아내는 것이라 할 수 있다.

여기서 또 하나 중요한 것은 K의 값을 알 수도 있고 모를 수도 있다는 것이다. 만약 K값을 모르는 상황이라면, CPD 모델에서는 K값도 추정해야하는 상황에 직면한다.

$$\mathcal{T} = \{t_{1},...,t_{K}\}$$는 데이터를 segmentation하는 시점에 대한 정보이며 CPD는 가장 데이터를 합리적으로 segmentation하는 지점을 찾는 것이라고 표현할 수도 있다.

objective function $$V(\mathcal{T},\mathbf{y})$$ 을 다음과 같이 정의하자.

$$ V(\mathcal{T},\mathbf{y}) \equiv \sum_{k=0}^{K}c(y_{t_{k}:t_{k+1}}) $$

여기서 $$c(\cdot)$$ 은 cost function이고 이에 따라서 $$V(\mathcal{T},\mathbf{y})$$ 는 각 segmentation의 cost를 합한 것과 같다. 그렇다면, CPD는 $$V(\mathcal{T},\mathbf{y})$$ 를 최소화(minimize) 하는 방향으로 가야한다.

분석 이전 Change Point의 개수 $$K$$에 대한 사전 정보 유무에 따라 objective function $$V(\mathcal{T},\mathbf{y})$$ 를 최소화하는 방법은 2가지로 나뉜다.

1. $$K$$의 값을 알 때,

$$ \text{min}_{\vert\mathcal{T}\vert = K} V(\mathcal{T},\mathbf{y}) $$

2. $$K$$의 값을 모를 때,

$$ \text{min}_{\mathcal{T}} V(\mathcal{T},\mathbf{y}) + \text{pen}(\mathcal{T}) $$

여기서 $$\text{pen}(\mathcal{T})$$ 는 segmentation $$\mathcal{T}$$의 complexity를 측정한 값이다.

[to be continued...]


#### 참조 문헌
1. [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
2. [인공지능 및 기계학습 개론II](https://www.edwith.org/machinelearning2__17/lecture/10868?isDesc=false)
