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

![CPD](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/CPD.png?raw=true){:width="80%" height="80%"}{: .aligncenter}

CPD에 대한 논의를 이어가기 위해서 몇가지 notation에 대한 정의를 내려야 한다.

관측된 데이터 집합을 $$\mathbf{y} = \left\{y_{t}  \right\}^{T}_{t=1}$$ 로 표현하고 관측 데이터의 subset 을 $$y_{a:b} = (y_{a},y_{a+1},...,y_{b})$$ 로 표현한다.

데이터가 총 K개의 point에서 급격하게(abruptly) 변화한다고 가정하자. 이러한 change point의 시점을 표현하는 데이터는 $$\mathcal{T} = \{t_{1},...,t_{K}\}$$ 로 표현하는데 CPD는 바로 이 $$t_{k}$$ 를 찾아내는 것이라 할 수 있다.

여기서 또 하나 중요한 것은 K의 값을 알 수도 있고 모를 수도 있다는 것이다. 만약 K값을 모르는 상황이라면, CPD 모델에서는 K값도 추정해야하는 상황에 직면한다.

$$\mathcal{T} = \{t_{1},...,t_{K}\}$$는 데이터를 segmentation하는 시점에 대한 정보이며 CPD는 가장 데이터를 합리적으로 segmentation하는 지점을 찾는 것이라고 표현할 수도 있다.

objective function $$V(\mathcal{T},\mathbf{y})$$ 을 다음과 같이 정의하자.

$$ V(\mathcal{T},\mathbf{y}) \equiv \sum_{k=0}^{K}c(y_{t_{k}:t_{k+1}}) $$

여기서 $$c(\cdot)$$ 은 cost function이고 이에 따라서 $$V(\mathcal{T},\mathbf{y})$$ 는 각 segmentation의 cost를 합한 것과 같다. 그렇다면, CPD는 $$V(\mathcal{T},\mathbf{y})$$ 를 최소화(minimize) 하는 방향으로 가야한다.

분석 이전 Change Point의 개수 $$K$$에 대한 사전 정보 유무에 따라 objective function $$V(\mathcal{T},\mathbf{y})$$ 를 최소화하는 방법은 2가지로 나뉜다.

- $$K$$의 값을 알 때,

$$ \text{min}_{\mathcal{T}} V(\mathcal{T},\mathbf{y}) \quad \text{s.t.} \quad \vert\mathcal{T}\vert = K $$

- $$K$$의 값을 모를 때,

$$ \text{min}_{\mathcal{T}} V(\mathcal{T},\mathbf{y}) + \text{pen}(\mathcal{T}) $$

여기서 $$\text{pen}(\mathcal{T})$$ 는 segmentation $$\mathcal{T}$$의 complexity를 측정한 값이다.

CPD를 수행하기 위해서는 Cost Function과 Search Method에 대해 알아야 한다.

#### Cost Function

먼저 Cost Function은 데이터의 subset $$y_{a:b}$$ 가 얼마나 동질적(homogenous)인가? 를 보여주는 수치라고 할 수 있다. 데이터의 subset $$y_{a:b}$$가 동질적이라는 것은 해당 subset에서의 Change Point가 존재하지 않는다는 걸 의미한다.

대표적인 방식은 MLE(Maximum Likelihood Estimation) 이다. MLE 방식을 선택할 때, $$\mathbf{y} = \left\{y_{t}  \right\}^{T}_{t=1}$$ 를 각각 independent한 random variable로 간주한다.

각 Change Point 시점 $$\mathcal{T} = \{t_{1},...,t_{K}\}$$ 별로 시점 $$t_{k}$$에 상응하는 parameter $$\theta_{k}$$ 가 존재하며 objective function $$V(\mathcal{T},\mathbf{y})$$ 를 negative log-likelihood로 정의하였을 때, CPD는 이에 대한 MLE를 취하는 것과 동등해진다.

$$ c(y_{a:b}) = - \text{Sup}_{\theta}\sum_{t=a+1}^{b}\log{f(y_{t}\vert \theta)} $$

그런데 분포를 가정하는 것에서부터 이 방식은 데이터에 대한 사전 지식(prior knowledge)이 필요하다고 볼 수 있다.

이에 대한 대안으로 Mean-Shift 방식이 있다. 이는 데이터 포인트 $$y_{t}$$ 에서 Segmentation 시점마다의 가우시안 분포의 평균을 뺀 것의 L2 norm을 구하는 것이다. $$\bar{y}_{a:b}$$ 는 empirical 하게 $$y_{a:b}$$의 평균으로 대체 가능하다.

이 방식은 패키지의 옵션으로도 만들어져있고 현실적으로 이 방법을 사용하는 것이 바람직해 보인다.

$$ c(y_{a:b}) = \sum_{t=a+1}^{b} \vert\vert y_{t}-\bar{y}_{a:b}\vert\vert^{2}_{2} $$

#### Search Method

Cost Function은 이미 Segmentation이 정해진 상태에서 Cost를 최소화하는 방식을 알아본 것이다. 따라서 이번에는 주어진 Signal(time-series) 데이터의 적절한 Segmentation을 찾는 방법을 알아보고자 한다.

가장 대표적인 Segmentation 방법은 Binary Segmentation이다. 이는 hyperparameter tuning에서의 greed search와 비슷한 느낌을 주는 방식이다.

최초의 Segmentation 지점을 $$\hat{t}_{1}$$ 이라 하면, 다음과 같은 방식으로 그 지점을 찾는다.

$$ \hat{t}_{1} \equiv \text{argmin}_{1\leq t \leq T-1} c(y_{0:t}) + c(y_{t:T}) $$

Cost Function의 합이 가장 작아지는 지점을 계속 찾는 방법이기 때문에 greedy한 방법이라 표현하는 것은 아주 합리적이다. t시점 전후로 데이터를 나누어 Cost Function의 합이 최소화되는 지점을 찾게 되며 이를 그림으로 표현하면 아래와 같을 것이다.

![CPD](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/CPD2.png?raw=true){:width="70%" height="70%"}{: .aligncenter}

먼저 중앙 지점에서 Segmentation이 이루어지며 그 이후에는 첫번째 Subset 내에서 다시 추가적인 Segmentation이 진행된다. 


[to be continued...]


#### 참조 문헌
1. [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
2. [인공지능 및 기계학습 개론II](https://www.edwith.org/machinelearning2__17/lecture/10868?isDesc=false)
