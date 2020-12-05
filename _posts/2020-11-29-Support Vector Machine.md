---
layout: post
title:  "서포트 벡터 머신(Support Vector Machine)"
date: 2020-11-29
author: seolbluewings
categories: 분류
---

훈련 데이터 $$\mathcal{D}= \left\{(x_{1},y_{1}),(x_{2},y_{2}),....,(x_{m},y_{m}) \right\}, y_{i} \in \{-1,1\}$$ 의 형태로 주어졌다고 가정하자.

분류 문제의 가장 기본적인 아이디어는 데이터를 서로 다른 클래스로 분리시킬 수 있는 hyperplane(초평면)을 발견해내는 것이다. 하지만 아래의 그림 중 왼쪽과 같이 훈련 데이터를 분리시킬 수 있는 hyperplane의 경우의 수가 여러가지일 때를 생각해보자. 이러한 상황에서 우리는 어떻게 hyperplane을 선택해야할까? 어떻게 hyperplane을 설정하는 것이 최선인가를 고민하게 된다.

![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SVM_1.png?raw=true){:width="100%" height="70%"}

수많은 hyperplane들 중에서 우리는 오른쪽 그림과 같은 두 클래스 사이 정중앙에 위치한 hyperplane을 선택해야할 것 같다. 이 hyperplane은 다른 hyperplane 대비 훈련 데이터의 변동에 대해 가장 robust할 것으로 생각되기 때문이다. 다른 hyperplane과 비교해서 정중앙에 위치한 것이 새롭게 추가될 데이터에 대해 가장 영향이 적을 것이다. 바꿔말하면, 아직 우리가 마주하지 못한 데이터들에 대해 가장 좋은 성능을 가질 것으로 기대가 된다.

데이터가 분포한 공간 상에서 hyperplane은 다음과 같은 선형모델로 표현될 수 있다.

$$\mathbf{w}^{T}\mathbf{x} + b = 0$$



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
