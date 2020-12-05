---
layout: post
title:  "서포트 벡터 머신(Support Vector Machine)"
date: 2020-11-29
author: seolbluewings
categories: 분류
---

훈련 데이터가 다음과 같은 형태, $$\mathcal{D}= \left\{(x_{1},y_{1}),(x_{2},y_{2}),....,(x_{m},y_{m}) \right\}, y_{i} \in \{-1,1\}$$ 로 주어졌다고 가정하자.

분류 문제의 가장 기본적인 아이디어는 데이터를 서로 다른 클래스로 분리시킬 수 있는 hyperplane(초평면)을 발견해내는 것이다. 하지만 아래의 그림 중 왼쪽과 같이 훈련 데이터를 분리시킬 수 있는 hyperplane의 경우의 수가 여러가지일 때를 생각해보자. 이러한 상황에서 우리는 어떻게 hyperplane을 선택해야할까? 어떻게 hyperplane을 설정하는 것이 최선인가를 고민하게 된다.

![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SVM_1.png?raw=true){:width="100%" height="70%"}

수많은 hyperplane들 중에서 우리는 오른쪽 그림과 같은 두 클래스 사이 정중앙에 위치한 hyperplane을 선택해야할 것 같다. 이 hyperplane은 다른 hyperplane 대비 훈련 데이터의 변동에 대해 가장 robust할 것으로 생각되기 때문이다. 다른 hyperplane과 비교해서 정중앙에 위치한 것이 새롭게 추가될 데이터에 대해 가장 영향이 적을 것이다. 바꿔말하면, 아직 우리가 마주하지 못한 데이터들에 대해 가장 좋은 성능을 가질 것으로 기대가 된다.

데이터가 분포한 공간 상에서 hyperplane은 다음과 같은 선형모델로 표현될 수 있다. 앞으로 이 hyperplane을 $$(\mathbf{w},b)$$로 표기할 것이다. 여기서 $$\mathbf{w}=(w_{1},...,w_{d})$$ 는 법선 벡터로 이 벡터가 $$(\mathbf{w},b)$$의 방향을 설정한다.

$$\mathbf{w}^{T}\mathbf{x} + b = 0$$

위와 같이 $$(\mathbf{w},b)$$가 주어진 상황에서 우리는 임의의 한점 $$x_{0}$$로부터 $$(\mathbf{w},b)$$까지의 거리가 자연스럽게 궁금해질 것이다. 임의의 한점에서 평면까지의 거리(r)를 구하는 수식은 [여기](https://m.blog.naver.com/PostView.nhn?blogId=bjjang3352&logNo=70102475166&proxyReferer=https:%2F%2Fwww.google.com%2F)를 참고하여 표현할 수 있다.

$$r = \frac{|\mathbf{w}^{T}x_{0}+b|}{||w||}$$

만약, $$(\mathbf{w},b)$$이 데이터를 정확하게 분류할 수 있다면 이는 다음과 동치일 것이다.

$$
\begin{align}
\mathbf{w}^{T}x_{i}+b &\geq 1 \quad y_{i}=1 \nonumber \\
\mathbf{w}^{T}x_{i}+b &\leq -1 \quad y_{i}=-1 \nonumber
\end{align}
$$

![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SVM_2.png?raw=true){:width="100%" height="70%"}

그림을 통해 확인할 수 있듯이, $$(\mathbf{w},b)$$에 가장 가까운 몇가지 데이터 포인트는 $$\mathbf{w}^{T}x_{i}+b = \pm 1$$을 만족하는 점이다. 이를 서포트 벡터(Support Vector)라고 부른다. 두가지 서로 다른 클래스$$(y_{i}=1,y_{i}=-1)$$의 서포트 벡터에서 $$(\mathbf{w},b)$$에 도달하는 거리의 합은 $$\frac{2}{||w||}$$이며 이를 마진(margin)이라 부른다.

데이터를 가장 잘 분리해내는 hyperplane(Optiaml Seperating Hyperplane, OSH)은 바로 이 margin이 최대화되는 지점에서 생길 것이다. 다만 분류를 정확히 해야한다는 제약 조건이 있으므로 이를 수식으로 표현하면 다음과 같을 것이다.

$$
\begin{align}
\text{max}\frac{2}{||w||} & \nonumber \\
\text{subject to} &
\begin{cases}
\mathbf{w}^{T}x_{i}+b \geq 1 \quad \forall i : y_{i}= 1 \\
\mathbf{w}^{T}x_{i}+b \leq -1 \quad \forall i : y_{i}= -1
\end{cases}
\end{align}
$$

$$||w||^{-1}$$을 최대화시키는 것은 $$||w||^{2}$$을 최소화시키는 것과 동일하다. 따라서 다음과 같은 수식으로 표현해도 의미는 변하지 않는다.

$$
\text{min}\frac{1}{2}||w||^{2} \quad \text{subject to} \quad y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1 \; \forall i
$$

이 수식을 통해 최적의 $$(\mathbf{w},b)$$, 즉 OSH를 구하고 싶다. 이는 어떠한 목적함수를 제약조건 하에서 최대/최소화시키는 문제이기 때문에 라그랑주 승수법을 활용해 답을 얻을 수 있다. 위의 수식을 라그랑주 함수로 표현하면 다음과 같다. 라그랑주 함수에 대해서는 [여기](https://ratsgo.github.io/convex%20optimization/2018/01/25/duality/)를 참고하길 바란다.

$$L(\mathbf{w},b,\alpha) = \frac{1}{2}||w||^{2}-\sum_{i=1}^{m}\alpha_{i}\{y_{i}(\mathbf{w}^{T}x_{i}+b)-1\}, \alpha_{i} \geq 0 $$



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
