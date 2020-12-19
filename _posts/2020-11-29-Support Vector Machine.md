---
layout: post
title:  "Basic of Support Vector Machine"
date: 2020-11-29
author: seolbluewings
categories: SVM
---

훈련 데이터가 다음과 같은 형태, $$\mathcal{D}= \left\{(x_{1},y_{1}),(x_{2},y_{2}),....,(x_{n},y_{n}) \right\}, y_{i} \in \{-1,1\}$$ 로 주어졌다고 가정하자.

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

![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SVM_3.png?raw=true){:width="70%" height="70%"}

그림을 통해 확인할 수 있듯이, $$(\mathbf{w},b)$$에 가장 가까운 몇가지 데이터 포인트는 $$\mathbf{w}^{T}x_{i}+b = \pm 1$$을 만족하는 점이다. 이를 서포트 벡터(Support Vector)라고 부른다. 두가지 서로 다른 클래스 $$(y_{i}=1,y_{i}=-1)$$의 서포트 벡터에서 $$(\mathbf{w},b)$$에 도달하는 거리의 합은 $$\frac{2}{\vert\vert w \vert\vert}$$ 이며 이를 마진(margin)이라 부른다.

데이터를 가장 잘 분리해내는 hyperplane(Optiaml Seperating Hyperplane, OSH)은 바로 이 margin이 최대화되는 지점에서 생길 것이다. 다만 분류를 정확히 해야한다는 제약 조건이 있으므로 이를 수식으로 표현하면 다음과 같을 것이다.

$$
\begin{align}
&\text{maximize}\frac{2}{\vert\vert w \vert\vert} \nonumber \\
\text{subject to} \quad &
\begin{cases}
\mathbf{w}^{T}x_{i}+b \geq +1 \quad \forall i : y_{i}= +1 \\
\mathbf{w}^{T}x_{i}+b \leq -1 \quad \forall i : y_{i}= -1
\end{cases}
\end{align}
$$

$$\vert\vert w \vert\vert^{-1}$$을 최대화시키는 것은 $$\vert\vert w \vert\vert^{2}$$을 최소화시키는 것과 동일하다. 따라서 다음과 같은 수식으로 표현해도 의미는 변하지 않는다.

$$
\text{minimize}\frac{1}{2}\vert\vert w \vert\vert^{2} \quad \text{subject to} \quad y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1 \; \forall i
$$

이 수식을 통해 최적의 $$(\mathbf{w},b)$$, 즉 OSH를 구하고 싶다. 이는 어떠한 목적함수를 제약조건 하에서 최대/최소화시키는 문제이기 때문에 라그랑주 승수법을 활용해 답을 얻을 수 있다. 위의 수식을 라그랑주 함수로 표현하면 다음과 같다. 라그랑주 함수에 대해서는 [여기](https://ratsgo.github.io/convex%20optimization/2018/01/25/duality/)를 참고하길 바란다.

$$L(\mathbf{w},b,\alpha) = \frac{1}{2} \vert\vert w \vert\vert^{2}-\sum_{i=1}^{n}\alpha_{i}\{y_{i}(\mathbf{w}^{T}x_{i}+b)-1\}, \quad \alpha_{i} \geq 0 $$

라그랑주 함수 $$L(\mathbf{w},b,\alpha)$$를 각 $$\mathbf{w},b$$에 대해 편미분을 하면 다음의 2가지 조건을 얻을 수 있다.

$$
\begin{align}
\frac{\partial L}{\partial \mathbf{w}} &= \mathbf{w}-\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}=0 \nonumber \\
\frac{\partial L}{\partial b} &= -\sum_{i=1}^{n}\alpha_{i}y_{i} = 0 \nonumber \\
\therefore \mathbf{w}&=\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i} \quad \sum_{i=1}^{n}\alpha_{i}y_{i} =0 \nonumber
\end{align}
$$

2가지 조건을 이용해서 라그랑주 함수에서 $$\mathbf{w},b$$를 제거할 수 있고 이를 통해 최대 마진을 구하려는 Primal problem의 Dual Problem 형태로 변경된다.

$$
\begin{align}

h(\alpha) &= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}-\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}+\sum_{i=1}^{n}\alpha_{i} \nonumber \\
&= -\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}+\sum_{i=1}^{n}\alpha_{i} \nonumber \\
&\text{subject to}\quad \sum_{i=1}^{n}\alpha_{i}y_{i}=0\quad \alpha_{i} \geq 0 \; \forall i \nonumber
\end{align}
$$

우리는 기존의 OSH $$(\mathbf{w},b)$$를 다음과 같이 변형해서 표현할 수 있다.

$$f(x) = \mathbf{w}^{T}x+b = \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}^{T}x+b $$

상기 과정들은 다음과 같은 KKT 조건을 만족해야한다. KKT 조건에 대해서는 [여기](https://ratsgo.github.io/convex%20optimization/2018/01/26/KKT/)를 참고하길 바란다.

$$
\begin{cases}
\alpha_{i} \geq 0 \; \forall i \\
\\
y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1 \; \forall i \\
\\
\alpha_{i}\{y_{i}(\mathbf{w}^{T}x_{i}+b)-1\} = 0 \; \forall i
\end{cases}
$$

이 KKT조건들 중 3번째 조건을 잘 살펴볼 필요가 있다. 3번째 조건을 만족하는 solution은 $$\alpha_{i} = 0$$ 또는 $$y_{i}(\mathbf{w}^{T}x_{i}+b)=1$$ 뿐이다. 따라서 $$y_{i}(\mathbf{w}^{T}x_{i}+b)=1$$ 일 때만 $$\alpha_{i} > 0$$일 수 있다. 그런데 $$y_{i}(\mathbf{w}^{T}x_{i}+b)=1$$ 조건을 만족시킨다는 의미는 해당 포인트가 서포트 벡터라는 것이다.

OSH를 다시금 상기해보자. 데이터를 가장 잘 분리해낼 수 있는 OSH, $$\mathbf{w}^{T}x+b=0$$를 발견하는 것이 우리의 목표였다. 법선벡터 $$\mathbf{w}$$와 $$b$$를 찾아내면 우리는 OSH를 구할 수 있다.

법선벡터 $$\mathbf{w}$$는 $$\mathbf{w}=\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}$$로 표현될 수 있다. $$(x_{i},y_{i})$$는 데이터로 주어지기 때문에 우리는 $$\alpha$$만 알면 $$\mathbf{w}$$를 알 수 있다. 수식을 통해서 확인할 수 있듯이, $$\alpha_{i} > 0$$인 것들만 $$\mathbf{w}$$에 영향을 미친다.

$$\alpha_{i}>0$$만 영향을 미치는 것인데 이러한 포인트는 앞서 서포트 벡터 뿐이라는 것을 언급했다. 이는 서포트 벡터 머신의 중요한 성질이다. 대부분의 데이터는 필요가 없게되며 모델은 오로지 서포트 벡터와 관련되기 때문이다.

$$b$$는 서포트 벡터에 대해서 $$y_{i}(\mathbf{w}^{T}x_{i}+b)=1$$ 식을 만족하는 성질을 활용해 계산한다. 임의의 서포트 벡터 $$x_{k}$$를 활용해 $$b$$를 구할 수 있다. 그러나 모든 서포트 벡터에 대해 평균을 내어 $$b$$를 구하는 것이 수치적으로 더욱 안정적인 방법으로 알려져있다.

집합 $$S$$가 서포트 벡터의 index 집합을 의미한다고 했을 때 $$y_{i}(\mathbf{w}^{T}x_{i}+b)=1$$ 수식은 다음과 동치일 것이다.

$$y_{i}\left\{\sum_{i \in S}\alpha_{i}y_{i}x_{i}^{T}x +b \right\} = 1$$

이 수식을 정리하여 평균값을 구하면 $$b$$는 다음과 같은 형태로 구할 수 있다. 여기서 $$N_{S}$$는 서포트 벡터의 총 개수를 의미한다.

$$b = \frac{1}{N_{S}}\sum_{i \in S} \left(y_{i}-\sum_{i \in S}\alpha_{i}y_{i}x_{i}^{T}x \right)$$

지금까지 우리는 가장 단순한 형태의 선형 SVM 형태를 살펴보았다. 단순한 선형결합이 아닌 커널 기법을 통해 SVM의 OSH를 비선형 형태로 확장시킬 수 있다. 더불어 우리는 OSH를 통해 데이터를 완벽하게 분리할 수 있음을 가정했는데 사실 훈련 데이터를 정확하게 분리하도록 모델을 훈련시키는 것은 모델의 일반화 측면에서 좋지 못할 수 있다. 이러한 단점을 보완하기 위해서 slack variable을 도입하여 SVM 모델을 심화시킬 수 있는데 다음 포스팅에서 그 내용을 알아보도록 하자.

#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
