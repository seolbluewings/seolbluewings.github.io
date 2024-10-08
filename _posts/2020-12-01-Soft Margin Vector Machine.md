---
layout: post
title:  "Soft Margin by using Slack Variables"
date: 2020-12-01
author: seolbluewings
categories: Statistics
---

앞선 포스팅에서 우리는 서로 다른 클래스의 데이터가 완벽하게 분리되는 OSH $$(\mathbf{w},b)$$가 존재함을 가정했다. 하지만 현실에서는 그러한 OSH가 있다고 보기 어렵다. 설령 그런 OSH가 있다한들 그 OSH는 훈련 데이터에 대한 과적합(overfitting)일 수 있다는 의심부터 하는 것이 바람직하다.

따라서 우리는 현실에 존재할법한, 즉 아래의 이미지와 같이 데이터를 완벽하게 분리시키지는 못하는(nonseparable cases) 경우에도 사용할 수 있는 SVM 모델을 만들어야한다. 그렇다면 우리는 기존의 SVM 모델에서 약간의 오류는 허용하는 방식으로 모델을 수정해야 한다. 이러한 방법은 소프트 마진(soft margin)이란 개념으로 이어진다.

앞선 포스팅에서 언급한 SVM모델은 모든 데이터가 제약조건 $$y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1$$ 을 만족하는 것을 요구한다. 이는 모든 데이터가 정확하게 분리된다는 것을 의미한다. 이러한 상황에서의 margin을 하드 마진(hard margin)이라 부른다. 그렇다면 soft margin은 일부 데이터 포인트에 대해 해당 제약 조건을 요구하지 않는다는걸 의미한다.

그렇다면 우리는 기존처럼 margin을 최대화하는 동시에 제약조건을 만족시키지 못하는 데이터가 존재하되 그 숫자가 최대한 적어지는 방향으로 SVM 모델을 수정해야 한다. 이 때, 우리는 여유 변수(slack variables,$$\xi_{i} \geq 0$$)를 도입하여 기존의 제약 조건을 아래와 같이 변형시킬 수 있다.

$$ y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1-\xi_{i}\quad \forall i$$

각 데이터 $$(x_{i},y_{i})$$ 에 대응되는 $$\xi_{i}$$가 존재하고 이 slack variable은 해당 데이터가 $$y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1$$ 조건을 만족하지 못하는 정도를 표현하기 위한 용도로 활용된다.

아래의 그림과 같이 분류를 위한 OSH는 $$\mathbf{w}^{T}x+b=0$$ 이며 분류가 제대로 되었는지는 $$y_{i}(\mathbf{w}^{T}x_{i}+b)$$ 의 부호를 통해 결정된다. 서포트 벡터인 포인트는 아래의 그림처럼 $$\xi_{i}=0$$이며, OSH를 통해 정확하게 클래스가 분류되나 서포트 벡터보다 OSH에 가까운 포인트는 $$\xi_{i} < 1$$ 이다.

한편, 잘못 분류된 포인트(아래의 그림에서 Misclassified point)에 해당되는 포인트는 $$\xi_{i} >1$$ 값을 갖는다. 즉, 오분류 포인트에 매칭되는 $$\xi_{i}$$값은 $$\xi_{i}>1$$ 이라고 표현할 수 있다. 전체 오분류 개수는 다음과 같은 상한(upper bound)를 가지며 상한을 최소화시킴으로써 전체 오분류 개수를 줄일 수 있다.

$$
\text{total number of misclassification} = \sum_{\xi>1} 1 \leq \sum_{\xi>1}\xi_{i} \leq \sum_{i=1}^{n}\xi_{i}
$$


![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/SVM_2.png?raw=true){:width="100%" height="70%"}

따라서 우리의 최적화 문제(optimization problem)은 다음과 같이 표현될 수 있다. 이 수식은 기존의 포스팅에서 마주했던 수식에 페널티항이 추가된 형태라고 할 수 있다. $$C$$값은 페널티항에 들어가는 regularization paramter이다. $$C$$값이 크다면, 이는 허용하는 오차의 개수가 적다는 것이다. 따라서 margin의 사이즈를 줄이는 결과를 초래한다. 반대로 $$C$$값이 작다면, 이는 허용하는 오차의 개수가 크다는 것이다. 이는 margin의 사이즈를 크게 만드는 결과를 초래한다.

$$
\begin{align}
&\text{minimize}\left\{\frac{1}{2}\vert\vert w \vert\vert^{2} + C\sum_{i=1}^{n}\xi_{i}\right\} \nonumber \\
&\text{subject to}
\begin{cases}
y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1-\xi_{i}\quad \forall i \\
\xi_{i} \geq 0 \quad \forall i
\end{cases}
\end{align}
$$

이에 대한 결과는 최적의 소프트마진 초평면(optimal soft-margin hyperplane)이라 부른다. 기존의 hard-margin처럼 이 역시 라그랑주 함수를 통해 다른 방식으로 표현할 수 있다.

$$
\begin{align}
L(\mathbf{w},b,\xi,\alpha,\beta) &= \frac{1}{2}\vert\vert w \vert\vert^{2} + C\sum_{i=1}^{n}\xi_{i} \nonumber \\
&- \sum_{i=1}^{n}\alpha_{i}\{y_{i}(\mathbf{w}^{T}x_{i}+b)-1+\xi_{i}\} - \sum_{i=1}^{n}\beta_{i}\xi_{i} \nonumber
\end{align}
$$

이 수식에서 모든 $$i$$에 대하여 라그랑주 승수는 $$\alpha_{i} \geq 0$$, $$\beta_{i} \geq 0$$ 조건을 만족해야한다.

Primal Problem인 라그랑주 함수를 Dual Problem으로 바꾸기 위해서는 다음의 KKT 조건이 성립해야 한다.

$$
\begin{cases}
\alpha_{i} \geq 0, \beta_{i} \geq 0 \quad \forall i \\
\\
y_{i}(\mathbf{w}^{T}x_{i}+b) \geq 1- \xi_{i} \; \xi_{i} \geq 0 \quad \forall i \\
\\
\alpha_{i}\{y_{i}(\mathbf{w}^{T}x_{i}+b)-1+\xi_{i}\}=0 \; \beta_{i}\xi_{i}=0 \quad \forall i
\end{cases}
$$

더불어 라그랑주 함수를 각각 $$\mathbf{w},b,\mathbf{\xi}$$ 에 대해 편미분을 수행하면 다음과 같은 결과를 얻는다.

$$
\begin{align}
\frac{\partial L}{\partial \mathbf{w}} &= \mathbf{w}- \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i} = 0 \nonumber \\
\frac{\partial L}{\partial b} &= \sum_{i=1}^{n}\alpha_{i}y_{i} = 0 \nonumber \\
\frac{\partial L}{\partial \mathbf{\xi}} &= C\bf{1}-\mathbf{\alpha}-\mathbf{\beta} = 0 \nonumber
\end{align}
$$

이를 활용하면 라그랑주 함수를 다움과 같이 최적의 $$\alpha, \beta$$를 찾기위한 dual function으로 표현할 수 있다.

$$
\begin{align}
h(\alpha,\beta) &= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j}+C\sum_{i=1}^{n}\xi_{i} \nonumber \\
&-\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j} - \sum_{i=1}^{n}\alpha_{i}y_{i}b + \sum_{i=1}^{n}\alpha_{i}(1-\xi_{i})-\sum_{i=1}^{n}\beta_{i}\xi_{i} \nonumber \\
&= -\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{T}x_{j} + \sum_{i=1}^{n}\alpha_{i}
\end{align}
$$

이 dual function은 다음의 문제와 동치이다. 벡터형태로 표현한 것 뿐이다.

$$
\begin{align}
\text{max}_{0 \leq \alpha \leq C,\; \mathbf{Y}^{T}\alpha} & -\frac{1}{2}\alpha^{T}\mathbf{Y}\mathbf{X}\mathbf{X}^{T}\mathbf{Y}\alpha + \bf{1}^{T}\alpha \nonumber \\
&\text{where} \mathbf{Y} = \text{diag}\{y_{1},...,y_{n}\} \nonumber
\end{align}
$$

이렇게 구한 수식을 이용해서 OSH를 찾는 과정은 다음과 같다. $$\alpha^{*}$$가 dual problem의 솔루션이라 가정하자. 앞선 [포스팅](https://seolbluewings.github.io/%EB%B6%84%EB%A5%98/2020/11/29/Support-Vector-Machine.html)을 참고하면, 우리는 $$\alpha_{i}^{*} >0$$인 경우, 즉 해당 포인트가 서포트 벡터인 경우에 한해서만 값의 의미가 있다는 것을 알 수 있다.

집합 $$S$$를 서포트 벡터의 집합이라 한다면 다음과 같이 OSH의 법선 벡터 $$\mathbf{w}^{*}$$를 구할 수 있다.

$$ \mathbf{w}^{*} = \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i} = \sum_{i \in S} \alpha_{i}^{*}y_{i}x_{i}$$

그리고 다음의 수식 $$\alpha_{i}^{*} + \beta_{i}^{*} = C \; \forall i$$ 이 성립하며 KKT 조건들 중 3번째 요건에 의해 $$\alpha_{i}^{*}>0,\beta_{i}^{*}>0$$인 서포트 벡터에 대해서 $$y_{i}(\mathbf{w}^{T}x_{i}+b)=1$$이 성립한다. 따라서 최적의 OSH를 결정짓기 위한 b값은 임의의 한점을 잡아 계산하거나 앞선 포스팅에서 마찬가지로 설명했듯이 모든 서포트 벡터를 통해 구한 값의 평균값으로 결정지을 수 있다. 


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
