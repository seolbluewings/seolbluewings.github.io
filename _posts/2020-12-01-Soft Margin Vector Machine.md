---
layout: post
title:  "여유 변수를 활용한 소프트 마진(Soft Margin by using Slack Variables)"
date: 2020-12-01
author: seolbluewings
categories: 분류
---

[작성중]

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

따라서 우리의 최적화 문제(optimization problem)은 다음과 같이 표현될 수 있다.

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


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
