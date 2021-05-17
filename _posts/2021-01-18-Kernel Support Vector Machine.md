---
layout: post
title:  "Kernel Support Vector Machine"
date: 2021-01-18
author: seolbluewings
categories: Statistics
---


앞선 SVM [포스팅](https://seolbluewings.github.io/%EC%84%9C%ED%8F%AC%ED%8A%B8%EB%B2%A1%ED%84%B0%EB%A8%B8%EC%8B%A0/2020/11/29/Support-Vector-Machine.html)에서는 데이터가 선형 hyperplane을 통해 분리가 가능함을 가정하였다. 여기에 Slack Variable을 추가하여 약간의 오차는 눈감아주는 모델도 생성했었다. 그러나 현실적으로 데이터를 선형 hyperplane을 통해 분류해낼 수 있는 경우는 없다. 단순한 XOR 게이트 문제 역시 선형분리시키지 못하는 사례 중 하나다.

이러한 문제를 해결하기 위한 방법으로는 Kernel 기법이 있다. 데이터를 기존의 차원보다 더 높은 차원의 공간으로 mapping 시키고 한층 높아진 차원에서 hyperplane을 이용한 선형분리를 시킨다. 원 데이터셋이 유한한 차원에 존재한다면, 우리는 이를 더 고차원 공간에서 Class를 분류해낼 수 있다.

![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Kernel_SVM.png?raw=true){:width="100%" height="70%"}

선형 hyperplane으로 분리가 불가능한 문제를 데이터 차원을 증가시키면서 해결할 수 있다는걸 위의 그림을 통해서도 확인할 수 있다. 왼쪽 이미지는 원래 데이터 차원에서 그림을 그린 것이다. 두 Class를 선형 hyperplane을 통해서는 분리시킬 수 없다. 그러나 오른쪽 이미지와 같이 가상의 z축을 생성시켜 두 Class 데이터간의 높이 차이를 발생시킨다면, 우리는 이제 선형 hyperplane으로 이 둘을 분리시킬 수 있게 된다.

데이터셋 $$x$$를 고차원으로 mapping시킨 결과를 $$\phi(x)$$라고 한다면, 이 새로운 차원에서의 hyperplane 모델은 다음과 같이 표현 가능하다.

$$ f(x) = w^{T}\phi(x)+b$$

여기서 $$w,b$$는 parameter이고 기존 SVM과 동일하게 margin을 최대화시키는 hyperplane을 생성하기 위해서는 다음의 식을 만족시켜야 한다.

$$
\begin{align}
&\text{min}_{w,b}\frac{1}{2}\vert\vert w\vert\vert^{2} \nonumber \\
&\text{s.t.} y_{i}(w^{T}\phi(x_{i})+b) \geq 1, \quad \forall i \nonumber
\end{align}
$$

이전과 마찬가지 방식으로 Dual Problem으로 바꿔서 표현하면 다음과 같은 식을 얻을 수 있다.

$$
\begin{align}
&\text{max}_{\alpha}\sum_{i=1}^{n}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}\phi(x_{i})^{T}\phi(x_{j}) \nonumber \\
&\text{s.t.} \sum_{i=1}^{n}\alpha_{i}y_{i} = 0 \quad \alpha_{i}\geq 0
\end{align}
$$

이 식의 해를 구하기 위해서는 $$\phi(x_{i})^{T}\phi(x_{j})$$를 계산해야 한다. 그런데 데이터를 활용해 생성한 새로운 차원 $$\phi(x)$$ 에서의 연산은 쉽지 않다. 그래서 우리는 다음과 같은 함수를 가정한다.

$$\kappa(x_{i},x_{j}) = \phi(x_{i})^{T}\phi(x_{j})$$

이러한 함수 $$\kappa(\cdot,\cdot)$$를 Kernel Function이라 부르며, 이러한 연산을 취하는 것을 커널 트릭(Kernel Trick)이라 부른다. 이는 원래 $$x_{i},x_{j}$$가 갖는 차원에서의 Kernel Function의 결과가 $$x_{i},x_{j}$$를 고차원으로 보낸 후 연산시킨 것과 동일하다는 것을 의미한다. 따라서 고차원 공간에서의 벡터 연산을 수행하지 않아도 된다.

자주 활용되는 Kernel을 소개하면 다음과 같다.

|Kernel 명칭|표현식|
|:---:|:---:|
|선형 Kernel|$$\kappa(x_{i},x_{j}) = x_{i}^{T}x_{j} $$|
|다항 Kernel|$$\kappa(x_{i},x_{j}) = (x_{i}^{T}x_{j})^{d}  $$|
|가우시안(RBF) Kernel|$$\kappa(x_{i},x_{j}) = \text{exp}\left(-\frac{\vert\vert x_{i}-x_{j} \vert\vert^{2}}{2\sigma^{2}} \right)  $$|

앞선 Dual Problem을 다시 표현하면 다음과 같이 표현할 수 있다.

$$
\begin{align}
&\text{max}_{\alpha}\sum_{i=1}^{n}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}\kappa(x_{i},x_{j}) \nonumber \\
&\text{s.t.} \sum_{i=1}^{n}\alpha_{i}y_{i} = 0 \quad \alpha_{i}\geq 0
\end{align}
$$

우리가 구하고자 하는 hyperplane 역시 Kernel Function을 통해 다시 표현할 수 있고 이 수식은 최적의 hyperplane은 훈련 데이터를 활용한 Kernel Function을 구함으로써 얻을 수 있다는 것을 의미한다.

$$
\begin{align}
f(x) &= w^{T}\phi(x)+b \nonumber \\
&= \sum_{i=1}^{n}\alpha_{i}y_{i}\phi(x_{i})^{T}\phi(x)+b \nonumber \\
&= \sum_{i=1}^{n}\alpha_{i}y_{i}\kappa(x,x_{i})+b \nonumber
\end{align}
$$

따라서 Kernel Trick에 의해 생성되는 고차원 공간이 데이터를 분리하는데 있어 좋고 나쁨은 Kernel SVM의 성능에 큰 영향을 미친다. 따라서 Kernel Function의 선택은 SVM의 최대 변수가 된다고 말할 수 있다.

#### Gaussian Kernel

가우시안 커널(Gaussian Kernel)은 가장 빈번하게 활용되는 Kernel Function이다. 선형 커널은 기존 SVM에서 활용했던 것이니 특별할 것이 없다고 볼 수 있다. 다항 커널은 사용자가 지정한 d값에 의해 새로운 Space의 차원이 결정된다.

가우시안 커널은 다음과 같이 표현할 수 있는데

$$
\begin{align}
\kappa(x_{i},x_{j}) &\propto \text{exp}\{(x_{i}-x_{j})^{2}\} \nonumber \\
&\propto \text{exp}(-x_{i}^{2})\text{exp}(-x_{j}^{2})\text{exp}(2x_{i}x_{j})
\end{align}
$$

exponential 함수 $$e^{x}$$는 Taylor Series에 의해 다음과 같이 표현될 수 있다.

$$e^{x} = \sum_{k=0}^{\infty}\frac{x^{k}}{k!} $$

이는 exponential 함수를 이용하는 것이 Kernel에 input값으로 들어오는 데이터의 공간을 무한한 차원으로 확장시킬 수 있다는 것을 의미한다. 따라서 가우시안 커널은 입력되는 데이터의 차원에 관계없이 무한대 차원으로 데이터를 mapping시킬 수 있다는 것에서 큰 장점을 지닌 Kernel이라 할 수 있다.


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
