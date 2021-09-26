---
layout: post
title:  "Concept of Hidden Markov Model"
date: 2021-09-17
author: seolbluewings
categories: Statistics
---

[작성중...]

데이터에 대한 분포 가정을 하는 과정에서 우리는 빈번하게 독립적이고 동일한 분포(iid condition)에서 생성된 데이터 집합에 초점을 둔다. 이 iid condition으로 인해 likelihood 값을 각 데이터 포인트에서 계산된 확률의 곱으로 표현이 가능하다.

그러나 실제 사례에서 마주하는 데이터가 iid 조건에 적절하지 않는 데이터일 수도 있다. 예를 들어 주가 예측에서는 전날의 종가가 주어진 상황에서 다음 날의 시초가를 예상해보고자 한다. 다음 영업일의 주식값을 예측하는 상황에서 2영업일 전의 값보다는 1영업일 전의 값이 더 의미가 있을 것이다. 이러한 상황에서는 iid 조건이 적절하지 못하다.

최신 관측값이 더 이전에 관측된 값보다 더 많은 정보를 포함하고 있을 것이라는 직관을 고려한 모델을 마르코프 모델(Markov Model)이라 부른다. Markov Model은 순차적인 데이터에서 iid 가정을 제거한 방법이라 할 수 있다.

미래에 대한 예측을 위해선 가장 최근의 관측값을 제외한 나머지 관측값들에 대해서는 독립적이라 가정하는 1차 Markov Model은 다음과 같이 표현이 될 수 있다.

$$ p(x_{1},x_{2},...,x_{N}) = p(x_{1})\prod_{n=2}^{N}p(x_{n}\vert x_{n-1})$$

이러한 사례에서 조건부 분포 $$ p(x_{n}\vert x_{n-1}) $$ 의 분포는 모든 데이터 포인트에서 동일한 것으로 가정한다. 데이터의 순차성을 고려한다는 점에서 iid 보다는 완화된 형식의 표현이지만 여전히 가정이 존재한다.

1차 Markov Chain은 바로 직전 상태의 데이터에 대해서만 영향을 받는다고 가정했는데 미래의 값에 바로 1단계 전 데이터와 2단계 이전 데이터가 영향을 미친다는 2차 Markov Cahin 은 다음과 같이 표현될 수 있을 것이다.

$$ p(x_{1},x_{2},...,x_{N}) = p(x_{1})p(x_{2}\vert x_{1})  \prod_{n=3}^{N}p(x_{n}\vert x_{n-1},x_{n-2}) $$

이러한 형태로 M차 Markov Chain까지 생성이 가능하다. 그러나 이 차수를 늘릴수록 추정해야할 parameter가 기하급수적으로 증가하여 큰 값에 대한 M차 Markov Chain을 적용하는 것은 비현실적이다.

따라서 우리는 어떠한 차수로든 Markov Chain 가정에 제약되지 않는 순차적인 데이터 모델을 만들길 희망하게 되는데 latent variable $$\mathbf{z}$$ 도입을 통해 우리는 순차 데이터를 표현할 수 있다. latent variable은 관측 변수와 다른 종류이거나 다른 차원을 가질 수 있다.

관측변수가 아닌 latent variable $$\mathbf{z}$$가 Markov Chain을 구성한다고 가정하자. 그렇다면 아래의 그림과 같은 state space model이라 불리는 그래프 구조를 얻을 수 있다.

이 모델은 $$z_{n}$$이 주어진 상태에서 $$z_{n-1}$$과 $$z_{n+1}$$이 독립이라는 conditional independence 을 만족한다.

$$ z_{n+1} \perp\!\!\!\perp z_{n-1} \vert  z_{n} $$

![HMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/HMM1.PNG?raw=true){:width="70%" height="70%"}{: .aligncenter}

그림에 표현된 형태의 모델은 다음과 같이 Factorization이 가능하다.

$$ p(x_{1},\cdots,x_{N},z_{1},\cdots,z_{N}) = p(z_{1})\left[\prod_{n=2}^{N}p(z_{n}\vert z_{n-1})\right]\prod_{n=1}^{N}p(x_{n}\vert z_{n})$$

관측 변수 $$\mathbf{x} = \{x_{1},\cdots,x_{N}\}$$ 은 latent variable $$\mathbf{z}$$를 통해 모두 연결 가능하며 이 경로는 막히지 않는다. 이전 관측값들이 주어졌을 때, 새로운 관측값 $$x_{n+1}$$ 에 대한 예측 분포 $$p(x_{n+1}\vert x_{1},\cdots,x_{n})$$ 은 어떠한 conditional independence 를 보이지 않아 $$x_{n+1}$$ 에 대한 예측은 이전의 관측값에 대해 종속적이게 된다. 이러한 모델에서 latent variable이 discrete한 형태일 때, 은닉 마르코프 모델(Hidden Markov Model)이라 부른다. Hidden Markov Model에서 관측변수 $$\mathbf{x}$$는 continuous, discrete 형태 모두 가능하다.

#### Hidden Markov Model

HMM에는 관측값 $$\mathbf{x}$$ 와 discrete한 형태의 latent variable $$\mathbf{z}$$ 이 있고 조건부 분포 $$p(z_{n} \vert z_{n-1})$$ 을 활용하여 $$z_{n}$$의 확률 분포가 직전의 latent variable $$z_{n-1}$$에 종속적임을 표현한다.

discrete한 latent variable이 총 K차원이라 한다면, 우리는 $$\mathbf{z}$$의 상태가 n-1번째에서는 j이었다가 n번째에서는 k가 될 확률을 알아야만 한다. 그리고 이 확률을 $$A_{jk}$$ 로 표현하고 이를 전이 확률(transition probability)라고 표현한다.

$$ A_{jk} = p(z_{n}=k \vert z_{n-1,}=j), \quad 0\leq A_{jk}\leq 1, \quad \sum_{K} A_{jk} = 1 $$

각 transition probability $$A_{jk}$$를 모으면 이를 하나의 확률 Matrix로 표현가능하며 행렬 $$\mathbf{A}$$ 는 $$K(K-1)$$개의 확률 값을 갖게 된다.

transition probability Matrix $$\mathbf{A}$$를 고려한 조건부 분포는 다음과 같이 표현 가능할 것이다.

$$ p(z_{n}\vert z_{n-1},\mathbf{A}) = \prod_{k=1}^{K}\prod_{j=1}^{K}A_{jk}^{I(z_{n-1}=j)I(z_{n}=k)} $$

그런데 최초 latent variable $$z_{1}$$의 경우는 영향을 미치는 또 다른 latent variable은 존재하지 않는다. 그렇다면, 우리는 $$z_{1}$$의 값을 결정짓는 분포를 가지고 있어야한다. 그 분포는 다음과 같이 결정 짓는다.

$$ p(z_{1}\vert \pi) = \prod_{k=1}^K \pi_{k}^{I(z_{1}=k)}, \quad \sum_{K}\pi_{k}=1 $$

그렇다면 최종적으로 관측변수 $$\mathbf{x}$$ 와 latent variable $$\mathbf{z}$$의 joint distribution은 다음과 같이 표현할 수 있을 것이다. 이 수식에서 $$\theta = (\pi,\mathbf{A},\phi)$$ 를 의미한다. 여기서 $$\phi$$는 $$\mathbf{x}$$ 를 생성하기 위한 분포에서 사용할 parameter이다.

$$ p(\mathbf{x},\mathbf{z}\vert \theta) = p(z_{1}\vert\pi) \left[\prod_{n=2}^{N}p(z_{n}\vert z_{n-1},\mathbf{A})\right]\prod_{m=1}^{N}p(x_{m}\vert z_{m},\phi_{m}) $$

Gaussian Mixture Model(GMM)의 과정을 떠올리면, HMM을 이해하는 것이 한결 쉬워진다. GMM에서는 먼저 parameter인 확률 $$\pi_{k}$$ 를 통해 latent variabled의 class를 결정 짓는다. 그 이후에는 결정된 class의 가우시안 분포로부터 데이터 $$x_{i}$$ 를 생성해냈다. 이러한 과정을 N번 반복해서 N개의 독립된 데이터 집합을 생성한다.

HMM의 경우는 살짝 다르다. 먼저 parameter인 확률 $$\pi_{k}$$에 의해 초기 latent variable $$z_{1}$$의 class를 결정 짓는다. 그리고 $$z_{1}$$ given인 상태에서의 $$x_{1}$$에 대한 sample을 추출한다. 이후 transition probability $$p(z_{2}\vert z_{1},\mathbf{A})$$에 의해 $$z_{2}$$의 class를 선택한다. $$z_{2}$$ 정보에 의해 $$x_{2}$$를 sampling하고 이러한 과정을 반복 수행한다.

#### HMM의 parameter 추정 방식

기본적으로 $$\mathbf{x} = \{x_{1},...,x_{N}\}$$ 이 존재한다면, MLE 방식을 통해 HMM의 parameter를 추정할 수 있을 것이다. 분포 $$p(\mathbf{x},\mathbf{z}\vert \theta)$$의 분포를 marginalized하여 $$p(\mathbf{x}\vert\theta)$$ 를 구할 수 있다.

$$ p(\mathbf{x}\vert\theta) = \sum_{\mathbf{z}}p(\mathbf{x},\mathbf{z}\vert\theta) $$

그러나 각 $$z_{n}$$이 서로 독립적인 관계가 아니라는 것이 marginalized 과정에서의 문제가 되는데 이 HMM 모델의 likelihood를 가장 효율적으로 최대화하는 방식인 EM 알고리즘을 활용하여 parameter를 추정할 수 있다.

EM알고리즘의 Q function은 다음과 같이 표현될 수 있다.

$$ Q(\theta\vert \theta^{(t)}) = \sum_{\mathbf{z}}p(\mathbf{z}\vert \mathbf{x},\theta^{(t)})\log{p(\mathbf{x},\mathbf{z}\vert\theta)} $$

여기서 다음과 같은 2가지 notation을 정의하자.

$$
\begin{align}
\gamma(z_{n}) &= p(z_{n}\vert \mathbf{x},\theta^{(t)}) \nonumber \\
\xi(z_{n-1},z_{n}) &= p(z_{n-1},z_{n}\vert \mathbf{x},\theta^{(t)}) \nonumber
\end{align}
$$

$$I(z_{n}=k)$$ 의 conditional probability는 $$\gamma(z_{n})$$이다. 각 latent variable이 전체 K개 중 어떤 class에 해당하는가? 라는 질문에 대한 기대값은 다음과 같이 표현할 수 있을 것이다.

$$
\begin{align}
\gamma(z_{n}=k) &= \mathbb{E}(z_{n}=k) = \sum_{\mathbf{z}}\gamma(\mathbf{z})I(z_{n}=k) \nonumber \\
\xi(z_{n-1}=j,z_{n}=k) &= \mathbb{E}(z_{n-1}=j,z_{n}=k) = \sum_{\mathbf{z}}\gamma(\mathbf{z})I(z_{n-1}=j)I(z_{n}=k)
\end{align}
$$

이러한 새로운 표현법을 기존의 Q function에 적용하면 Q function을 다음과 같이 바꾸어 표현 가능하다.

$$ Q(\theta\vert\theta^{(t)}) = \sum_{k=1}^{K}\gamma(z_{1}=k)\log{\pi_{k}}+ \sum_{n=2}^{N}\sum_{j=1}^{K}\sum_{k=1}^{K}\xi(z_{n-1}=j,z_{n}=k)\log{A_{jk}} + \sum_{n=1}^{N}\sum_{k=1}^{K}\gamma(z_{n}=k)\log{p(x_{n}\vert \phi_{k})} $$

$$\gamma(z_{n})$$과 $$\xi(z_{n-1},z_{n})$$의 값을 계산해야하는데 이에 대한 계산 방법은 다음의 포스팅에서 소개하기로 한다.

M Step에서는 Q(\theta\vert\theta^{(t)})를 각 $$\theta = (\pi,\mathbf{A},\phi)$$에 대해서 최대화시키며, 이 과정에서 $$\gamma(z_{n}), \xi(z_{n-1},z_{n})$$은 상수 취급하며 $$\pi, \mathbf{A}$$의 최대화는 Lagrange Multiplier를 활용하여 다음과 같은 결과를 얻을 수 있다.

$$
\begin{align}
\pi_{k} &= \frac{ \gamma(z_{1}=k) }{\sum_{j=1}^{K}\gamma(z_{1}=j)} \nonumber \\
A_{jk} &= \frac{ \sum_{n=2}^{N}\xi(z_{n-1}=j,z_{n}=k) }{ \sum_{l=1}^{K}\sum_{n=2}^{N}\xi(z_{n-1}=j,z_{n}=l)} \nonumber
\end{align}
$$

[To be Continued...]



#### 참조 문헌
1. [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
2. [인공지능 및 기계학습 개론II](https://www.edwith.org/machinelearning2__17/lecture/10868?isDesc=false)
