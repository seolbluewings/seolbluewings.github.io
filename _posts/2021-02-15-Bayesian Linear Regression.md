---
layout: post
title:  "Bayesian Linear Regression"
date: 2021-02-15
author: seolbluewings
categories: Statistics
---

[작성중...]

먼저 논의했던 Gibbs Sampler, Variational Inference를 활용해 통계학에서 가장 자주 접하게 되는 회귀 분석 문제를 해결해보고자 한다. 두 방법을 사용한 회귀분석 풀이는 널리 알려져있는 [LSE 추정 방식](https://seolbluewings.github.io/statistics/2019/04/13/Linear-Regression.html)과는 차이가 있다. Gibbs Sampler, Variational Inference를 사용하는 Bayesian Linear Regression(베이지안 회귀 분석)은 모델에서 활용되는 parameter 값에 대한 Prior Distribution(사전 분포)를 가정하고 데이터 포인트에 대한 Likelihood를 활용해 Posterior Distribution(사후 분포)을 구해서 Parameter에 대한 추정을 Posterior Distribution을 사용해 진행한다.

$$ p(\theta\very y) \propto p(y\vert\theta)p(\theta)$$

널리 알려진 LSE 방식과 Bayesian Linear Regression 방식은 parameter를 바라보는 관점에서의 차이가 있다. LSE는 parameter인 회귀계수 $$\beta$$를 고정된 상수값으로 간주한다. parameter의 점추정값에 대한 오차는 추정값에 대한 신뢰구간(confidence interval)을 통해 반영한다.

반면 Bayesian Linear Regression에서는 parameter $$\beta$$는 상수가 아닌 변수다. 따라서 Bayesian Linear Regression에서는 parameter $$\beta$$에 대한 분포를 활용하게 되며 불확실성은 Posterior Distribution에 대한 신용구간(Credible Interval)을 통해 반영한다.

#### Bayesian Update

데이터 $$ \mathbf{y} = \{y_{1},...,y_{n}\} $$가 $$ \mathbf{y} \sim \mathcal{N}_{n}(X\beta,\sigma^{2}I) $$ 를 따른다고 하자. 그렇다면, 이 문제에서 관심 parameter는 $$ \beta,\sigma $$일 것이다.

관심 parameter에 대한 Posterior Distribution은 다음과 같을 것이다.

$$ p(\beta,\sigma \vert \mathbf{y}) \propto p(\mathbf{y} \vert \beta,\sigma)p(\beta)p(\sigma)$$

새로운 데이터 $$y_{\text{new}}$$가 추가되었다고 할 때, 이 때의 Posterior Distribution은 다음과 같은 형태를 가질 것이다.

$$ p(\beta,\sigma\vert \mathbf{y},y_{\text{new}}) \propto p(\mathbf{y},y_{\text{new}}\vert \beta,\sigma)p(\beta)p(\sigma) $$

$$\mathbf{y}$$,$$y_{\text{new}}$$가 서로 독립적이라고 한다면 Posterior는 다음과 같이 표현될 것이다.

$$
\begin{align}
p(\beta,\sigma\vert \mathbf{y},y_{\text{new}}) &\propto p(y_{\text{new}}\vert\beta,\sigma)p(\mathbf{y}\vert\beta,\sigma)p(\beta)p(\sigma) \\
&propto p(y_{\text{new}}\vert\beta,\sigma)p(\beta,\sigma\vert\mathbf{y})
\end{align}
$$

즉, 수식의 두번째 줄을 통해서 확인할 수 있듯이, 새로운 데이터가 추가되면, 새로운 데이터는 Likelihood 형태로 반영되며 기존의 Posterior Distribution의 경우 Prior의 역할을 수행하게 된다. 이렇게 데이터가 추가될 때, 기존의 Posterior가 새롭게 Prior의 역할을 수행하게 되는 것을 Bayesian Update라고 한다.

![BLR](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Figure3.7.png?raw=true){:width="70%" height="70%"}{: .center}

이 그림은 Bayesian Linear Regression에서 Bayesian Update가 어떠한 역할을 수행하는지 보여준다. 그림의 첫번째 줄은 데이터를 하나도 확인하지 못한 상황을 의미한다. 회귀계수 $$w$$에 대한 Prior가 넓게 분포해 있고 이 사전분포에서 추출한 회귀계수 $$w$$에 대한 Sampling을 수행해서 6개의 서로 다른 회귀분석 식을 만들었다.

두번째 행은 첫번째 데이터가 추가된 경우를 의미한다. 첫번째 열에 있는 그림은 데이터 포인트에 대한 Likelihood를 표현한 것이다. 이 Likelihood와 앞선 행의 Prior를 곱해 두번째 열에 있는 새로운 Posterior 생성한다. 기존보다 parameter $$w$$가 가질 수 있는 값의 범위가 좁혀졌다. 마지막 열은 parameter $$w$$에 대한 Sampling 진행 후, 적합한 회귀식이 데이터 포인트를 얼마나 잘 지나가는가를 보여준다. 앞선 시행과 달리 일단 데이터 포인트 근처를 지나가기 시작한다.

이후는 동일한 과정의 반복이다. 1. 데이터의 추가 2. 기존의 Posterior가 Prior로 작용해서 새로운 Posterior의 생성 3. 회귀식 적합성의 증가의 선순환 구조가 이어진다. 반복시행 할수록 Posterior Distribution의 불확실성이 줄어드는 것을 그림을 통해 확인 가능하다.

#### Bayesian Linear Regression via Gibbs Sampler

#### Bayesian Linear Regression via Variational Inference



