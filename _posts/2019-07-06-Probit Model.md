---
layout: post
title:  "Probit model"
date: 2019-07-06
author: YoungHwan Seol
categories: Bayesian
---

프로빗 모형(Probit Model)은 반응변수 Y가 0 또는 1일 때, n개의 반응변수 $$Y_{i}, i=1,...,n$$ 각각에 대하여 독립적으로 $$Ber(p_{i})$$를 가정하여 진행하는 것을 바탕으로 한다.

이 때 $$Y_{i}$$의 기대값인 $$p_{i}$$와 설명변수의 선형결합인 $$x_{i}^{T}\beta$$의 관계를 생각해볼 필요가 있다. $$p_{i}$$는 0과 1사이에서의 값을 가져야 하므로 $$p_{i}=x_{i}^{T}\beta$$의 관계식을 가정하는 것은 적절하지 않다.

이 경우 범위를 $$(0,1)$$로 한정시키는 함수로 정규분포의 누적분포함수인 $$\Phi$$를 사용하여 다음과 같은 관계들을 고려할 수 있으며 이를 프로빗 모형이라고 한다.

$$
\begin{align}
	Y_{i} &\sim Ber(p_{i}) \\
    p_{i} &= \Phi(x_{i}^{T}\beta) \\
    \Phi^{-1}(p_{i}) &= x_{i}^{T}\beta
\end{align}
$$

프로빗 모형에서 관심 모수는 $$\beta$$이며 likelihood는 다음과 같이 구할 수 있다.

$$l(\beta|y) = \prod_{i=1}^{n}\Phi(x_{i}^{T}\beta)^{y_{i}}(1-\Phi(x_{i}^{T}\beta))^{1-y_{i}}$$

관심모수 $$\beta$$에 대해서는 다음의 사전분포를 가정한다.

$$\beta \sim \mathcal{N}(\beta_{0},\Sigma_{0})$$

다만 $$\beta$$에 대한 사전분포와 likelihood의 형태가 서로 conjugate하지 않아 posterior가 편리한 형태로 주어지지 않아 베이지안 추론이 쉽지 않게 된다.

그러나 다음과 같이 잠재변수($$Y_{i}^{*}$$)를 고려하면 Gibbs Sampler를 이용하여 베이지안 추론을 비교적 간단하게 이끌어낼 수 있다. 잠재변수 $$Y_{i}^{*}$$는 다음과 같다.

$$ Y_{i}^{*} &\sim \mathcal{N}(x_{i}^{T}\beta,1) $$

$$
	Y_{i} =	\begin{cases}
    1 & \text{ if $$Y_{i}^{*}>0$$} \\
    0 & \text{ if $$Y_{i}^{*}\leq 0$$}
    \end{cases}
$$


따라서 이항변수 $$Y_{i}=1$$인 사건은 연속형 변수 $$Y_{i}^{*}>0$$인 사건과 동일하며, 다음과 같이 정리할 수 있다.

$$P(Y_{i}=1)=P(Y_{i}^{*}>0)=1-\Phi(-x_{i}^{T}\beta)=\Phi(x_{i}^{T}\beta)$$

$$Y_{i}^{*}$$는 관측변수가 아니므로 모수로 취급하여 새로운 likelihood를 구할 수 있으며 이를 표현하면 다음과 같을 것이다.

$$l(y^{*}|\beta,y) = \prod_{i=1}^{n} \pi(y_{i}^{*}|x_{i}^{T}\beta,1)[I(y_{i}^{*}>0,y_{i}=1)+I(y_{i}^{*}\leq 0,y_{i}=0)]$$











