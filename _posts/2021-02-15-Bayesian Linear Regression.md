---
layout: post
title:  "Bayesian Linear Regression"
date: 2021-02-15
author: seolbluewings
categories: Statistics
---

[작성중...]

먼저 논의했던 Gibbs Sampler, Variational Inference를 활용해 통계학에서 가장 자주 접하게 되는 회귀 분석 문제를 해결해보고자 한다. 두 방법을 사용한 회귀분석 풀이는 널리 알려져있는 [LSE 추정 방식](https://seolbluewings.github.io/statistics/2019/04/13/Linear-Regression.html)과는 차이가 있다. Gibbs Sampler, Variational Inference를 사용하는 Bayesian Linear Regression(베이지안 회귀 분석)은 모델에서 활용되는 parameter 값에 대한 Prior Distribution(사전 분포)를 가정하고 데이터 포인트에 대한 Likelihood를 활용해 Posterior Distribution(사후 분포)을 구해서 Parameter에 대한 추정을 Posterior Distribution을 사용해 진행한다.

널리 알려진 LSE 방식과 Bayesian Linear Regression 방식은 parameter를 바라보는 관점에서의 차이가 있다. LSE는 parameter인 회귀계수 $$\beta$$를 고정된 상수값으로 간주한다. parameter의 점추정값에 대한 오차는 추정값에 대한 신뢰구간(confidence interval)을 통해 반영한다.

반면 Bayesian Linear Regression에서는 parameter $$\beta$$는 상수가 아닌 변수다. 따라서 Bayesian Linear Regression에서는 parameter $$\beta$$에 대한 분포를 활용하게 되며 불확실성은 Posterior Distribution에 대한 신용구간(Credible Interval)을 통해 반영한다.

#### Bayesian Update

데이터 $$ \mathbf{y} = \{y_{1},...,y_{n}\} $$가 $$ \mathbf{y} \sim \mathcal{N}_{n}(X\beta,\sigma^{2}I) $$ 를 따른다고 하자. 그렇다면, 이 문제에서 관심 parameter는 $$ \beta,\sigma $$일 것이다.




