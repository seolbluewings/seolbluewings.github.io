---
layout: post
title:  "Posterior Predictive Distribution"
date: 2020-12-22
author: seolbluewings
comments : true
categories: Bayesian
---

[작성중...]

앞서 소개했던 Bayesian Linear Regression [포스팅](https://seolbluewings.github.io/bayesian/2019/04/22/Bayesian-Linear-Regression.html)을 통해서 우리는 기존의 Frequentist들의 회귀계수 $$\beta$$에 대한 추정방법과 다른 Bayesian 방식을 학습하였다. 그러나 분석 목적에 따라 $$\beta$$를 추정하는 것보다 $$\beta$$ 값을 이용하여 새로운 독립적인 데이터 $$x_{new}$$가 주어졌을 때, 예측값인 $$\tilde{y_{new}}$$ 를 구하는 것이 더 중요할 수도 있다. 앞으로 편의상 $$y$$에 대한 예측값은 $$\tilde{y}$$로 표현하겠다.

$$\tilde{y}$$를 추정하기 위해서 Bayesian이 취할 수 있는 방식은 $$\tilde{y}$$에 대한 분포를 구하는 것일거다. 이를 $$\tlide{y}$$에 대한 예측 분포(Predictive Distribution)이라 부르며 일반적으로 Posterior Predictive Distribution이라 말한다.

#### Posterior Predictive Distribution

$$\tilde{y}$$의 Posterior Predictive Distribution에 대한 일반적인 수식은 다음과 같다.

$$
\begin{align}
\it{p}(\tilde{y}\vert y) &= \int \it{p}(\tilde{y},\theta \vert y)d\theta \nonumber \\
&= \int \it{p}(\tilde{y}\vert\theta,y)p(\theta\vert y)d\theta \nonumber \\
&= \int \it{p}(\tilde{y}\vert\theta)p(\theta\vert y)d\theta \nonumber
\end{align}
$$





#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>
