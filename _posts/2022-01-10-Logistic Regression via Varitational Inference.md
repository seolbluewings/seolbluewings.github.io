---
layout: post
title:  "Logistic Regression via Variational Inference"
date: 2022-01-10
author: seolbluewings
categories: Statistics
---

[작성중...]

이전 포스팅에서 Logistic Regression 문제를 Metropolis-Hastings Algorithm을 활용하여 해결한 적이 있다. 이는 Logistic Regression 문제를 Sampling Based Inference 방식 중 하나인 M-H Algorithm을 사용해 해결한 것인데 이번에는 parameter에 대한 posterior에 가장 가까운 분포를 근사하는 Variational Inference를 활용해 Logistic Regression을 풀어보고자 한다. Logistic Regression에 대한 Variational Inference는 Bayesian Neural Network 과정에서도 활용되어 한번 정리해두고 가는 것이 좋아 보인다.

Variational Inference에서는 결국 $$p(y)$$ likelihood의 lower bound를 최대화하게 된다.

$$\log{p(y)} = \text{ELBO} + \text{KL}(q(\theta)\vert p(\theta\vert y)) $$

$$p(y)$$는 다음의 수식을 marginalized한 결과이다.

$$
\begin{align}
p(y) &= \int p(\mathbf{y}\vert\beta)p(\beta)d\beta \nonumber \\
&= \int \left[\prod_{n=1}^{N}p(y_{n}\vert\beta)\right]p(\beta)d\beta
\end{align}
$$

현재 가정된 Logistic Regression 상황을 고려한다면, $$p(\mathbf{y}\vert\beta)$$ 는 $$a = \mathbf{x}\beta$$ 라 할 때 다음과 같이 표현할 수 있다.

$$
p(y\vert\beta) = \sigma(a)^{y}\{1-\sigma(a)\}^{1-y} = \text{exp}(ay)\sigma(-a)
$$

따라서 이러한 상황에서 $$p(\mahtbf{y})$$의 lower bound를 구하기 위해서는 sigmoid function $$\simga(\cdot)$$ 에 대한 variational transform이 필요하며 sigmoid function에 대한 lower bound는 다음과 같다.

$$
\begin{align}
\sigma(z) &\geq \sigma(\xi)\text{exp}\{(z-\xi)/2-\lambda(\xi)(z^{2}-\xi^{2})\} \nonumber \\
\lambda(\xi) = \frac{1}{2\xi}\left[\sigma(\xi)-\frac{1}{2}\right] \nonumber
\end{align}
$$





상기 내용에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Dirichlet%20Process%20Mixture%20Model.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [인공지능 및 기계학습심화](https://www.edwith.org/aiml-adv/joinLectures/14705)
2. [Density Estimation with Dirichlet Process Mixtures using PyMC3](https://austinrochford.com/posts/2016-02-25-density-estimation-dpm.html)