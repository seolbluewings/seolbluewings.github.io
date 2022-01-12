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

따라서 이러한 상황에서 $$p(\mathbf{y})$$의 lower bound를 구하기 위해서는 sigmoid function $$\sigma(\cdot)$$ 에 대한 variational transform이 필요하며 sigmoid function에 대한 lower bound는 다음과 같다.

$$
\begin{align}
\sigma(z) &\geq \sigma(\xi)\text{exp}\{(z-\xi)/2-\lambda(\xi)(z^{2}-\xi^{2})\} \nonumber \\
\lambda(\xi) &= \frac{1}{2\xi}\left[\sigma(\xi)-\frac{1}{2}\right] \nonumber
\end{align}
$$

따라서 $$p(\mathbf{y}\vert\beta)$$ 는 다음과 같은 lower bound를 갖는다.

$$ p(\mathbf{y}\vert\beta) = \text{exp}(a\mathbf{y})\sigma(-a) \geq \text{exp}(a\mathbf{y})\text{exp}\{(-a+\xi)/2-\lambda(\xi)(a^{2}-\xi^{2})\} $$

그런데 이 수식은 Variational Inference의 가장 대표적인 Mean-Field 가정에 의해 각각의 관측 데이터 $$(x_{n},y_{n})$$ 에 대해서도 적용 가능하며 그 때마다의 데이터 포인트 $$\xi_{n}$$ 역시 존재한다.

$$
\begin{align}
p(\mathbf{y}\vert\beta)p(\beta) &\geq h(\beta,\mathbf{\xi})p(\beta) \nonumber \\
h(\beta,\mathbf{\xi}) &= \prod_{n=1}^{N}\sigma(\xi_{n})\text{exp} \left\{ x_{n}^{T}\beta y_{n} - (x_{n}^{T}\beta+\xi_{n})/2 -\lambda(\xi_{n})([\x_{n}^{T}\beta]^{2}-\xi_{n}^{2}) \right\}
\end{align}
$$

타겟 $$\mathbf{y}$$ 와 $$\beta$$의 joint distribution의 closed form 계산이 어려운 관계로 대신 우측의 식을 사용하여 $$q(\beta)$$ 분포를 근사하게 된다.

로그를 취해도 부등호 방향은 변하지 않기 때문에 위의 수식은 다음과 같이 변할수 있다.

$$
\begin{align}
\log{p(\mathbf{y}\vert\beta)p(\beta)} &\geq \log{p(\beta)} + \sum_{n=1}^{N}\left\{\log{\sigma(\xi_{n})} + x_{n}^{T}\beta y_{n} - (x_{n}^{T}\beta+\xi_{n})/2 - \lambda(\xi_{n})([x_{n}^{T}\beta]^{2}-\xi_{n}^{2}) \right\} \nonumber \\
&\geq -\frac{1}{2}(\beta-\mu_{0})^{T}\Sigma_{0}^{-1}(\beta-\mu_{0}) + \sum_{n=1}^{N}\left\{x_{n}^{T}\beta(y_{n}-1/2)-\lambda(\xi_{n})\beta(x_{n}x_{n}^{T})\beta^{T}\right\} + C
\end{align}
$$

따라서 $$p(\mathbf{y},\beta)$$ 의 lower-bound 에 대한 분포는 $$\beta$$ 에 대한 2차 함수 형태를 가지므로 $$q(\beta)$$를 가우시안 분포 형태로 구할 수 있게 된다.

$$
\begin{align}
q(\beta) &\sim \mathcal{N}(\beta \vert \mu_{N},\Sigma_{N}) \nonumber \\
\Sigma^{-1}_{N} &= \Sigma^{-1}_{0} + 2\sum_{n=1}^{N}\lambda(\xi_{n})x_{n}x_{n}^{T} \nonumber \\
\mu_{N} &= \Sigma_{N}\left(\Sigma_{0}^{-1}\mu_{0} + \sum_{n=1}^{N}(y_{n}-1/2)x_{n}\right)
\end{align}
$$

Variational Family 분포를 결정하였으니 새로운 데이터가 추가되었을 때의 Predictive Distribution을 계산할 수 있어야 한다.







상기 내용에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Dirichlet%20Process%20Mixture%20Model.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [인공지능 및 기계학습심화](https://www.edwith.org/aiml-adv/joinLectures/14705)
2. [Density Estimation with Dirichlet Process Mixtures using PyMC3](https://austinrochford.com/posts/2016-02-25-density-estimation-dpm.html)