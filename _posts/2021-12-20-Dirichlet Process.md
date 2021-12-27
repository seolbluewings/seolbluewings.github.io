---
layout: post
title:  "Dirichlet Process"
date: 2021-12-20
author: seolbluewings
categories: Statistics
---

[작성중...]

Dirichlet Process는 Dirichlet Distribution을 따르는 Random Process로 Unsupervised Learning에 자주 활용된다. 기존에 학습했던 Unsupervised Learning은 GMM, K-Means와 같은 Clustering 문제였다. 기존 GMM, K-Means 문제에서는 cluster의 개수 k를 분석가가 명시적으로 지정해야하는, 사람이 개입해야하는 이슈가 있었다. 이러한 문제에서 조금 더 자유로워지고자 할 때, cluster의 개수 k를 사람이 명시적으로 지정하지 않는 방식을 취하고자 할 때 선택할 수 있는 옵션이 Dirichlet Process라고 할 수 있다.

Dirichlet Process 이전에 Dirichlet Distribution과 Multinomial Distribution에 대해 짚고 넘어갈 필요가 있다.

#### Dirichlet Distribution & Multinomial Distribution

Multinomial Distribution은 어떠한 시행에서 M가지 값이 나올 수 있고 각각의 값이 나올 확률을 $$\theta_{1},...,\theta_{M}$$ 이라 할 때, N번의 시행에서 i번째 값이 $$x_{i}$$번 나타날 확률을 표현한다.

$$ p(\mathbf{x}\vert \mathbf{\theta}) = \frac{N!}{\prod_{i=1}^{M}x_{i}!}\prod_{i=1}^{M}\theta_{i}^{x_{i}} $$

한편, Dirichlet Distribution은 K차원의 continuous한 확률 변수를 return하는 확률 분포로 2이상의 자연수 K와 양수 $$\alpha_{1},...,\alpha_{K}$$, 양의 실수 $$\theta_{1},...,\theta_{K}$$ 가 $$\sum_{i=1}^{K}\theta_{i} = 1$$ 을 만족할 때, 다음과 같은 확률 분포를 갖는다.

$$ p(\theta \vert \alpha) = \frac{\Gamma\left(\sum_{i=1}^{K}\alpha_{i}\right)}{\prod_{i=1}^{K}\Gamma(\alpha_{i})} \prod_{i=1}^{K}\theta_{i}^{\alpha_{i}-1}  $$

Dirichlet Distribution으로 얻을 수 있는 결과 $$\theta = (\theta_{1},...,\theta_{K})$$ 가 Probability의 공리 조건들을 만족하기 때문에 우리는 Dirichlet Distribution의 확률 변수를 Multinomial Distribution의 parameter로 활용할 수 있다.

이 2개의 분포는 서로 Conjugate한 관계를 지니고 있고 Dirichlet Distribution이 Prior로서의 역할, Multinomial Distribution이 Likelihood로의 역할을 하여 Dirichlet Distribution 형태의 Posterior가 도출된다.

$$
\begin{align}
p(\theta \vert \mathbf{x},\alpha) &\propto p(\theta\vert\alpha)p(\mathbf{x}\vert\theta) \nonumber \\
&\propto \frac{\Gamma\left(\sum_{i=1}^{K}\alpha_{i}\right)}{\prod_{i=1}^{K}\Gamma(\alpha_{i})} \prod_{i=1}^{K}\theta_{i}^{\alpha_{i}-1} \times \frac{N!}{\prod_{i=1}^{M}x_{i}!}\prod_{i=1}^{M}\theta_{i}^{x_{i}} \nonumber \\
&\propto \frac{\Gamma\left(\sum_{i=1}^{K}\alpha_{i}+x_{i}\right)}{\prod_{i=1}^{K}\Gamma(\alpha_{i}+x_{i})} \prod_{i=1}^{K}\theta_{i}^{\alpha_{i}+x_{i}-1} \nonumber
\end{align}
$$

이러한 관계에 따라 Dirichlet Distribution이 Prior로 Multinomial Distribution에 영향을 미치고 이 때, 데이터가 하나 Sampling 된다면 이 결과는 Posterior에 반영되어 Dirichlet Distribution을 업데이트하는 것으로 받아들일 수 있다.


#### Definition of Dirichlet Process

Dirichlet Process(이하 DP)는 discrete한 distribution $$G$$에 대한 flexible한 probability distribution이라 볼 수 있다. 이 probability distribution $$G$$는 다음의 조건을 만족시키는 probability measure $$\Omega$$에서 discrete 형태로 정의된다.

$$
\begin{align}
\Omega &= S_{1} \cup S_{2} \cup \cdots \cup S_{r} \nonumber \\
\phi &= S_{1} \cap S_{2} \cap \cdots \cap S_{r} \nonumber
\end{align}
$$

$$\Omega$$는 확률이 정의되는 Space로 인식하는 것이 이해하는데 직관적이며 확률이 정의되는 공간이 서로 겹치지 않는 상태로 r개의 공간으로 완전하게 나뉘어진다고 한다.

이러한 상황에서 Dirichlet Distribution은 다음과 같이 표현할 수 있다.

$$
\begin{align}
G\vert\alpha,H &\sim DP(\alpha, H) \nonumber \\
\left(G(A_{1}),...,G(A_{r})\right)\vert\alpha,H &\sim \text{Dirichlet}\left(\alpha H(A_{1}),...,\alpha H(A_{r})\right) \nonumber
\end{align}
$$

여기서 $$\alpha$$는 기존 Dirichlet Distribution에서 사용했던 parameter이며, H는 $$\Omega$$의 공간에서의 base distribution으로 DP는 기존의 Dirichlet Distribution의 parameter에 pdf를 곱해준 것이라 볼 수 있다.

DP를 통해 생성되는 $$\theta_{1},...,\theta_{n}$$ 도 DP를 따르며 아래와 같은 방식으로 표현할 수 있다.

$$ G\vert\theta_{1},...,\theta_{n} \sim DP\left(\alpha + n, \frac{\alpha}{\alpha+n}H + \frac{1}{\alpha+n}\sum_{i=1}^{n}\delta_{\theta_{i}}(S)\right) $$

새로운 observation인 $$\theta \vert \theta_{1},...,\theta_{n}$$ 에 대한 posterior predictive distribution은 기존의 base function H와 관측값이 결합된 형태이다

이는 앞서 소개했던 Multinomial Distribution와 Dirichlet Distribution의 Conjugate 성질을 고려하면 받아들일 수 있는 구조이다.

여기서 $$\delta$$ func은 DP에서 sampling된 $$\theta_{i}$$ 가 Space $$S$$에 포함되어 있는지 여부에 따라 값이 결정되는 함수이다.

$$
\delta_{\theta_{i}}(S) = \begin{cases}
1 \quad \text{if} \quad \theta_{i} \in S \\
0 \quad \text{if} \quad \theta_{i} \notin S
\end{cases}
$$

probability measure $$\Omega$$가 r개의 구역으로 나뉘어지는데 $$\theta_{i}$$가 r개의 구역 중 어느 한 곳에 속하는가를 표현하는 것으로 보면 되고 DP는 이 r이란 값을 무한대로 늘리는 과정까지 포함한다.

Probability Measure Space $$\Omega$$를 r개로 나누었는데 이를 $$r \to \infty$$ 로 확장시켜보자. 이는 $$\Omega$$를 r개의 disjoint한 공간으로 분할시키는 것으로 이해할 수 있다.

$$\pi_{r}$$은 r번째 공간에 분류될 확률이며 $$\sum_{r=1}^{\infty}\pi_{r} = 1$$ 이 성립한다. $$\theta_{r}$$은 r번째 공간의 평균값으로 받아들일 수 있어 DP는 다음과 같이도 표현할 수 있다.

$$ G(S) = \sum_{r=1}^{\infty}\pi_{r}\delta_{\theta_{r}}(S) $$

DP를 통한 Sampling을 실질적으로 수행하기 위해서는 무한대 차원에서 $$\pi_{r},\theta_{r}$$를 만들어낼 방법이 먼저 필요하다고 볼 수 있다. 이에 대한 이론적 배경으로는 Stick Breaking Process, Poly-Urn Scheme, Chinese Restaurant Process가 있다.

#### Stick Breaking Process

Stick Breaking Process 무한대 차원에서의 Probability Distribution을 정의하기 위해 사용되는 방법이다. 이 과정을 통해 우리는 $$\pi_{k}$$와 $$\theta_{k}$$ 에 대한 생성을 그림 그릴 수 있다.

$$k=1,2,...,\infty$$ 인 상황에서 $$v_{k}$$가 다음의 Beta분포를 따르며 $$\pi_{k}$$가 $$v_{k}$$ 로 인해 정의된다고 하자.

$$
\begin{align}
v_{k}\vert\alpha &\sim \text{Beta}(1,\alpha) \nonumber \\
\pi_{k} &= v_{k}\prod_{i=1}^{k-1}(1-v_{i}) \nonumber \\
\end{align}
$$

이 수식은 아래와 같은 길이가 1인 막대가 있다고 가정했을 때 이해가 한결 쉬워진다.

![DP](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/DP2.png?raw=true){:width="70%" height="70%"}{: .aligncenter}




포스팅 내용에 대한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/cheating%20sheet/pandas%20cheating%20sheet.ipynb)에서 확인 가능합니다.
