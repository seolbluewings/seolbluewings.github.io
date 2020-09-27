---
layout: post
title:  "라플라스 근사(Laplace Approximation)"
date: 2019-06-24
author: seolbluewings
categories: Statistics
---

베이지안 방법에서는 parameter $$\theta$$의 posterior 분포를 직접 구하기 어려운 상황에서 posterior 분포를 근사적으로 계산할 필요가 있다. 라플라스 근사법(Laplace Approximation)은 true posterior 분포의 mode 위치에서의 가우시안 분포로 근사하는 기법을 의미한다.

임의의 분포함수(density function) $$f(\theta)$$는 Taylor Series 를 이용해서 근사가 가능하다. $$\theta_{0}$$는 parameter $$\theta$$의 posterior distribution의 mode값을 의미한다.

$$
\begin{align}
f(\theta) &\simeq f(\theta_{0}) + f^{'}(theta_{0})(\theta-\theta_{0}) + \frac{1}{2!}f^{''}(\theta_{0})(\theta-\theta_{0})^{2} + \frac{1}{3!}f^{'''}(\theta_{0})(\theta-\theta_{0})^{3} + \cdct\cdot\cdot \nonumber \\
f(\theta) &\simeq f(\theta_{0}) + \frac{1}{2!}f^{''}(\theta_{0})(\theta-\theta_{0})^{2} \nonumber
\end{align}
$$

그리고 $$(\theta-\theta_{0})^{2}$$ 이란 term은 우리가 가우시안 분포에서 자주 보던 항이라 할 수 있다.

$$\theta$$의 posterior 분포 $$p(\theta\mid y)$$가 있다고 했을 때, log posterior 값 $$l(\theta) = \log{p(\theta\mid y)}$$ 값을 mode $$\theta_{0}$$에서 second order Taylor expansion을 수행하여 approximation 하는 것이 라플라스 근사 방법이라 할 수 있다.


$$
\begin{align}
\log{p(\theta\mid y)} &= l(\theta) \simeq l(\theta_{0}) + l^{'}(\theta_{0})(\theta-\theta_{0}) + \frac{1}{2}l^{''}(\theta_{0})(\theta-\theta_{0})^{2} \nonumber \\
p(\theta\mid y) &= \text{exp}\left(l(\theta_{0})+\frac{1}{2}l^{''}(\theta_{0})(\theta-\theta_{0})^{2}\right) = q(\theta) \nonumber \\
&\propto \text{exp}\left(\frac{-1}{2}(-l^{''}(\theta_{0}))(\theta-\theta_{0})^{2}\right)
\end{align}
$$

따라서 posterior distribution $$p(\theta\mid y)$$ 는 Gaussian 분포를 따른다고 볼 수 있으며 posterior를 가우시안 분포로 근사한 함수 $$q(\theta)$$를 다음과 같이 표현할 수 있을 것이다.

$$
\begin{align}
q(\theta) &\sim \mathcal{N}(\theta\mid\mu,\tau^{2}) \nonumber \\
\mu &= \theta_{0}
\tau^{2} &= (-l^{''}(\theta_{0}))^{-1}
\end{align}
$$





