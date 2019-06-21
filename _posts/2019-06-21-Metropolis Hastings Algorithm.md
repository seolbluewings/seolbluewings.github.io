---
layout: post
title:  "Metropolis-Hastings Algorithm"
date: 2019-06-21
author: YoungHwan Seol
categories: Bayesian
---
모수 $$\mathbf{\theta}=(\theta_{1},...,\theta_{p})$$ 의 Posterior distribution인 $$\pi(\theta\|x)$$로 부터 $$\mathbf{\theta}$$의 사후 표본을 추출하려고 한다.

Gibbs Sampler의 경우는 각 원소 $$\theta_{i}$$의 full-conditional posterior를 활용하며 이 경우에는 $$p(\theta_{i}\|x,\theta_{1},\theta_{2},...,\theta_{p})$$로부터 직접적인 표본 생성이 가능해야 한다.

만약 $$\theta_{i}$$에 대해 full-conditional posterior가 closed form으로 나오지 않는다면, 다음과 같이 Metropolis-Hastings Algorithm을 이용한다. Metropolis-Hastings Algorithm을 사용하여 $$\pi(\mathbf{\theta}\|x)$$로부터 표본 추출이 가능하다.

Metropolis-Hastings Algorithm은 다음과 같은 절차를 통해 진행된다.

Metropolis-Hastings Algorithm의 t번째 step은 다음과 같다.

1. \theta^{*}의 추출은 다음의 분포를 통해 이루어진다. $$\theta^{*} \sim T(\theta^{*}\|\theta^{t})$$

2. 새롭게 proposed되는 $$\theta^{*}$$의 채택 확률 $$\alpha$$는 다음과 같은 식으로 구할 수 있다.

$$
\begin{align}
	\alpha &= \frac{\pi(\theta^{*})/T(\theta^{*}|\theta^{(t)})}{\pi(\theta^{(t)})/T(\theta^{(t)}|\theta^{*})} \\
    p &= min(\alpha,1)
\end{align}
$$

3. $$\theta^{(t+1)}$$는 p의 확률로 $$\theta^{*}$$로 채택되며, 1-p의 확률로 기존의 \theta^{(t)}로 결정된다.

3번째 단계는 코드로 구현하는 단계에서 $$u \sim U(0,1)$$를 통해 u를 생성해내고 이렇게 생성된 u값과 $$\alpha$$의 크기를 비교하여 $$u \leq \alpha$$이면 $$\theta^{(t+1)}=\theta^{*}$$가 되고 $$u > \alpha$$이면, $$\theta^{(t+1)}=\theta^{(t)}$$로 정해진다.

Metropolis-Hastings Algorithm에서 생성된 표본 $$\theta^{(t+1)}$$은 $$\pi(\theta\|x)$$로 수렴한다. 따라서 앞서 소개한 Gibbs Sampler와 마찬가지로 수렴시점 이후의 표본을 사용하며 연속된 표본은 서로 상관관계를 가지고 있다.




