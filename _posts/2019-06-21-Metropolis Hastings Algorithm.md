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

1. $$\theta^{*}$$의 추출은 다음의 분포를 통해 이루어진다. $$\theta^{*} \sim T(\theta^{*}\|\theta^{t})$$

2. 새롭게 proposed되는 $$\theta^{*}$$의 채택 확률 $$\alpha$$는 다음과 같은 식으로 구할 수 있다.

$$
\begin{align}
	\alpha &= \frac{\pi(\theta^{*})/T(\theta^{*}|\theta^{(t)})}{\pi(\theta^{(t)})/T(\theta^{(t)}|\theta^{*})} \\
    p &= min(\alpha,1)
\end{align}
$$

3. $$\theta^{(t+1)}$$는 p의 확률로 $$\theta^{*}$$로 채택되며, 1-p의 확률로 기존의 $$\theta^{(t)}$$로 결정된다.

3번째 단계는 코드로 구현하는 단계에서 $$u \sim U(0,1)$$를 통해 u를 생성해내고 이렇게 생성된 u값과 $$\alpha$$의 크기를 비교하여 $$u \leq \alpha$$이면 $$\theta^{(t+1)}=\theta^{*}$$가 되고 $$u > \alpha$$이면, $$\theta^{(t+1)}=\theta^{(t)}$$로 정해진다.

Metropolis-Hastings Algorithm에서 생성된 표본 $$\theta^{(t+1)}$$은 $$\pi(\theta\|x)$$로 수렴한다. 따라서 앞서 소개한 Gibbs Sampler와 마찬가지로 수렴시점 이후의 표본을 사용하며 연속된 표본은 서로 상관관계를 가지고 있다.

분포 $$T(\theta^{*}\|\theta^{(t)})$$는 $$\theta^{*}$$를 추출하기 위해 임의로 선택되는 밀도함수이며, 이를 transition kernel이라 부른다.

$$T(\theta^{*}\|\theta^{(t)})$$가 $$\mathcal{N}(\theta^{(t)},\delta^{2})$$의 분포라면, ($$\delta$$의 값은 주어졌다고 가정) 이를 random-walk Metropolis-Algorithm이라 부른다.

이 경우에는 $$\delta$$의 적절한 크기를 결정하는 것이 중요하다. 만약 $$\delta$$가 작다면, 현재의 표본 근처에서만 이동하게 되고 Posterior의 전 영역을 이동하기에 시간이 많이 필요하다. 만약 $$\delta$$가 크다면, 확률이 낮은 영역에서 새로운 표본을 추출할 가능성이 높아져 새로운 표본을 채택할 확률이 낮아지게 된다. $$\delta$$의 적절한 값은 $$\theta^{(t)}$$ 시점의 분산 추정치의 $$2.4/\sqrt{p}$$ 배가 좋은 것으로 권장된다. 여기서 p는 $$\mathbf{\theta}$$의 차원이다.

Gibbs Sampler처럼 원소들을 분할하여 Metropolis-Hastings Algorithm을 진행할 수 있다.

$$\mathbf{\theta}=(\theta_{1},\theta_{2})$$로 나누어질 때, (t+1)번째 step은 다음과 같다.

$$\theta_{1}^{(t+1)}$$ 추출

1. $$\theta_{1}^{*} \sim T(\theta_{1}|x,\theta^{(t)}_{1},\theta_{2}^{(t)})$$

2. $$ \alpha = \frac{\pi(\theta_{1}^{*},\theta_{2}^{(t)}|x)/T(\theta_{1}^{*}|\theta_{1}^{(t)},\theta_{2}^{(t)})}{\pi(\theta_{1}^{(t)},\theta_{2}^{(t)}|x)/T(\theta_{1}^{(t)}|\theta_{1}^{*},\theta_{2}^{(t)})} $$

3. set $$\theta^{(t+1)}_{1}=\theta^{*}_{1}$$ with $$p=min(\alpha,1)$$, $$\theta^{(t+1)}_{1}=\theta^{(t)}_{1}$$ with $$1-p$$

$$\theta_{2}^{(t+1)}$$ 추출

1. $$\theta_{2}^{*} \sim T(\theta_{2}|x,\theta^{(t+1)}_{1},\theta_{2}^{(t)})$$

2. $$ \alpha = \frac{\pi(\theta_{2}^{*},\theta_{1}^{(t+1)}|x)/T(\theta_{2}^{*}|\theta_{1}^{(t+1)},\theta_{2}^{(t)})}{\pi(\theta_{2}^{(t)},\theta_{1}^{(t+1)}|x)/T(\theta_{2}^{(t)}|\theta_{2}^{*},\theta_{1}^{(t+1)})} $$

3. set $$\theta^{(t+1)}_{2}=\theta^{*}_{2}$$ with $$p=min(\alpha,1)$$, $$\theta^{(t+1)}_{2}=\theta^{(t)}_{2}$$ with $$1-p$$

Gibbs Sampler는 Metropolis-Hastings Algorithm의 특수한 경우이며, 이 때 transition kernel이 각 원소(원소 벡터)의 full-condtional posterior이다.

$$
\begin{align}
	T(\theta_{1}|x,\theta_{1}^{(t)},\theta_{2}^{(t)}) &= \pi(\theta_{1}|x,\theta_{2}^{(t)}) \\
    T(\theta_{2}|x,\theta_{1}^{(t+1)},\theta_{2}^{(t)}) &= \pi(\theta_{2}|x,\theta_{1}^{(t+1)})
\end{align}
$$

이 경우 $$\alpha$$의 값은 다음과 같다.

$$
\begin{align}
	\alpha &= \frac{\pi(\theta_{1}^{*},\theta_{2}^{(t)}|x)/\pi(\theta_{1}^{*}|x,\theta_{2}^{(t)})}{\pi(\theta_{1}^{(t)},\theta_{2}^{(t)}|x)/\pi(\theta_{1}^{(t)}|x,\theta_{2}^{(t)})} \\
    &= \frac{\pi(\theta_{1}^{*}|\theta_{2}^{(t)},x)\pi(\theta_{2}^{(t)}|x)}{\pi(\theta_{1}^{(t)}|\theta_{2}^{(t)},x)\pi(\theta_{2}^{(t)}|x)} \times \frac{\pi(\theta_{1}^{(t)}|x,\theta_{2}^{(t)})}{\pi(\theta_{1}^{*}|x,\theta_{2}^{(t)})} \\
    &= 1
\end{align}
$$

이는 매번 $$\theta^{*}$$를 $$\theta^{(t+1)}$$로 받아들이는 Algorithm으로 Gibbs Sampler는 항상 Accept하는 Metropolis-Hasting Algorithm이라 할 수 있다. 





