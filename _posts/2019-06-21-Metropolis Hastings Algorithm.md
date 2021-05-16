---
layout: post
title:  "Metropolis-Hastings Algorithm"
date: 2019-06-21
author: seolbluewings
comments : true
categories: Statistics
---
모수 $$\Theta=(\theta_{1},...,\theta_{p})$$ 의 Posterior distribution인 $$\pi(\theta\mid x)$$로 부터 $$\Theta$$에 대한 Sampling을 진행하고자 한다. Gibbs Sampler의 경우, 각 parameter $$\theta_{k}$$의 full conditional posterior를 활용하며, 이 경우에는 $$p(\theta_{k}\mid\theta_{-k})$$ 가 우리가 아는 closed form 형태의 분포 형태를 가져야 한다.

만약 $$\theta_{k}$$에 대해 full conditional posterior가 closed form으로 나오지 않는다면, 다음과 같이 Metropolis-Hastings Algorithm을 사용하여 $$\pi(\mathbf{\theta}\mid x)$$로부터 표본 추출이 가능하다.

Metropolis-Hastings Algorithm은 다음과 같은 절차를 통해 진행된다.

Metropolis-Hastings Algorithm의 t번째 step은 다음과 같다.

첫번째, $$\theta^{*}$$의 추출은 다음의 분포를 통해 이루어진다.

$$\theta^{*} \sim T(\theta^{*}\mid\theta^{t})$$

두번째, 새롭게 proposed되는 $$\theta^{*}$$의 채택 확률 $$\alpha$$는 다음과 같은 식으로 구할 수 있다.

$$
\begin{align}
	\alpha &= \frac{\pi(\theta^{*})/T(\theta^{*}|\theta^{(t)})}{\pi(\theta^{(t)})/T(\theta^{(t)}|\theta^{*})} \\
    p &= min(\alpha,1)
\end{align}
$$

세번째 $$\theta^{(t+1)}$$는 p의 확률로 $$\theta^{*}$$로 채택되며, 1-p의 확률로 기존의 $$\theta^{(t)}$$로 결정된다.

세번째 단계는 코드로 구현하는 단계에서 $$u \sim U(0,1)$$를 통해 u를 생성해내고 이렇게 생성된 u값과 $$\alpha$$의 크기를 비교하여 $$u \leq \alpha$$이면 $$\theta^{(t+1)}=\theta^{*}$$가 되고 $$u > \alpha$$이면, $$\theta^{(t+1)}=\theta^{(t)}$$로 정해진다.

Metropolis-Hastings Algorithm에서 생성된 표본 $$\theta^{(t+1)}$$은 $$\pi(\theta\mid x)$$로 수렴한다. 따라서 앞서 소개한 Gibbs Sampler와 마찬가지로 수렴시점 이후의 표본을 사용하며 연속된 표본은 서로 상관관계를 가지고 있다.

분포 $$T(\theta^{*} \mid \theta^{(t)})$$는 $$\theta^{*}$$를 추출하기 위해 임의로 선택되는 밀도함수이며, 이를 transition kernel이라 부른다.

$$T(\theta^{*}\mid \theta^{(t)})$$가 $$\mathcal{N}(\theta^{(t)},\delta^{2})$$의 분포라면, ($$\delta$$의 값은 주어졌다고 가정) 이를 random-walk Metropolis-Algorithm이라 부른다.

이 경우에는 $$\delta$$의 적절한 크기를 결정하는 것이 중요하다. 만약 $$\delta$$가 작다면, 현재의 표본 근처에서만 이동하게 되고 Posterior의 전 영역을 이동하기에 시간이 많이 필요하다. 만약 $$\delta$$가 크다면, 확률이 낮은 영역에서 새로운 표본을 추출할 가능성이 높아져 새로운 표본을 채택할 확률이 낮아지게 된다.

Gibbs Sampler처럼 원소들을 분할하여 Metropolis-Hastings Algorithm을 진행할 수 있다.

$$\mathbf{\theta}=(\theta_{1},\theta_{2})$$로 나누어질 때, (t+1)번째 step은 다음과 같다.

$$\theta_{1}^{(t+1)}$$ 추출 과정은 다음의 순서를 따를 것이다. 먼저
$$
\theta_{1}^{*} \sim T(\theta_{1}\mid x,\theta^{(t)}_{1},\theta_{2}^{(t)}) $$ 에서 $$\theta_{1}^{*} $$을 추출해내고

$$
\alpha = \frac{\pi(\theta_{1}^{*},\theta_{2}^{(t)}|x)/T(\theta_{1}^{*}|\theta_{1}^{(t)},\theta_{2}^{(t)})}{\pi(\theta_{1}^{(t)},\theta_{2}^{(t)}|x)/T(\theta_{1}^{(t)}|\theta_{1}^{*},\theta_{2}^{(t)})} $$ 다음의 계산을 통해 $$\alpha$$ 값을 구한다. 이후 $$ p=min(\alpha,1) $$ 의 확률로 $$\theta^{(t+1)}_{1} = \theta^{*}_{1}$$ 처럼 $$\theta^{(t+1)}_{1}$$ 를 채택하고 $$1-p$$의 확률로 $$  \theta^{(t+1)}_{1} = \theta^{(t)}_{1} $$ 값을 설정한다.

마찬가지로 $$\theta_{2}^{(t+1)}$$ 추출 과정을 진행할 수 있다.

즉, Gibbs Sampler는 Metropolis-Hastings Algorithm의 특수한 경우이며, 이 때 transition kernel이 각 원소(원소 벡터)의 full-condtional posterior이다.

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

#### 예제

parameter $$\theta$$의 분포가 다음과 같이 주어졌다고 하자.

$$
p(\theta) \propto \frac{1}{\sqrt{8\theta^{2}+1}}\text{exp}\left(-\frac{1}{2}\left(\theta^{2}-8\theta-\frac{16}{8\theta^{2}+1}\right)\right)
$$

이 때, 가우시안분포의 mixture 형태의 분포인 transition kernel를 이용하여 M-H 알고리즘을 통해 샘플링을 진행해보도록 하자.

$$
T(\theta^{*}\mid\theta) = 0.6\mathcal{N}(\theta-1.5,1)+0.4\mathcal{N}(\theta+1.5,1)
$$

이에 대한 코드는 각각 코드는 다음의 링크 1. [R코드](https://github.com/seolbluewings/R_code/blob/master/M-H%20Algorithm.ipynb) 2. [Python코드](https://github.com/seolbluewings/pythoncode/blob/master/7.MH%20Algorithm.ipynb) 에서 확인할 수 있습니다.


#### 참조 문헌

1. [BDA](http://www.stat.columbia.edu/~gelman/book/)




