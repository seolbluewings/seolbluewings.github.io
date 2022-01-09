---
layout: post
title:  "Dirichlet Process Mixture Model"
date: 2022-01-01
author: seolbluewings
categories: Statistics
---

Mixture Model에서 Dirichlet Distribution을 사용하는 일반적인 방식은 parameter의 차원이 k로 고정되어 있고 이 k차원의 parameter에 대한 prior로 활용하는 것이었다. GMM에서 cluster의 개수는 정해져 있었고 k번째 cluster로 할당될 latent variable $$z_{k}$$를 정의하고 $$p(z_{k}=1) = \pi_{k}$$ 에서의 $$\pi_{k}$$에 대한 prior로 Dirichlet Distribution을 활용했다.

그러나 Dirichlet Process Mixture Model(DPMM)은 k를 특정 차원으로 한정짓지 않고 $$k \to \infty$$ 인 경우에 대해 논의하며 $$k \to \infty$$ 처리로 인해 DP를 활용하게 된다.

![DPMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/DP4.PNG?raw=true){:width="70%" height="30%"}{: .aligncenter}

DPMM에 대한 Graphical View는 위의 그림과 같다. 기존 $$H$$로 표현되었던 Base Distribution $$G_{0}$$ 와 $$\alpha$$가 DP로 인한 Multinomial Distribution $$G$$를 생성하며 데이터에 대한 분포는 $$G$$에서 결정된 $$\theta_{i}$$로 인해 결정된다. 무한대 차원의 GMM이라 하면 $$\theta_{i}$$는 $$i$$번째 데이터가 선택하게 될 k번째 Gaussian Distribution의 parameter($$\mu,\Sigma$$)가 될 것이다

이 Graphical View는 Chinese Restaurant Process(CRP)를 통해 보다 직관적으로 다가온다. CRP에서 표현했던 DP는 다음과 같다.

$$
\theta_{n}\vert\theta_{1},...,\theta_{n-1},\alpha,H \sim \text{DP}\left(\alpha+n-1, \frac{\alpha}{\alpha+n-1}H + \frac{1}{\alpha+n-1}\sum_{i=1}^{n-1}\delta_{\theta_{i}}\right)$$

![DPMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/DP3.png?raw=true){:width="70%" height="70%"}{: .aligncenter}

i번째 데이터(손님) $$x_{i}$$에 대해 $$\theta_{i}$$의 값을 결정하여 Cluster Assign(테이블 부여)를 진행한다고 하자.

Cluster 할당에 대한 parameter $$\theta_{i}$$ 는 기존의 데이터가 어떤 Cluster에 할당되어 있는지에 영향을 받는다. 그래서 각 Cluster(테이블)마다 존재하는 데이터(손님) 수로 확률이 주어지며, 특정 확률로 새로운 Cluster를 형성할 수도 있다. $$\theta_{i}$$ 로 결정되는 Cluster에는 데이터 $$x_{i}$$에 대한 parameter, $$x_{i}$$가 Gaussian 분포에서 생성되었다고 한다면 $$\mu_{k},\Sigma_{k}$$ 에 대한 정보가 존재하는 것으로 볼 수 있다.

DPMM을 통한 Sampling Based Inference를 위해선 다음과 같은 절차를 거친다. GMM의 경우를 가정해서 진행하도록 하겠다.

1. 모든 n개의 데이터 포인트에 대해 random하게 cluster를 먼저 assign 한다.
2. 다음의 Gibbs Sampler 과정을 수행하는데 결과가 stationary distribution 형태를 나타날 때까지 충분히 수행한다.
- 다음의 과정을 모든 n개의 데이터 포인트에 대해 수행한다.
	- sampling 대상이 될 데이터 포인트 $$x_{i}$$ 를 기존 Cluster Assign 에서 제거한다.
	- $$x_{i}$$가 제거된 상태에서의 Prior $$ \theta_{n}\vert\theta_{1},...,\theta_{n-1},\alpha,H \sim \text{DP} $$ 를 계산한다. 총 데이터 포인트가 4개인데 3번째 데이터에 대해 Sampling을 한다면 $$\theta_{3}\vert\theta_{1},\theta_{2},\theta_{4}$$ 에 대한 Conditional Distribution을 구한다.
	- 각 Cluster 마다의 parameter $$(\mu_{k},\Sigma_{k})$$를 활용하여 각 Cluster 마다의 x_{i}에 대한 likelihood $$\mathcal{N}(x_{i}\vert \mu_{k},\Sigma_{k}) $$를 계산한다. 만약 새로운 Cluster 할당 시, $$H$$ 활용한다.
	- Posterior $$\theta_{i}\vert x_{i}$$를 계산하고 Posterior 활용하여 $$x_{i}$$에 대한 Cluster를 재할당 수행한다.
	- 새롭게 할당된 데이터를 바탕으로 Cluster의 parameter 정보를 Update 한다


상기 내용에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Dirichlet%20Process%20Mixture%20Model.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [인공지능 및 기계학습심화](https://www.edwith.org/aiml-adv/joinLectures/14705)
2. [Density Estimation with Dirichlet Process Mixtures using PyMC3](https://austinrochford.com/posts/2016-02-25-density-estimation-dpm.html)