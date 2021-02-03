---
layout: post
title:  "Gaussian Mixture Model"
date: 2021-01-24
author: seolbluewings
categories: Bayesian
---

[작성중...]

우리는 통계문제를 해결하는 과정에서 데이터가 가우시안 분포(정규 분포)를 따를 것이라는 가정을 자주 한다. 그러나 데이터의 분포를 단 1개의 가우시안 분포만을 사용하여 표현하려는 것은 위험한 부분이 있다. 다음과 같은 경우를 고려해보자.

![GMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/GMM_1.png?raw=true){:width="70%" height="70%"}

그림을 통해 확인할 수 있는 것처럼 이 데이터들은 2개의 집단으로 나누어진 것으로 판단하는 것이 옳다. 하나의 가우시안 분포만으로는 데이터를 설명하기는 부적절하다. 대신 2개의 가우시안 분포를 선형 결합시킨다면, 우리는 이 데이터 집합을 더욱 잘 표현할 수 있다.

가우시안 분포들을 선형 결합하여 우리는 새로운 확률 분포를 생성해낼 수 있다.

![GMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/GMM_2.png?raw=true){:width="70%" height="70%"}

이 이미지는 파란색 곡선으로 표기된 3개의 가우시안 분포를 혼합하여 새로운 1개의 분포(빨간색)를 생성해낸 것을 의미한다. 새로운 분포를 생성하는 과정에서 우리는 혼합하는 가우시안 분포의 개수를 조정할 수도 있고 선형 결합의 대상이 되는 가우시안 분포의 parameter 값을 조정할 수도 있다. 또한 선형 결합의 계수를 조정함으로써 우리는 새롭게 생성되는 분포를 변주시킬 수 있다.

가우시안 혼합 분포(Gaussian Mixture Model)의 일반식은 다음과 같다.

$$ p(x) = \sum_{k=1}^{K} \pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$

이 수식은 총 K개의 가우시안 분포를 선형 결합하여 새로운 분포를 생성해내는 것을 표현한다. 또한 선형 결합의 대상이 되는 k번째 가우시안 분포의 평균과 분산이 각 $$\mu_{k},\Sigma_{k}$$임을 나타낸다.

이 수식에서 parameter $$\pi_{k}$$는 mixing coefficient로 불리며 $$\sum_{k=1}^{K}\pi_{k}=1$$ 조건을 만족한다. 또한 $$\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$이기 때문에 모든 k에 대하여 $$\pi_{k} \geq 0$$인 것은 $$p(x) \geq 0$$ 을 만족시키기 위한 충분조건이다. 그리고 모든 것을 종합해보면 우리는 $$ 0 \leq \pi_{k} \leq 1$$ 임을 알 수 있고 mixing coefficient $$\pi_{k}$$가 확률의 조건을 만족시키는걸 알 수 있다.

$$\pi_{k} = p(k)$$ 로 k번째 가우시안 분포에 속할 사전 확률(Prior)로 간주할 수 있다. $$\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$ 는 k가 주어진 상황에서 x의 확률, $$p(x \vert k)$$로 볼 수 있다.

$$p(x) = \sum_{k=1}^{K}p(k)p(x\vert k) = \sum_{k=1}^{K}\pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$

가 성립되어 Posterior distribution에 해당하는 $$p(k\vert x)$$ 값은 다음과 같이 계산될 것이다.

$$
p(k\vert x) = \frac{p(k)p(x\vert k)}{\sum_{k=1}^{K}p(x)p(x\vert k)} = \frac{\pi_{k}\mathcal{N}(x \vert \mu_{k},\Sigma_{k})}{\sum_{k=1}^{K}\pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k})}
$$

#### Gaussian Mixture Model의 활용

Gaussian Mixture Model을 활용하여 우리는 데이터 x가 주어졌을 때, 이 데이터가 어느 집단에 속하는지에 대한 확률 $$p(k \vert x)$$값을 구할 수 있다. 즉 우리는 Gaussian Mixture Model을 통하여 데이터를 Clustering 시킬 수 있다.

각 데이터에 대하여 $$p(k \vert x)$$를 구할 수 있고 이 값이 가장 큰 k를 찾아내어 해당 데이터를 k번째 군집에 속하는 것으로 판단하게 된다.

$$k_{i} = \text{argmax}_{k} \pi_{ik}$$

#### latent variable을 활용한 Gaussian Mixture Model

Gaussian Mixture Model은 이산형 잠재변수(latent variable)를 활용할 때 더욱 계산하기가 쉽다. 다음과 같은 조건을 만족하는 K차원의 이산확률변수 $$\mathbf{z}$$ 가 존재한다고 하자.

$$ z_{k} \in \{0,1\}, \quad \sum_{k}z_{k}= 1 $$

즉, $$\mathbf{z} = (z_{1},z_{2},...,z_{K})$$ 인데 특정 원소 $$z_{k}=1$$이고 나머지는 0인 경우를 의미한다. 벡터 $$\mathbf{z}$$는 어떤 원소가 0인지 아닌지에 따라 K개의 서로 다른 상태를 가질 수 있다. 따라서 이 latent variable을 활용하여 우리는 특정 데이터가 어느 집단에 속하게 될 것인가를 추론하는데 활용이 가능하다.

새로운 변수 $$\mathbf{z}$$의 추가로 우리의 관심이 되는 분포는 $$p(x,\mathbf{z})$$라고 표현이 가능하다. 또한 이 Joint Distribution은 $$p(x,\mathbf{z}) = p(\mathbf{z})p(x\vert \mathbf{z})$$ 로도 표현이 가능하다.

$$\mathbf{z}$$에 대한 분포는 k번째 집단에 속할 확률 $$\pi_{k}$$로 표현이 가능하다. 이 때, $$\pi_{k}$$는 확률로써 유효하기 위해 $$0 \leq \pi_{k} \leq 1$$, $$\sum_{k=1}^{K}\pi_{k}=1$$ 조건을 만족해야 한다.

$$ p(z_{k}=1) = \pi_{k} $$

이를 조금 더 일반적으로 표현하면, $$p(z)$$는 다음과 같이 표현이 가능하다.

$$ p(\mathbf{z}) = \prod_{k=1}^{K}\pi_{k}^{z_{k}} $$

한편, $$\mathbf{z}$$가 given일 때 x의 조건부 분포 $$p(x\vert\mathbf{z})$$ 는 다음과 같다.

$$ p(x\vert z_{k}=1) \sim \mathcal{N}(x\vert \mu_{k},\Sigma_{k}) $$

이를 일반식으로 표현하면, 다음과 같을 것이다.

$$ p(x\vert\mathbf{z}) = \prod_{k=1}^{K}\mathcal{N}(x\vert\mu_{k},\Sigma_{k})^{z_{k}} $$

따라서 우리가 원래 알고 있었던 $$ p(x) = \sum_{k=1}^{K} \pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k})$$ 식은 다음의 과정을 통해서 동일하게 얻을 수 있다.

$$ p(x) = \sum_{\mathbf{z}}p(\mathbf{z})p(x\vert\mathbf{z}) = \sum_{k=1}^{K} \pi_{k}\mathcal{N}(x\vert \mu_{k},\Sigma_{k}) $$

우리는 본래는 존재하지 않았던 latent variable $$\mathbf{z}$$를 도입하여 Gaussian Mixture Model을 표현할 수 있고 $$p(x)$$ 대신 Joint Distribution인 $$p(x,\mathbf{z})$$를 활용하여 Gaussian Mixture Model 문제를 해결할 수 있다. 그리고 이 방법이 더 편리하다.

Gaussian Mixture Model 문제를 해결하는 가장 대표적인 방법은 EM 알고리즘을 활용하는 것이다. [EM알고리즘 포스팅](https://seolbluewings.github.io/bayesian/2020/06/27/EM-Algorithm.html)의 예제에서 EM 알고리즘을 활용하여 Gaussian Mixture Model의 각 parameter를 추정하는 것을 예제로 해결한 바 있다. 따라서 이번 포스팅에서는 EM 알고리즘이 아닌 Gibbs Sampler를 통해 Gaussian Mixture Model을 해결하는 것을 살펴보고자 한다.

#### Gaussian Mixture Model with Gibbs Sampler

데이터 $$ \{(x_{1},y_{1}),...,(x_{n},y_{n})\} $$ 가 주어지고 우리는 이 데이터가 다른 집단에 속하는 데이터들이 서로 섞여있는 형태라고 생각한다. 따라서 다음과 같은 Gaussian Mixture Model을 가정할 수 있다.

$$ y_{i} \sim \sum_{k=1}^{K}\pi_{k}\mathcal{N}(\mu_{k},\sigma^{2}_{k}), \quad i=1,2,...,n $$

여기서 $$\mu_{k},\sigma_{k}^{2}$$는 각 가우시안 분포의 평균과 분산을 의미한다. 데이터 집단의 개수인 K는 사실 알지 못하지만, 우리는 K를 알고 있는 상황에서 Gaussian Mixture Model을 생성할 수 있다. 따라서 [K-Means Clustering Algorithm](https://seolbluewings.github.io/%EA%B5%B0%EC%A7%91%ED%99%94/2020/06/12/Cluster-Analysis.html)처럼 사전에 K값에 대한 결정을 내려야 한다. 다만 이 경우 $$\sum_{k=1}^{K}\pi_{k}=1$$ 조건만 만족하면 된다.

Gaussian Mixture Model을 해결하기 위해 사용하는 latent variable $$z_{i}$$는 확률 $$\pi = (\pi_{1},...,\pi_{k})$$ 에 대응하는 indicator variable로의 역할을 수행한다.

Gibbs Sampler 과정을 수행하기 위해 우리는 다음의 parameter $$\Theta = (\pi,\mu_{k},\sigma^{2}_{k})$$ 에 대한 Prior Distribution을 설정해야 한다.

$$
\begin{align}
\pi &= (\pi_{1},...,\pi_{k}) \sim \text{Dirichlet}(\frac{1}{K},...\frac{1}{K}) \nonumber \\
\mu_{k} &\sim \mathcal{N}(0,10^{2}) \quad k=1,2,...,K \nonumber \\
\sigma^{2}_{k} &\sim \text{IG}(100,1) \quad k=1,2,...,K \nonumber
\end{align}
$$

$$\mathbf{z} = (z_{1},...,z_{n}), \mathbf{y} = (y_{1},...,y_{n}) $$이라 할 때, 우리는 다음의 과정들을 거쳐 Gibbs Sampler 계산을 수행할 수 있다.

Step 1. Target Posterior Distribution 구하기

$$
\begin{align}
p(\mathbf{z},\pi,\mu,\sigma^{2}\vert \mathbf{y})
&\propto p(\mathbf{y}\vert \mathbf{z},\mu,\sigma^{2})p(\mathbf{z}\vert\pi)p(\pi)p(\mu)p(\sigma^{2}) \nonumber \\
&\propto \prod_{i=1}^{n} \prod_{k=1}^{K}\left\{ (\sigma_{k}^{2})^{-1/2} \text{exp} \left(-\frac{1}{2\sigma^{2}_{k}}(y_{i}-\mu_{k})^{2} \right) \right\}^{I(z_{i}=k)} \nonumber \\
&\times \prod_{i=1}^{n}\prod_{k=1}^{K}(\pi_{k})^{I(z_{i}=k)} \times \prod_{k=1}^{K} (\pi_{k})^{\frac{1}{K}-1} \times \prod_{k=1}^{K} \text{exp}\left(-\frac{1}{2\cdot 10^{2}}\mu_{k}^{2}\right) \nonumber \\
&\times \prod_{k=1}^{K} (\sigma^{2}_{k})^{-100-1} \text{exp}(-1/\sigma^{2}_{k})
\end{align}
$$

Step 2. $$\mathbf{z}$$ 에 대한 Sampling Step 설정

$$
\begin{align}
p(\mathbf{z}\vert\pi,\mu,\sigma^{2},\mathbf{y}) &\propto \prod_{i=1}^{n}\prod_{k=1}^{K}\left\{(\sigma_{k}^{2})^{-1/2}\text{exp}\left(-\frac{1}{2\sigma^{2}_{k}}(y_{i}-\mu_{k})^{2} \right) \right\}^{I(z_{i}=k)} \times \prod_{i=1}^{n}\prod_{k=1}^{K}(\pi_{k})^{I(z_{i}=k)} \nonumber \\
p(z_{i}\vert\pi,\mu,\sigma^{2},\mathbf{y}) &\propto \prod_{k=1}^{K}\left\{ \pi_{k}\text{exp}\left(-\frac{1}{2\sigma^{2}_{k}}(y_{i}-\mu_{k})^{2}\right) \right\}^{I(z_{i}=k)} \nonumber \\
p(z_{i}=k\vert\pi,\mu,\sigma^{2},\mathbf{y}) &= \frac{ \pi_{k}\text{exp}\left(-\frac{1}{2\sigma^{2}_{k}}(y_{i}-\mu_{k})^{2}\right) }{ \sum_{k=1}^{K}\left\{ \pi_{k}\text{exp}\left(-\frac{1}{2\sigma^{2}_{k}}(y_{i}-\mu_{k})^{2}\right) \right\} }
\end{align}
$$

Step 3. $$\pi$$에 대한 Sampling Step 설정

$$
\begin{align}
p(\pi\vert\mathbf{z},\mu,\sigma^{2},\mathbf{y}) &\propto \prod_{i=1}^{n}\prod_{k=1}^{K}(\pi_{k})^{I(z_{i}=k)}\prod_{k=1}^{K}(\pi_{k})^{\frac{1}{K}-1} \nonumber \\
&\propto \prod_{k=1}^{K}\left[ \pi_{k}^{\sum_{i=1}^{n}I(z_{i}=k)+\frac{1}{K}-1} \right] \nonumber \\
\pi_{k} &\sim \text{Dirichlet}\left(\sum_{i=1}^{n}I(z_{i}=k)+\frac{1}{K} \right)
\end{align}
$$



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
