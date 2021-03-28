---
layout: post
title:  "Variational Inference"
date: 2019-05-26
author: seolbluewings
categories: Statistics
---

변분 추론(이하 Variational Inference)는 속도가 느린 MCMC 방법을 대체할 수 있는 방법으로 MCMC보다는 정확성이 떨어지지만, 속도 측면에서는 MCMC보다 우수한 성질을 가지고 있다.

Variational Inference는 우리가 추정하고자하는 복잡한 형태의 Target Distribution을 근사하는 분포를 찾는 과정에서 우리가 이미 알고 있는 간단한 형태의 분포를 활용하여 Target Distribution을 근사한다. 주로 Exponential Family에 속하는 분포들을 활용하게 되며 이를 Variational Inference에서 사용할 때, Variational Family라고 부르기도 한다.

![VI](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/VI.png?raw=true){:width="70%" height="70%"}{: .aligncenter}

위의 그림처럼 Target Distribution과 가장 가까운 형태의 Closed-Form 분포 Q를 찾는 것이 Variational Inference이다. 따라서 우리는 두 Distribution의 차이를 비교할 수 있어야 한다. 여기서 비교대상이 되는 2개의 probability distribution은 target posterior distribution인 $$p(\theta \mid y)$$와 이 target posterior distribution에 근사하다고 생각되는 $$q(\theta)$$이다.

#### Kullback-Leibler Divergence

두 분포의 차이를 계산하기 위해서 KLD(Kullback-Leibler Divergence)를 알아야 한다. KLD는 두 분포의 차이를 측정하는데 사용되는 값이다. KLD는 식의 구조상 symmetric하지 않으며 아래를 통해서 확인할 수 있듯이, 항상 0보다 큰 값을 가진다. $$p(\theta\vert y)$$는 Target Distribtuion이고 $$q(\theta)$$는 Variational Family에 속하는 간단한 형태의 분포를 의미한다.

$$
\begin{align}
KL(q(\theta)\vert p(\theta\mid y)) &= \int q(\theta)\log{\frac{q(\theta)}{p(\theta \mid y)}}d\theta \\
-KL(q(\theta)\vert p(\theta\mid y)) &= \int q(\theta)\log{\frac{p(\theta \mid y)}{q(\theta)}}d\theta \\
&= \mathbb{E}_{q}\left[\log{\frac{p(\theta \mid y)}{q(\theta)}}\right] \leq \log{\left[\mathbb{E}_{q}\left[\frac{p(\theta \mid y)}{q(\theta)}\right]\right]} &\quad \text{by Jensen's Inequality} \\
\log{\left[\mathbb{E}_{q}\left[\frac{p(\theta \mid y)}{q(\theta)}\right]\right]} &= \log{\left[\int\frac{p(\theta \mid y)}{q(\theta)}q(\theta)d\theta \right]}=0
\end{align}
$$

따라서 KLD는 언제나 0보다 크다.

$$\text{KL}(q(\theta)\vert p(\theta\mid y)) \geq 0 $$

Variational Inference를 사용하는 우리의 목적은 실제로 계산하기 어려운 Target Distribution을 간단한 형태의 분포로 근사하기 위함이다. KLD값이 항상 0보다 크거나 같다는 것은 KLD값을 최소화시킬 $$q(\theta)$$를 찾아야한다는걸 의미한다.

$$q^{*}(\theta) = \text{argmin}_{q \in Q} KL(q(\theta)\vert p(\theta\mid y))$$


#### ELBO(Evidence Lower Bound)

KLD를 활용하여 얻을 수 있는 사실은 또 하나가 있다.

$$
\begin{align}
	KL(q(\theta)\vert p(\theta\mid y)) &= \int q(\theta)\log{\frac{q(\theta)}{p(\theta\mid y)}}d\theta \geq 0 \\
    &=\int q(\theta)\log{q(\theta)}d\theta - \int q(\theta)\log{p(\theta \mid y)}d\theta \geq 0 \\
    &=\int q(\theta)\log{q(\theta)}d\theta - \int q(\theta)\log{p(\theta,y)}d\theta + \int q(\theta)\log{p(y)}d\theta \geq 0 \\
    &=\int q(\theta)\log{q(\theta)}d\theta - \int q(\theta)\log{p(\theta,y)}d\theta + \log{p(y)} \geq 0 \\
    \log{p(y)} &\geq -\int q(\theta)\log{q(\theta)}d\theta + \int q(\theta)\log{p(\theta,y)}d\theta
\end{align}
$$

바로 위의 부등식에서 부등호 이후의 부분을 ELBO(Evidence Lower Bound)라고 한다.

다음의 부등식이 성립하며, 데이터가 주어진 경우 $$\log{p(y)}$$는 값이 고정되기 때문에 KLD를 minimize하는 것은 ELBO를 maximize 하는 것과 동등하다고 할 수 있다.

$$\log{p(y)} = \text{ELBO} + KL(q(\theta)\vert p(\theta\mid y)) \geq \log{p(y)}$$

$$ q^{*}(\theta) = \text{argmax}_{q \in Q}\text{ELBO}$$

#### Mean Field Variational Inference

우리가 근사추정 하고자하는 $$q(\theta)$$가 다음과 같이 서로 independent하게 factorization 된다고 하자.

$$q(\theta) = \prod_{j=1}^{J} q_{j}(\theta_{j})$$

우리가 추정하고자 하는 Target distribution의 parameter들은 서로 independent하지 않는다. 하지만 Variational distribution $$q(\theta)$$는 서로 independent하다는 가정을 한다.

joint distribution $$p(\theta,y)$$ 는 다음과 같이 factorization될 수 있다.

$$
\begin{align}
p(\theta,y) &= p(\theta_{k},\theta_{-k},y) \\
&= p(\theta_{k},\theta_{-k} \mid y)p(y) \\
&= p(\theta_{k}\mid\theta_{-k},y)p(\theta_{-k}\mid y)p(y)
\end{align}
$$

이를 ELBO에 적용하면 ELBO를 다음과 같이 표현할 수 있다.

$$
\begin{align}
	\text{ELBO} &= \mathbb{E}_{q}\left[\log{p(\theta,y)}\right] - \mathbb{E}_{q}\left[\log{q(\theta)}\right] \\
    &= \mathbb{E}_{q}[\log{p(\theta_{k}\mid\theta_{-k},y)}]+\mathbb{E}_{q}[\log{p(\theta_{-k}\mid y)}]+\mathbb{E}_{q}[\log{p(y)}]-\sum_{j=1}^{J}\mathbb{E}_{q}[\log{q_{j}(\theta_{j})}]
\end{align}
$$

여기서 $$\theta_{k}$$에 대한 값을 추정하기 위해 ELBO를 $$\theta_{k}$$에 대해 conditional maximization을 Coordinate Ascent를 활용해 진행한다.

$$
\begin{align}
\text{ELBO}_{k} &= \mathbb{E}_{q}[\log{p(\theta_{k}\mid\theta_{-k},y)}]-\mathbb{E}_{q}[\log{q_{k}(\theta_{k})}] \\
&= \int\int \log{p(\theta_{k}\mid\theta_{-k},y)}\prod_{j=1}^{J}q_{j}(\theta_{j})d\theta - \int\log{q_{k}(\theta_{k})}\prod_{j=1}^{J}q_{j}(\theta_{j})d\theta \\
&= \int q_{k}(\theta_{k}) \left[\int\log{p(\theta_{k}\mid\theta_{-k},y)}\prod_{j\neq k}q_{j}(\theta_{j})d\theta_{-k} \right]d\theta_{k} - \int\log{q_{k}(\theta_{k})}q_{k}(\theta_{k})\left[\prod_{j \neq k}q_{j}(\theta_{j})d\theta_{-k} \right]d\theta_{k}
\end{align}
$$

$$\int\log{p(\theta_{k}\mid\theta_{-k},y)\prod_{j \neq k}q_{j}(\theta_{j})d\theta_{-k}} = \mathbb{E}_{-k}[\log{p(\theta_{k}\mid\theta_{-k},y)}]$$ 이기 때문에 이를 적용하면 위의 식을 다음과 같이 표현할 수 있다.

$$
\begin{align}
\text{ELBO}_{k} &= \int q_{k}(\theta_{k})\mathbb{E}_{-k}[\log{p(\theta_{k}\mid\theta_{-k},y)}]d\theta_{k}-\int q_{k}(\theta_{k})\log{q_{k}(\theta_{k})}d\theta_{k} \\
&= \int q_{k}(\theta_{k}) \log{\frac{\text{exp}[\mathbb{E}_{-k}\log{p(\theta_{k}\mid\theta_{-k},y)}]}{q_{k}(\theta_{k})}}d\theta_{k} \leq 0
\end{align}
$$

ELBO가 Maximization 되기 위해서는 이 값이 0에 가까워지는 것이 좋다. 그러기 위해서는 $$ g_{k}(\theta_{k}) \backsimeq \text{exp}(\mathbb{E}_{-k}[\log{p(\theta_{k}\mid\theta_{-k},y)}])$$이 성립되는 것이 최선일 것이다. 따라서 다음의 수식을 이용하여 $$q(\theta_{k})$$의 parameter를 계속해서 Optimization 한다.

$$
\begin{align}
\quad q^{*}_{k}(\theta_{k}) &\propto \text{exp}(\mathbb{E}_{-k}[\log{p(\theta_{k}\mid\theta_{-k},y)}]) \nonumber \\
q^{*}_{k}(\theta_{k}) &\propto \text{exp}(\mathbb{E}_{-k}[\log{p(\theta_{k},\theta_{-k},y)}])
\end{align}
$$

각 스텝이 parameter의 conditional 분포의 비례하는 형태를 가지고 이것이 또한 full joint distribution에 비례하는 것을 통해 우리는 Variational Inference가 Gibbs Sampler와 꽤 유사한 점이 있다는 것을 알 수 있다.

실전에서 Variational Inference를 활용할 때는 어떠한 Variational Family 분포를 선택하는지가 중요하다. 변수의 분포를 적절하지 못하게 가정한다면, Variational Inference의 효율은 낮다. 이후의 포스팅에서 Variational Inference를 활용하는 것을 업로드하고자 한다.

#### 출처


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>
2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)

