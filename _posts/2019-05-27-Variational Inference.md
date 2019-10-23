---
layout: post
title:  "Variational Inference"
date: 2019-05-26
author: YoungHwan Seol
categories: Bayesian
---

Variational Inference는 속도가 느린 MCMC 방법을 대체할 수 있는 방법으로 MCMC보다는 정확성이 떨어지지만, 속도 측면에서는 MCMC보다 우수한 성질을 가지고 있다. Variational Inference는 우리가 추정하고자하는 target distribution을 approximate하는 density를 찾는 것으로 target posterior distribution과 가장 가까운 형태의 closed-form approximation을 찾는 것이라 할 수 있다.

따라서 우리는 2개의 probability distribution의 차이를 비교할 수 있어야 한다. 여기서 비교대상이 되는 2개의 probability distribution은 target posterior distribution인 $$p(\theta \mid y)$$와 이 target posterior distribution에 근사하다고 생각되는 $$q(\theta)$$이며, 이 두가지 분포의 divergence를 계산한다.

##### Kullback-Leibler Divergence

두 분포의 divergence를 계산하기 위해서는 KLD(Kullback-Leibler Divergence)를 알아보아야 한다. KLD는 식의 구조상 symmetric하지 않으며 아래를 통해서 확인할 수 있듯이, 항상 0보다 큰 값을 가진다.

$$
\begin{align}
	KL(q(\theta)\mid\mid p(\theta\mid y)) &= \int q(\theta)\log{\frac{q(\theta)}{p(\theta \mid y)}}d\theta \\
    -KL(q(\theta)\mid\mid p(\theta\mid y)) &= \int q(\theta)\log{\frac{p(\theta \mid y)}{q(\theta)}}d\theta \\
    &= \mathbb{E}_{q}\left[\log{\frac{p(\theta \mid y)}{q(\theta)}}\right] \leq \log{\left[\mathbb{E}_{q}\left[\frac{p(\theta \mid y)}{q(\theta)}\right]\right]} \\
    &\quad \text{by Jensen's Inequality} \\
    \log{\left[\mathbb{E}_{q}\left[\frac{p(\theta \mid y)}{q(\theta)}\right]\right]} &= \log{\left[\int\frac{p(\theta \mid y)}{q(\theta)}q(\theta)d\theta \right]}=0 \\
    &\therefore -KL(q(\theta)\mid\mid p(\theta\mid y)) \leq 0
\end{align}
$$

따라서 KLD는 언제나 0보다 크다.

우리의 목표는 KLD를 최소화하는 분포 $$q(\theta)$$를 찾는 것이다.

$$q^{*} = argmin_{q \in Q} KL(q(\theta)\mid\mid p(\theta\mid y))$$

만약 $$\theta = (\theta_{1},\theta_{2})$$라면, $$q(\theta) = q(\theta_{1})q(\theta_{2})$$로 factorization될 수 있다.

##### ELBO(Evidence Lower Bound)

KLD를 활용하여 얻을 수 있는 사실은 또 하나가 있다.

$$
\begin{align}
	KL(q(\theta)\mid\mid p(\theta\mid y)) &= \int q(\theta)\log{\frac{q(\theta)}{p(\theta\mid y)}}d\theta \geq 0 \\
    &=\int q(\theta)\log{q(\theta)}d\theta - \int q(\theta)\log{p(\theta \mid y)}d\theta \geq 0 \\
    &=\int q(\theta)\log{q(\theta)}d\theta - \int q(\theta)\log{p(\theta,y)}d\theta + \int q(\theta)\log{p(y)}d\theta \geq 0 \\
    &=\int q(\theta)\log{q(\theta)}d\theta - \int q(\theta)\log{p(\theta,y)}d\theta + \log{p(y)} \geq 0 \\
    \log{p(y)} &\geq -\int q(\theta)\log{q(\theta)}d\theta + \int q(\theta)\log{p(\theta,y)}d\theta
\end{align}
$$

바로 위의 부등식에서 부등호 이후의 부분을 ELBO(Evidence Lower Bound)라고 한다.

다음의 부등식이 성립하며, 데이터가 주어진 경우 $$\log{p(y)}$$는 값이 고정되기 때문에 KLD를 minimize하는 것은 ELBO를 maximize 하는 것과 동등하다고 할 수 있다.

$$\log{p(y)} = \text{ELBO} + KL(q(\theta)\mid\mid p(\theta\mid y)) \geq \log{p(y)}$$

$$ q^{*} = argmax_{q \in Q}\text{ELBO}$$

##### Mean Field Variational Inference

우리가 approximate하는 posterior $$q(\theta)$$가 다음과 같이 factorization 된다고 하자.

$$q(\theta) = \prod_{j=1}^{J} q_{j}(\theta_{j})$$

Mean Field Variational Inference는 variational distribution family를 사용한다.

$$ Q= \left\{q : q(\theta) = \prod_{j=1}^{J}q_{j}(\theta_{j})\right\} $$

우리가 추정하고자 하는 true posterior distribution의 parameter들은 서로 independent하지 않는다. variational distribution의 경우는 서로 independent하다는 가정을 하기 때문에 variational distribution family는 true posterior distribution을 포함하지 못한다.

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

여기서 $$\theta_{k}$$에 대한 값을 추정하기 위해 $$\theta_{k}$$에 대한 conditional maximization을 coordinate ascent를 활용해 진행한다.




coordinate ascent algorithm은 각 $$q(z_{k})$$를 update하며, 그 결과 local maximum으로 수렴한다, $$q(z_{k})$$에 대한 coordinate ascent update는 오로지 $$q(z_{j}),k \neq j$$ 근사값에 의존한다.

Variational Inference의 coordinate ascent algorithm과 Gibbs Sampling은 비슷한 성질을 갖는다.

1. Gibbs Sampling은 conditional posterior로부터 sample하며
2. Variational Inference의 coordinate ascent algorithm은 다음과 같은 형태를 갖는다. $$q(z_{k}) \propto exp\{\mathbb{E}\left[\log{(conditional)}\right]\}$$

#### Exponential Family Conditionals

각각의 conditional이 exponential family라고 가정하자.

$$p(z_{j}|z_{-j},x) = h(z_{j})exp\{\eta(z_{-j},x)^{T}t(z_{j})-a(\eta(z_{-j},x))\}$$

이 조건에서 mean field variational inference는 직관적이다.

conditional에 log를 취하면

$$\log{p(z_{j}|z_{-j},x)}=\log{h(z_{j})}+\eta(z_{-j},x)^{T}t(z_{j})-a(\eta(z_{-j},x))$$

q(z_{-j})에 대하여 기대값을 취하면,

$$\mathbb{E}\left[\log{p(z_{j}|z_{-j},x)}\right]=\log{h(z_{j})}+\mathbb{E}\left[\eta(z_{-j},x)\right]^{T}t(z_{j})-\mathbb{E}\left[a(\eta(z_{-j},x))\right]$$

마지막 term은 $$q_{j}$$에 의존하지 않기 때문에

$$q^{*}(z_{j}) \propto h(z_{j})exp\{\mathbb{E}\left[\eta(z_{-j},x)\right]^{T}t(z_{j})\}$$

이고 이를 통해서 우리는 optimal $$q(z_{j})$$ 역시 conditional과 같은 exponential family임을 알 수 있다.

#### Variational EM Algorithm

실제 문제에서 우리는 posterior $$p(z\|x)$$에 근사한 $$q(z)$$의 parameter $$\nu$$ 뿐만 아니라 $$p(x\|z)$$의 parameter $$\theta$$를 구해야 한다. 이 때 우리는 EM 알고리즘을 사용할 수 있다.

$$
\begin{align}
	\log{p(y|\theta)} &\geq ELBO(\theta,\nu) \\
    &= \int q_{\nu}(z)\log{\frac{p(x,z|\theta)}{q_{\nu}(z)}}dz
\end{align}
$$

$$(\theta,\nu)$$에 대하여 ELBO를 iteratively하게 최대화하는 과정을 진행한다.

E-step : given $$\theta$$일 때, $$\nu$$에 대하여 ELBO를 maximize 한다. 이는 곧 $$\nu$$를 parameter로 갖는 $$q_{\nu}(z)$$와 $$p(z\|x,\theta)$$의 KLD를 줄이는 것과 같다.

$$q^{*}_{\nu}(z)=argmin_{\nu}D_{KL}(q_{\nu}(z)|p(z|x,\theta))$$

M-step : 먼저 E-step에서 구한 $$\nu$$가 given일 때, $$\theta$$에 대하여 ELBO를 maximize 한다. 이는 complete-data log-likelihood의 조건부 기대값을 maximize하는 것과 동일하다.

$$\theta^{*}=argmax_{\theta}\mathbb{E}_{q}\left[\log{p(z,x|\theta)}\right]$$

#### Coordinate Ascent Variational Inference

우리의 목적은 posterior에 근사하는 분포를 알아내는 것이며 이는 최적화(optimization)의 문제로 바라볼 수 있다. 그리고 최적화 문제는 종종 coordinate ascent algorithm을 통해 해결할 수 있다.

우리는 다음의 함수($$\mathcal{L}$$)를 최대화하는 $$q_{\nu}(z)$$를 찾아야 한다.

$$\mathcal{L} = \int q_{\nu}(z)\log{p(z,x)}dz-\int q_{\nu}(z)\log{q_{\nu}(z)}dz$$

mean-field variational inference를 적용하면, 우리의 target function은 다음과 같이 표기할 수 있다.

$$q(\mathbf{Z})=\prod_{j=1}^{m}q_{j}(z_{j})$$

$$
\begin{align}
	\mathcal{L}&=\left[\int\prod_{j=1}^{m}q_{j}(z_{j})\left[\log{p(\mathbf{Z}|\mathbf{X})} +\log{p(\mathbf{X})}\right]d\mathbf{Z} -\int\prod_{j=1}^{m}q_{j}(z_{j})\log{\prod_{j=1}^{m}q_{j}(z_{j})}d\mathbf{Z}\right] \\
   	&= \int q_{k}(z_{k})\prod_{j \neq k} q_{j}(z_{j})\log{p(Z_{k}|Z_{-k},\mathbf{X})}d\mathbf{Z} \\
   	&+\int q_{k}(z_{k})\prod_{j \neq k} q_{j}(z_{j})\log{p(Z_{-k}|\mathbf{X})}d\mathbf{Z} \\
   	&+ \int q_{k}(z_{k})\prod_{j \neq k} q_{j}(z_{j})\log{p(\mathbf{X})}d\mathbf{Z} \\
  	&-\int q_{k}(z_{k})\prod_{j \neq k} q_{j}(z_{j})\sum_{j=1}^{m}\log{q_{j}(z_{j})}d\mathbf{Z}
\end{align}
$$

여기서 $$\prod_{j \neq k} q_{j}(z_{j})$$는 $$q_{-k}(z_{k})$$를 의미한다.

$$
\begin{align}
	\log{q_{k}(z_{k})} &= \mathbb{E}_{-k}\left[\log{p(Z_{k}|Z_{-k},x)}\right] + C \\
    &\Leftrightarrow q_{k}(z_{k}) \propto exp\left[\mathbb{E}_{-k}(\log{p(Z_{k}|Z_{-k},x)})\right] \\
    q_{k}^{*}(z_{k}) &\propto exp(\mathbb{E}_{-k}\left[\log{p(Z_{k}|Z_{-k},x)}\right]) \times exp(\log{p(Z_{-k},x)}) \\
    q_{k}^{*}(z_{k}) &\propto exp(\mathbb{E}_{-k}\left[log{p(Z_{k},Z_{-k},x)}\right])
\end{align}
$$

우리는 모든 variational distributions에 대하여 위와 같은 iteration을 진행한다.

아래와 같은 예시를 통해 Coordinate Ascent Variational Inference에 대해 논의를 진행해보자.

관측치가 다음과 같은 계층적 모델의 형태로 주어졌다고 가정하자.

$$
\begin{align}
	\theta_{k} &\sim \mathcal{N}(0,\sigma^{2}), k=1,...,K \\
    z_{i} &\sim Categorical(1/K,...,1/K), i=1,...,n \\
    y_{i}|(z_{i},\theta_{1:K}) &\sim \mathcal{N}(\theta_{z_{i}},1), i=1,...,n
\end{align}
$$

우리는 $$\sigma^{2}$$에 대해서 알고 있으며, $$\theta_{1:K}=(\theta_{1},...,\theta_{K}),z_{1:n} = (z_{1},...,z_{n}),y_{1:n}=(y_{1},...,y_{n})$$ 이라 하자.

우리가 찾길 희망하는 target posterior distribution은 $$p(z_{1:n},\theta_{1:K}\|y_{1:n})$$ 이며 이는 다음과 같은 관계를 가진다.

$$p(z_{1:n},\theta_{1:K}|y_{1:n}) \propto p(z_{1:n},\theta_{1:K},y_{1:n})$$

그리고 이 joint distribution은 다음과 같이 계산될 수 있다.

$$
\begin{align}
	p(z_{1:n},\theta_{1:K},y_{1:n}) &= p(y_{1:n}|z_{1:n},\theta_{1:K})p(z_{1:n})p(\theta_{1:K}) \\
    &= \prod_{i=1}^{n} p(y_{i}|z_{i},\theta_{1:K})p(z_{i})\prod_{k=1}^{K}p(\theta_{k})
\end{align}
$$

우리는 target posterior distribution $$p(z_{1:n},\theta_{1:K} \| y_{1:n})$$을 다음의 variational distribution $$q(z_{1:n},\theta_{1:K})$$를 통해 근사할 수 있으며, mean-field approximation을 활용하면 다음과 같이

$$q(z_{1:n},\theta_{1:K}) = \prod_{i=1}^{n} q_{1}(z_{i}|\pi_{i})\prod_{k=1}^{K} q_{2}(\theta_{k}|\eta_{k},\tau^{2}_{k})$$

로 표현될 수 있다. 이제 우리는 $$q(\mathbf{Z},\mathbf{\theta})$$를 알아내기 위하여, $$\pi,\eta,\tau$$를 추정해야 한다. 따라서 variational parameters는 $$\mathbf{\lambda}=(\pi_{1:n},\eta_{1:K},\tau^{2}_{1:K})$$ 이다.

ELBO($$\lambda$$}를 maximizing하는 것은 KLD를 minimizing 하는 것과 동일하므로 다음과 같은 과정을 진행한다.

$$ ELBO(\lambda) = \mathbb{E}_{q}[\log{p(z_{1:n},\theta_{1:K},\y_{1:n})}|\lambda]-\mathbb{E}_{q}[\log{q(z_{1:n},\theta_{1:K})|\lambda}] $$

$$
\begin{align}
	p(z_{1:n},\theta_{1:K}|y_{1:n}) &\propto p(z_{1:n},\theta_{1:K},y_{1:n}) \\
    &\propto p(y_{1:n}|z_{1:n},\theta_{1:K})p(z_{1:n})p(\theta_{1:K}) \\
    &\propto \prod_{i=1}^{n}p(y_{i}|z_{i},\theta_{1:K})p(z_{i})\prod_{k=1}^{K}p(\theta_{k})
\end{align}
$$

$$
\begin{align}
	ELBO(\lambda) &= \mathbb{E}_{q}[\log{p(z_{1:n},\theta_{1:K},y_{1:n})}|\lambda]-\mathbb{E}_{q}[\log{q(z_{1:n},\theta_{1:K})|\lambda}] \\
    &= \sum_{i=1}^{n}\mathbb{E}_{q}[\log{p(y_{i}|z_{i},\theta_{1:K})}|\lambda] +\sum_{i=1}^{n}\mathbb{E}_{q}[\log{p(z_{i})}|\phi_{i}]+\sum_{k=1}^{K}\mathbb{E}_{q}[\log{p(\theta_{k})}|\eta_{k},\tau^{2}_{k}] \\
    &-\sum_{i=1}^{n}\mathbb{E}_{q1}[\log{q_{1}(z_{i})}|\phi_{i}] - \sum_{k=1}^{K}\mathbb{E}_{q2}[\log{q_{2}(\theta_{k})}|\eta_{k},\tau^{2}_{k}]
\end{align}
$$

given $$(\eta_{1:K},\tau^{2}_{1:K})$$일 때, 다음과 같은 과정을 통해 $$z_{1:n}$$ 과 $$\theta_{1:K}$$의 분포에 대한 추론을 할 수 있다.

$$
\begin{align}
	q_{1}^{*}(z_{i}) &\propto exp[\mathbb{E}_{-z_{i}}\log{p(z_{1:n},\theta_{1:K},y_{1:n})}] \\
    &\propto exp(\mathbb{E}_{-z_{i}}[\log{y_{1:n}|z_{1:n},\theta_{1:K}}]+\mathbb{E}_{-z_{i}}[\log{p(z_{1:n)})}]) \\
    &\propto exp(\mathbb{E}_{-z_{i}}[\log{p(y_{i}|z_{i},\theta_{1:K})}]+\mathbb{E}_{-z_{i}}[\log{p(z_{i})}]) \\
    &\propto exp(\mathbb{E}_{-z_{i}}[\sum_{k=1}^{K}I(Z_{i}=k)\log{p(y_{i}|\theta_{k})}]-\log{K}) \\
    &\propto exp(\int\int\cdot\int\sum_{Z_{-k}}[\sum_{k=1}^{K}I(z_{i}=k)\log{p(y_{i}|\theta_{k})}]q_{1}(z_{-k})q_{2}(\theta_{1:K})d\theta -\log{K}) \\
    &\propto exp(\sum_{k=1}^{K}I(z_{i}=k)\int(-\frac{1}{2}(x_{i}-\theta_{k})^{2})q_{2}(\theta_{k}|\eta_{k},\tau^{2}_{k})d\theta_{k}) \\
    q_{1}^{*}(z_{i}=k) &\propto exp(\sum_{k=1}^{K}I(z_{i}=k)(y_{i}\eta_{k}-\frac{\eta_{k}^{2}+\tau_{k}^{2}}{2}))
\end{align}
$$

따라서 다음과 같이 구할 수 있다.

$$
\begin{align}
	q_{1}^{*}(z_{i}=1) &\propto A_{1} \\
    q_{1}^{*}(z_{i}=2) &\propto A_{2} \\
    ...\\
    q_{1}^{*}(z_{i}=K) &\propto A_{K} \\
    \therefore q_{1}^{*}(z_{i}=k) &=\frac{A_{k}}{A_{1}+A_{2}+...+A_{K}} 
\end{align}
$$

$$
\begin{align}
	q_{2}^{*}(\theta_{k}) &\propto exp(\mathbb{E}_{-\theta_{k}}[\log{p(y_{1:n},\theta_{1:K},z_{1:n})}]) \\
    &\propto exp(\mathbb{E}_{-\theta_{k}}[\log\{\prod_{i=1}^{n}\prod_{k=1}^{K}p(y_{1:n}|\theta_{k})^{I(z_{i}=k)}\prod_{k=1}^{K}p(\theta_{k}\}]) \\
    q_{2}^{*}(\theta_{k}) &\propto exp(\mathbb{E}_{-\theta_{k}}[\sum_{i=1}^{n}\sum_{k=1}^{K}I(z_{i}=k)\log{p(y_{i}|\theta_{k})}+\sum_{k=1}^{K}\log{p(\theta_{k})}]) \\
    &\propto (\sum_{i=1}^{n}\phi_{ik}\log{p(y_{i}|\theta_{k})}+\log{p(\theta_{k})}) \\
    &\propto exp\left(\sum_{i=1}^{n}\phi_{ik}\left(-\frac{(y_{i}-\theta_{k})^{2}}{2}\right)+\left(-\frac{\theta_{k}^{2}}{2\sigma^{2}}\right)\right) \\
	&exp \left(-\frac{1}{2}\left(\sum_{i=1}^{n}\phi_{ik}+\frac{1}{\sigma^{2}}\right)\left(\theta_{k}-\frac{\sum_{i=1}^{n}\phi_{ik}y_{i}}{\sum_{i=1}^{n}\phi_{ik}+\frac{1}{\sigma^{2}}}\right)\right)
\end{align}
$$

따라서 k번째 mixture component의 평균과 분산은 다음과 같다.

$$
\begin{align}
	\eta_{k} &= \frac{\sum_{i=1}^{n}\phi_{ik}y_{i}}{\sum_{i=1}^{n}\phi_{ik}+\frac{1}{\sigma^{2}}} \\
	\tau^{2}_{k} &= \frac{1}{\sum_{i=1}^{n}\phi_{ik}+\frac{1}{\sigma^{2}}}
\end{align}
$$