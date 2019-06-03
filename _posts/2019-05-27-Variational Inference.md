---
layout: post
title:  "Variational Inference"
date: 2019-05-26
author: YoungHwan Seol
categories: Bayesian
---

Variational Inference란 복잡한 형태의 posterior 분포 $$p(z\|x)$$를 다루기 쉬운 형태의 $$q(z)$$로 근사하는 것을 말한다.

Variational Inference의 핵심적인 아이디어는 다음과 같다.

1. variational parameter $$\nu$$를 갖는 latent variables $$\{z_{1},z_{2},...,z_{m}\}$$의 분포$$q(z_{1},z_{2},...,z_{m}\|\nu)$$를 찾는다.
2. 이 분포를 찾아가는 과정에서 posterior distribution에 가장 가까이 근사하는 모수 $$\nu$$를 찾아낸다.
3. 이렇게 구한 분포 $$q$$를 posterior 대신 사용한다.

그렇다면, 우리는 posterior distribution에 근사한 $$q(z)$$를 만들기 위해 쿨백-라이블러 발산(Kullback-Leibler Divergence, 이하 KLD)에 대해 이해해야 한다. 

KLD은 두 확률분포의 차이를 계산할 수 있는 방식으로 $$p(z\|x)$$와 $$q(z)$$의 KLD 값을 계산한 이후, KLD이 줄어드는 방향으로 $$q(z)$$를 update하는 과정을 반복하면, posterior를 잘 근사하는 $$q^{*}(z)$$를 얻을 수 있다.

$$
\begin{align}
	D_{KL}(q(z)|p(z|x)) &= \mathbb{E}_{q}\left[\log{\frac{q(z)}{p(z|x)}}\right] = \int q(z)\log{\frac{q(z)}{p(z|x)}}dz \\
	&= \int q(z) \log{\frac{q(z)p(x)}{p(x|z)p(z)}} dz \\
	&= \int q(z) \log{\frac{q(z)}{p(z)}}dz + \int q(z)\log{p(x)}dz - \int q(z)\log{p(x|z)}dz \\
	&= D_{KL}(q(z)|p(z)) + \log{p(x)}-\mathbb{E}_{q}[\log{p(x|z)}]
\end{align}
$$

몬테 카를로 방법(Monte Carlo Method)을 KLD에 적용하면 다음과 같다.

$$
\begin{align}
	D_{KL}(q(z)|p(z|x)) &= D_{KL}(q(z)|p(z)) + \log{p(x)}-\mathbb{E}_{q}[\log{p(x|z)}] \\
    &= \mathbb{E}_{q}\left[\log{\frac{q(z)}{p(z)}}\right]+\log{p(x)}-\mathbb{E}_{q}\left[\log{p(x|z)}\right] \\
    &\simeq \frac{1}{K}\sum_{i=1}^{K}\left[\log{\frac{q(z_{i})}{p(z_{i})}}\right]+\log{p(x)}-\frac{1}{K}\sum_{i=0}^{K}\left[\log{p(x|z_{i})}\right] \\
    &= \frac{1}{K}\sum_{i=0}^{K}\left[\log{q(z_{i})}-\log{p(z_{i})}-\log{p(x|z_{i})}\right]+\log{p(x)}
\end{align}
$$

여기서 $$z_{i} \sim q(z)$$ 이기 때문에 몬테 카를로 방법이라 할 수 있다. 또한 이렇게 Monte Carlo Method를 사용하기 때문에 $$q(z)$$를 설정하는 것이 자유로워진다. 

posterior에 대한 정보가 없다고 할 때, $$q(z)$$를 정규 분포로 설정하자. 정규분포에서 K개의 z를 뽑아낸 후, KLD를 계산할 수 있다. 이후 정규분포의 parameter를 바꾸며 KLD를 최소화화는 정규분포를 발견해낼 수 있고 이렇게 구한 정규분포를 Variational Inference의 결과라 할 수 있다.

바로 위를 통해서 우리는 Variational Inference를 위해선 근사한 분포 $$q(z)$$와 posterior $$p(z\|x)$$의 KLD를 최소화시켜야한다는걸 알 수 있다. 하지만 우리는 곧바로 KLD를 최소화시키는 것은 불가능하다. 대신 ELBO(Evidence lower bound)라는 개념을 통해 KLD를 최소화시키는 것을 진행할 수 있다.

ELBO를 구하기 위해서는 observations에 대한 log (marginal) probability에 Jensen's inequality를 적용한다.

$$
\begin{align}
	\log{p(x)} &=\log{\int_{z}p(x,z)dz} \\
    &= \log{\int_{z}p(x,z)\frac{q(z)}{q(z)}} \\
    &= \log\left(\mathbb{E}_{q}\frac{p(x,z)}{q(z)}\right) \\
    &\geq \mathbb{E}_{q}\left[\log{p(x,z)}\right]-\mathbb{E}_{q}\left[\log{q(z)}\right]
\end{align}
$$

이 식에서 second term은 엔트로피(entropy)이며 두개의 Expectation이 모두 계산될 수 있는 variational distributions family를 선택해야 한다.

KLD는 다음과 같이 다시 적힐 수 있다.

$$
\begin{align}
	D_{KL}(q(z)|p(z|x)) &= \mathbb{E}_{q}\left[\log{\frac{q(z)}{p(z|x)}}\right] \\
    &= \mathbb{E}_{q}\left[\log{q(z)}\right]-\mathbb{E}_{q}\left[\log{p(z|x)}\right] \\
    &= -\left(\mathbb{E}_{q}\left[\log{p(x,z)}\right]-\mathbb{E}_{q}\left[\log{q(Z)}\right]\right)+\log{p(x)}
\end{align}
$$

즉 다음과 같이 evidence라고 불리기도 하는 $$\log{p(x)}$$는

$$\log{p(x)}=ELBO+D_{KL}(q(z)|p(z|x))$$

의 형태로 적힐 수 있고 KLD는 항상 0보다 크거나 같기 때문에 $$\log{p(x)} \geq ELBO$$ 이다. 등호는 $$q(z)=p(z\|x)$$ 일 때 성립한다. ELBO는 위에서 볼 수 있는 바와 같이 2가지 형태로 표현할 수 있다.

$$
\begin{align}
	ELBO &= \mathcal{E}_{q}\left[\log{p(x,z)}\right]-\mathcal{E}_{q}\left[\log{q(z)}\right] \\
    &= \mathcal{E}_{q}\left[\log{p(x|z)}\right]-D_{KL}(q(z)|p(z))
\end{align}
$$

$$ELBO=\log{p(x)}-D_{KL}(q(z)|p(z|x))$$

이기 때문에 ELBO를 maximize 하는 것은 KLD를 minimize하는 것과 동일하다.

#### Mean Field Variational Inference

latent variable에 대한 variational distribution이 다음과 같이 factorization 된다고 가정하자.

$$q(z_{1},...,z_{m}) = \prod_{j=1}^{m}q(z_{j})$$

Chain rule에 의해 다음과 같은 식을 얻을 수 있고

$$p(z_{1},...,z_{m},x_{1},...,x_{n})=p(x_{1},...,x_{n})\prod_{j=1}^{m}p(z_{j}|z_{1:(j-1)},x_{1},...x_{n})$$

ELBO의 엔트로피 부분은 다음과 같이 바꿀 수 있다.

$$\mathbb{E}_{q}\left[\log{(q_{1},...,q_{m})}\right]=\sum_{j=1}^{m}\mathbb{E}_{j}\left[\log{(q_{j})}\right] $$

여기서 $$\mathbb{E}_{j}$$란 $$q(z_{j})$$에 대한 기대값을 의미한다.

위에서 언급한 2가지 성질을 이용하여 ELBO($$\mathcal{L}$$)을 다음과 같이 적을 수 있다.

$$\mathcal{L}=\log p(x_{1},...,x_{n})+\sum_{j=1}^{m}\{\mathbb{E}\left[\log p(z_{j}|z_{1},...z_{j-1},x_{1},...,x_{n})\right]-\mathbb{E}_{j}\left[\log q(z_{j})\right]\}$$

ELBO를 $$q(z_{k})$$의 함수라 생각하고 variable $$z_{k}$$를 가장 마지막 variable이라 생각하고 Chain Rule를 사용하면 다음과 같은 objective function을 구할 수 있다.

$$
\begin{align}
	\mathcal{L}&=\mathbb{E}\left[\log{p(z_{k}|z_{-k},\mathbf{x})}\right]-\mathbb{E}_{j}\left[\log{q(z_{k})}\right]+C \\
    \mathcal{L}_{k}&=\int q(z_{k})\mathbb{E}_{-k}\left[\log{p(z_{k}|z_{-k},\mathbf{x})}\right]dz_{k} - \int q(z_{k})\log{q(z_{k})}dz_{k}
\end{align}
$$

$$q(z_{k})$$에 대한 derivative를 구하면 다음과 같다.

$$\frac{d\mathcal{L}_{k}}{dq(z_{k})}=\mathbb{E}_{-k}\left[\log{p(z_{k}|z_{-k},\mathbf{x})}\right]-\log{q(z_{k})}-1=0$$

이 결과를 바탕으로 $$q(z_{k})$$에 대한 coordinate ascent upate를 진행할 수 있고 posterior의 분모 부분이 $$z_{k}$$에 의존하지 않으므로

$$
\begin{align}
	q^{*}(z_{k}) &\propto exp\{\mathbb{E}_{-k}\left[\log p(z_{k}|z_{-k},\mathbf{x})\right]\} \\
    q^{*}(z_{k}) &\propto exp\{\mathbb{E}_{-k}\left[\log p(z_{k},z_{-k},\mathbf{x})\right]\}
\end{align}
$$

coordinate ascent algorithm은 각 $$q(z_{k})$$를 update하며, 그 결과 local maximum으로 수렴한다, $$q(z_{k})$$에 대한 coordinate ascent update는 오로지 $$q(z_{j}),k \neq j$$ 근사값에 의존한다.

Variational Inference의 coordinate ascent algorithm과 Gibbs Sampling은 비슷한 성질을 갖는다.

1. Gibbs Sampling은 conditional posterior로부터 sample하며
2. Variational Inference의 coordinate ascent algorithm은 다음과 같은 형태를 갖는다. $$q(z_{k}) \propto exp\{\mathbb{E}\left[\log{(conditional)}\right]\}$$













