---
layout: post
title:  "Variational Inference"
date: 2019-05-26
author: YoungHwan Seol
categories: Bayesian
---

Variational Inference란 복잡한 형태의 posterior 분포 $$p(z\|x)$$를 다루기 쉬운 형태의 $$q(z)$$로 근사하는 것을 말한다.

다음과 같은 형태의 posterior distribution이 있다고 하자. 

$$p(z|x) = \frac{p(z,x)}{\int_{z}p(z,x)dz}$$ 

여기서 분모의 적분이 계산하기 어려운 경우 Variational Inference를 사용한다고 할 수 있다.

Variational Inference의 핵심적인 아이디어는 다음과 같다.

1. variational parameter $$\nu$$를 갖는 latent variables \{z_{1},z_{2},...,z_{m}\}의 분포$$q(z_{1},z_{2},...,z_{m}\|\nu)$$를 찾는다.
2. 이 분포를 찾아가는 과정에서 posterior distribution에 가장 가까이 근사하는 모수 $$\nu$$를 찾아낸다.
3. 이렇게 구한 분포 $$q$$를 posterior 대신 사용한다.

그렇다면, 우리는 posterior distribution에 근사한 $$q(z)$$를 만들기 위해 쿨백-라이블러 발산(Kullback-Leibler Divergence, 이하 KLD)에 대해 이해해야 한다. 

$$KL(q||p) = \int_{z}q(z)\log{\frac{q(z)}{p(z|x)}}=\mathbb{E}\bigg[\log{\frac{q(z)}{p(z|x)}}\bigg]$$

KLD은 두 확률분포의 차이를 계산할 수 있는 방식으로 $$p(z\|x)$$와 $$q(z)$$의 KLD 값을 계산한 이후, KLD이 줄어드는 방향으로 $$q(z)$$를 update하는 과정을 반복하면, posterior를 잘 근사하는 $$q^{*}(z)$$를 얻을 수 있다.

$$
\begin{align}
	D_{KL}(q(z)|p(z|x)) &= \mathbb{E}_{q}[\log{\frac{q(z)}{p(z|x)}}] = \int q(z)\log{\frac{q(z)}{p(z|x)}}dz \\
	&= \int q(z) \log{\frac{q(z)p(x)}{p(x|z)p(z)}} dz \\
	&= \int q(z) \log{\frac{q(z)}{p(z)}}dz + \int q(z)\log{p(x)}dz - \int q(z)\log{p(x|z)}dz \\
	&= D_{KL}(q(z)|p(z)) + \log{p(x)}-\mathbb{E}_{q}[\log{p(x|z)}]
\end{align}
$$

몬테 카를로 방법(Monte Carlo Method)을 KLD에 적용하면 다음과 같다.

$$
\begin{align}
	D_{KL}(q(z)||p(z|x)) &= D_{KL}(q(z)|p(z))+\log{p(x)}-\mathbb{E}_{q}\left[\log{p(x|z)}\right] \\
    &= E_{q}\left[\log{\frac{q(z)}{p(z)}}\right]+\log{p(x)}-\mathbb{E}_{q}\left[\log{p(x|z)\right] \\
    &\backsimeq \frac{1}{K}\sum_{i=0}^{K}\left[\log{\frac{q(z_{i})}{p(z_{i})}}\right] +\log{p(x)}-\frac{1}{K}
\end{align}
$$
