---
layout: post
title:  "Laplace Approximation"
date: 2019-06-24
author: YoungHwan Seol
categories: Bayesian
---

라플라스 근사법(Laplace Approximation)은 임의의 함수를 특정 위치에서의 정규 분포로 근사하는 기법을 의미한다. 현재 게시하고 있는 베이즈 관점에서는 Posterior 분포의 가장 비슷한 형태의 가우시안 분포로 대응시키는 것을 의미한다고 할 수 있다.

단일 연속형 변수 $$z$$가 있다고 하자. $$z$$의 확률분포 $$p(z)$$는

$$
\begin{align}
	p(z) &= \frac{1}{Z}f(z) \\
    Z &= \int f(z)dz
\end{align}
$$

여기서 $$Z$$에 대해서는 알지 못한다고 하고 분포 $$p(z)$$의 최빈값을 중심으로 하는 가우시안 근사 $$q(z)$$를 찾아내도록 하자.

우선 다음의 수식 $$\frac{df(z)}{dz}\|_{z=z_{0}}=0$$ 을 통해서 $$p^{'}(z_{0})=0$$을 만족시키는 $$p(z)$$의 최빈값을 찾아낸다.

가우시안 분포는 로그를 취한 결과가 변수들의 이차함수 형태로 표현된다. 따라서 최빈값 $$z_{0}$$를 중심으로 하는 $$\log{f(z)}$$의 테일러 급수 전개를 생각해보자.

$$
\begin{align}
	\log{f(z)} &\simeq \log{f(z_{0})}-\frac{1}{2}A(z-z_{0})^{2} \\
    A &= -\frac{-d^{2}}{dz^{2}}\log{f(z)}|_{z=z_{0}}
\end{align}
$$

여기에 지수함수를 취하면 다음과 같이 $$f(z) \simeq f(z_{0})exp\left[-\frac{A}{2}(z-z_{0})^{2}\right]$$ 의 식을 구할 수 있다. 이로써 이제 가우시안 분포의 형태와 비슷해졌다.

다음과 같은 표준화 작업을 통해 정규화된 분포 $$q(z)$$를 구할 수 있다.

$$q(z) = \left(\frac{A}{2\pi}\right)^{1/2}\times exp\left[-\frac{A}{2}(z-z_{0})^{2}\right]$$

이제 이를 M차원으로 확장시켜보자. M차원 공간에서 $$\mathbf{z}$$에 대해 마찬가지로 다음의 관계가 성립할 것이다.

$$
\begin{align}
	p(\mathbf{z}) &= \frac{1}{\mathbf{Z}}f(\mathbf{z}) \\
    \mathbf{Z} &= \int f(\mathbf{z})d\mathbf{z}
\end{align}
$$

$$\mathbf{z}_{0}$$에서의 기울기는 $$\bigtriangledown f(\mathbf{z})=0$$ 일 것이다.

따라서 다음과 같이 테일러 급수 전개를 적용하면

$$
\begin{align}
	\log{f(\mathbf{z})} &\simeq \log{f(\mathbf{z}_{0})}-\frac{1}{2}(\mathbf{z}-\mathbf{z}_{0})^{T}\mathbf{A}(\mathbf{z}-\mathbf{z}_{0}) \\
    A &= -\bigtriangledown\bigtriangledown\log{f(\mathbf{z})}|_{\mathbf{z}=\mathbf{z}_{0}} \\
    f(\mathbf{z}) &= f(\mathbf{z}_{0})exp\left[-\frac{1}{2}(\mathbf{Z}-\mathbf{Z}_{0})^{T}\mathbf{A}(\mathbf{Z}-\mathbf{Z}_{0})\right]
\end{align}
$$

근사분포 $$q(\mathbf{z})$$는 $$f(\mathbf{z})$$에 비례하며

$$
\begin{align}
	q(\mathbf{z}) &= \frac{|A|^{1/2}}{(2\pi)^{M/2}} exp\left[-\frac{1}{2}(\mathbf{Z}-\mathbf{Z}_{0})^{T}\mathbf{A}(\mathbf{Z}-\mathbf{Z_{0}})\right] \\
    &= \mathcal{N}(\mathbf{Z}|\mathbf{Z}_{0},\mathbf{A}^{-1})
\end{align}
$$

이 때, $$M \times M$$ 크기의 Hessian matrix A는 positive definite이어야 한다.

위의 라플라스 근사법은 모델 선택과정 방법 중 하나인 BIC에 적용될 수 있다. 

$$
\begin{align}
	\mathbf{Z} &= \int f(\mathbf{z})d\mathbf{z} \\
    &\simeq f(\mathbf{z}_{0})\int exp\left[-\frac{1}{2}(\mathbf{z}-\mathbf{z}_{0})^{T}\mathbf{A}(\mathbf{z}-\mathbf{z}_{0})\right]d\mathbf{z} \\
    &= f(\mathbf{z}_{0})\times\frac{|2\pi|^{M/2}}{|A|^{1/2}}
\end{align}
$$

데이터 집합 $$\mathcal{D}$$와 모델집합 $$\{\mathbcal{M}_{i}\}$$를 고려하자. 이 모델들은 매개변수 $$\{\theta_{i}\}$$를 갖는다. 다양한 모델들 중에서 하나를 선택하는 것이 관심사라 할 때, 그 기준으로 모델 증거(model evidence) $$p(\mathbcal{D}\|\mathcal_{i})$$를 활용할 수 있다. 모델 증거에 대한 식은 다음과 같이 표현할 수 있다.

$$p(\mathcal{D}|\mathcal{M}_{i})=\int p(\mathcal{D}|\theta_{i},\mathcal{M}_{i})p(\theta_{i}|\mathcal{M}_{i})d\theta_{i}$$

베이즈 정리에 의하여 $$\int$$의 뒤에 위치한 부분은 다음과 같이 표현될 수 있다.

$$
\begin{align}
	p(\theta_{i}|\mathcal{D},\mathcal{M}_{i}) &\propto p(\theta_{i}|\mathcal{M}_{i})\times p(\mathcal{D}|\mathcal{M}_{i},\theta_{i}) \\
    \log{p(\theta_{i}|\mathcal{D},\mathcal{M}_{i})} &= \log{p(\theta_{i}|\mathcal{M}_{i})}+ \log{p(\mathcal{D}|\mathcal{M}_{i},\theta_{i})}+C
\end{align}
$$

이제 다음과 같이 $$\mathbb{E}(\theta_{i})$$와 $$\hat{\theta}_{i}$$를 정의하자.
$$
\begin{align}
	\mathbb{E}(\theta_{i}) &= -\log{p(\theta_{i}|\mathcal{M}_{i})}-\log{p(\mathcal{D}|\mathcal{M}_{i},\theta_{i})} \\
    \hat{\theta}_{i} &= argmin_{\theta_{i}} \mathbb{E}(\theta_{i})
\end{align}
$$

$$\int p(\mathcal{D}|\mathcal{M}_{i},\theta_{i})p(\theta_{i}|\mathcal{M}_{i})d\theta_{i} = \int exp(-\mathbb{E}(\theta_{i}))d\theta_{i}$$

이제 $$exp(-\mathbb{E}(\theta_{i}))$$를 라플라스 근사법을 사용해 근사할 것이다.

