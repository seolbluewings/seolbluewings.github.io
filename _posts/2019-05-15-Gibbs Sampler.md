---
layout: post
title:  "Gibbs Sampler"
date: 2019-05-22
author: YoungHwan Seol
categories: Bayesian
---

다음과 같은 모수가 존재한다고 하자. 

$$\mathbf{\theta}=(\theta_{1},\theta_{2},\theta_{3})$$ 

모수들의 결합 사후분포(joint posterior distribution)를 아는 것이 좋겠지만, 이 결합 사후분포

$$p(\theta_{1},\theta_{2},\theta_{3}|\mathbf{X})$$

가 계산하기 어려운 형태로 주어지는 반면, 완전 조건부 사후분포(full conditional posterior distribution)이 계산하기 쉬운 형태로 주어질 때가 있다. 

완전 조건부 사후분포란, 관심모수를 제외한 나머지가 모두 주어진 조건인 분포를 말하며 다음과 같다.

$$p(\theta_{1}|\theta_{2},\theta_{3},\mathbf{X})$$

$$p(\theta_{2}|\theta_{1},\theta_{3},\mathbf{X})$$ 

$$p(\theta_{3}|\theta_{1},\theta_{2},\mathbf{X})$$

Gibbs Sampler의 과정은 다음과 같다.

우선 $$\theta_{1},\theta_{2},\theta_{3}$$ 의 initial value인 $$\theta_{1}^{(0)},\theta_{2}^{(0)},\theta_{3}^{(0)}$$ 를 정한다.

첫번째 step은 다음과 같다.

$$\theta_{1}^{(1)} \sim p(\theta_{1}|\theta_{2}^{(0)},\theta_{3}^{(0)}|\bf{X})$$

$$\theta_{2}^{(1)} \sim p(\theta_{2}|\theta_{1}^{(1)},\theta_{3}^{(0)}|\bf{X})$$

$$\theta_{3}^{(1)} \sim p(\theta_{3}|\theta_{1}^{(1)},\theta_{2}^{(1)}|\bf{X})$$

두번째 step은 다음과 같다.

$$\theta_{1}^{(2)} \sim p(\theta_{1}|\theta_{2}^{(1)},\theta_{3}^{(1)}|\bf{X})$$

$$\theta_{2}^{(2)} \sim p(\theta_{2}|\theta_{1}^{(2)},\theta_{3}^{(1)}|\bf{X})$$

$$\theta_{3}^{(2)} \sim p(\theta_{3}|\theta_{1}^{(2)},\theta_{2}^{(2)}|\bf{X})$$

이처럼 여러 step을 반복하면 m번째 step은 다음과 같을 것이다.

$$\theta_{1}^{(m)} \sim p(\theta_{1}|\theta_{2}^{(m-1)},\theta_{3}^{(m-1)}|\bf{X})$$

$$\theta_{2}^{(m)} \sim p(\theta_{2}|\theta_{1}^{(m)},\theta_{3}^{(m-1)}|\bf{X})$$

$$\theta_{3}^{(m)} \sim p(\theta_{3}|\theta_{1}^{(m)},\theta_{2}^{(m)}|\bf{X})$$

이를 일반적인 방식으로 표현하면 다음과 같다. 관심모수가 다음과 같이 d 차원인 경우를 생각해보자.

$$\Theta=(\theta_{1},\theta_{2},\theta_{3},....,\theta_{d})$$

그리고 $$\theta_{-k}^{(t)}=(\theta_{1}^{(t+1)},\theta_{2}^{(t+1)},\theta_{3}^{(t+1)},...,\theta_{k-1}^{(t+1)},\theta_{k+1}^{(t)},....\theta_{d}^{(t)})$$ 라고 정의를 하면 매 step의 각각의 단계는 다음과 같이 표기할 수 있다. 

$$\theta_{k}^{(t+1)} \sim p(\theta_{k}|\theta_{-k}^{(t)})$$ 

이렇듯 조건에 들어가는 모수값을 가장 최근의 값으로 대체하면서 차례대로 $$\theta_{1},\theta_{2},\theta_{3}$$를 완전 조건부 사후분포로부터 추출해내면 m이 충분히 클 때, $$\Theta^{(m)} = (\theta_{1}^{(m)},\theta_{2}^{(m)},\theta_{3}^{(m)})$$ 은 결합 사후분포(joint posterior distribution)의 분포를 따르므로 $$\Theta^{(m)} = (\theta_{1}^{(m)},\theta_{2}^{(m)},\theta_{3}^{(m)})$$ 을 $$\Theta$$ 의 posterior distribution에서 만들어진 표본으로 사용한다.

충분히 큰 m과 N에 대하여

$${\hat E}[\psi(\Theta)|x]=\frac{1}{N}\sum_{i=1}^{N}\psi(\Theta^{(m+i)})$$

식을 얻을 수 있다. 여기서 m은 $$\Theta^{(i)}$$의 분포가 $$p(\Theta\|x)$$로 수렴하기까지 걸리는 iteration 횟수를 의미한다.

지금까지의 내용을 통해 확인할 수 있는 사항은 다음과 같다.

$$\theta_{1}^{(m)}$$은 가장 최근의 $$\theta_{2},\theta_{3}$$값인 $$(\theta_{2}^{(m-1)},\theta_{3}^{(m-1)})$$ 에만 의존할 뿐, 그 이전의 $$\theta_{2},\theta_{3}$$ 값에는 의존하지 않는다. 그러므로 $$\Theta^{(i)} = (\theta_{1}^{(i)},\theta_{2}^{(i)},\theta_{3}^{(i)})$$ 은 Markov Chain이며, 우리는 표본 $$\Theta^{(i)} = (\theta_{1}^{(i)},\theta_{2}^{(i)},\theta_{3}^{(i)})$$ 을 이용하여 posterior distribution을 구하므로 Gibbs Sampler는 Markov Chain Monte Carlo(MCMC) 방법이다.

MCMC의 특징은 $$\Theta^{(i)} = (\theta_{1}^{(i)},\theta_{2}^{(i)},\theta_{3}^{(i)})$$ 가 Markov Chain이기 때문에 독립적인 표본이 아니라는 것이다. 독립적인 표본을 얻기 위해 다음과 같은 방법을 활용한다.

1. Gibbs Sampler를 N번 독립적으로 시행하여 $$\Theta_{1}^{(m)},...,\Theta_{N}^{(m)}$$을 얻는다. 여기서 $$\Theta_{k}^{(m)}$$ 이란 k번째 독립된 Gibbs Sampler에서 만들어낸 $$\Theta^{(m)}$$ 이다. 이 방법을 통하서 독립적인 표본을 구할 수 있지만, 시간이 많이 걸린다는 단점이 있다.

2. Gibbs Sampler를 충분히 많이 돌리고 m번째 iteration 이후에서는 크기 $$\mathit{l}$$ 만큼의 간격으로 표본을 추출해낸다. 즉, $$\Theta_{1}^{(m)},\Theta_{1}^{(m+l)},...,\Theta_{N}^{(m+(N-1)l)}$$ 을 구하는 것이다. $$\mathit{l}$$의 크기만 적당하다면, 각각은 서로의 연관성이 약해져 독립적인 표본이라 간주할 수 있다.  

다음과 같은 예시를 생각해보자.

$$X_{1},...,X_{10} \sim \mathcal{N}(\mu,\sigma^{2})$$

$$\mu \sim \mathcal{N}(\mu_{0},\sigma^{2}_{0})$$

$$\sigma^{2} \sim \mathcal{IG}(a,b)$$

$$x_{1},...,x_{10}=(10,13,15,11,9,18,20,17,23,21)$$일 때, $$\theta=(\mu,\sigma^{2})$$ 에 대해 Gibbs Sampler를 적용해보자.

우리는 다음과 같은 conditional posterior distribution을 구할 수 있다.

$$p(\mu|\sigma^{2}) \sim \mathcal{N}(\mu_{\pi},\sigma^{2}_{\pi})$$

$$\mu_{\pi}=\frac{\frac{n}{\sigma^{2}}{\bar x}+\frac{1}{\sigma^{2}_{0}}\mu_{0}}{\frac{n}{\sigma^{2}}+\frac{1}{\sigma^{2}_{0}}}$$

$$ \sigma^{2}_{\pi} = \frac{1}{\frac{n}{\sigma^{2}}+\frac{1}{\sigma^{2}_{0}}} $$

$$ p(\sigma^{2}|X)=p(X|\sigma^{2})\cdot p(\sigma^{2}) $$

$$ p(\sigma^{2}|X) \propto (\sigma^{2})^{-n/2-a-1}\cdot exp\{-\frac{1}{\sigma^{2}}(\beta+\frac{\sum(x_{i}-\mu)^{2}}{2})\}$$

간단한 Gibbs Sampler 시행 결과는 다음과 같다. 

![Gibbs Sampler](/images/gibbs_sampler_posterior.png)

Gibbs Sampler에서 $$(\mu,\sigma^{2})$$ 표본의 경로는 아래의 그림과 같다. 아래의 그림은 $$(\mu,\sigma^{2})$$의 이동경로를 나타낸 그림으로 첫 5회, 15회, 100회의 이동 경로를 보여준다.

![Gibbs Sampler](\images/gibbs_sampler_path.png)


