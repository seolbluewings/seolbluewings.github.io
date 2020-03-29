---
layout: post
title:  "깁스 샘플러(Gibbs Sampler)"
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

Gibbs Sampler는 Metropolis Hastings Algorithm의 하나의 special case라고 할 수 있다.

Metropolis Hastings Algorithm에서 사용하는 Transition Kernel이 다음과 같이 주어진다고 가정해보자.

$$
\begin{equation}
T_{k}(\theta^{*} \mid \theta^{(t)}) =\left \{\begin{array}{ll}
p(\theta_{k}^{*} \mid \theta_{-k}^{(t)}) \quad \text{if} \; \theta_{-k}^{(t)}=\theta_{-k}^{*} \\
0 \quad \text{otherwise}
\end{array}
\right.
\end{equation}
$$

따라서 M-H 알고리즘에서 활용되는 acceptance probability는 다음과 같이 계산될 수 있다.

$$
\begin{align}
	\alpha &= \frac{\pi(\theta^{*})/T_{k}(\theta^{*}\mid\theta^{(t)})}{\pi(\theta^{(t)})/T_{k}(\theta^{(t)}\mid\theta^{*})} \\
    &= \frac{\pi(\theta^{*})/p(\theta_{k}^{*}\mid\theta_{-k}^{(t)})}{\pi(\theta^{(t)})/p(\theta_{k}^{(t)}\mid\theta_{-k}^{*})} \\
    &=\frac{\pi(\theta_{-k}^{*})}{\pi(\theta_{-k}^{(t)})}=1 \\
    \because \pi(\theta^{*}) &= \frac{\pi(\theta_{k}^{*},\theta_{-k}^{(t)})}{\pi(\theta_{k}^{*}\mid\theta_{-k}^{(t)})} = \pi(\theta_{-k}^{*})
\end{align}
$$

따라서 Gibbs Sampler는 항상 새롭게 proposed 되는 parameter의 값을 accept하는 M-H 알고리즘이라 할 수 있다.

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

바로 위의 간단한 Gibbs Sampler 시행 코드와 결과는 다음과 같다.

~~~
x = c(10,13,15,11,9,18,20,17,23,21)
xbar = mean(x)
xsig = var(x)
n=length(x)

### setting prior ###
mu0 = 10
sig0 = 100
alpha = 1
beta = 1

iter = 10000
theta = matrix(0,ncol=2,nrow=iter)

for(t in 2:iter){
  ### set initial value
  theta[1,1] = 0
  theta[1,2] = 1
  
  ### sampling mu from posterior ###
  new_sig = 1/((n/xsig)+(1/sig0))
  new_mu = new_sig*((n/xsig)*xbar+(1/sig0)*mu0)
  theta[t,1] = rnorm(1,new_mu,sqrt(new_sig))
  
  ### sampling sigma from posterior ###
  new_alpha = n/2+alpha
  new_beta = 0.5*sum((x-theta[t,1])^2)+beta
  theta[t,2] = 1/rgamma(1,shape=new_alpha,rate=new_beta)
}
~~~

Gibbs Sampler에서 $$(\mu,\sigma^{2})$$ 표본의 경로는 아래의 그림과 같다.
~~~
par(mfrow=c(1,3))
plot(theta[1:10,],type="n",xlab=expression(mu),ylab=expression(sigma^2))
lines(theta[1:10,],lty=2)
for(i in 1:10){
  text(theta[i,1],theta[i,2],i)
}

plot(theta[1:100,],type="n",xlab=expression(mu),ylab=expression(sigma^2))
lines(theta[1:100,],lty=2)
for(i in 1:100){
  text(theta[i,1],theta[i,2],i)
}

plot(theta[1:10000,],type="n",xlab=expression(mu),ylab=expression(sigma^2))
lines(theta[1:10000,],lty=2)
for(i in 1:10000){
  text(theta[i,1],theta[i,2],i)
}
~~~

![Gibbs Sampler](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Gibbs.png?raw=true)


