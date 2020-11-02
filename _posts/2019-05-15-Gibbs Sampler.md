---
layout: post
title:  "깁스 샘플러(Gibbs Sampler)"
date: 2019-05-22
author: seolbluewings
categories: 샘플링
---

깁스 샘플러(Gibbs Sampler)는 대표적인 MCMC기법 중 하나로 $$d$$차원의 parameter인 $$\Theta = (\theta_{1},....,\theta_{d})$$의 sampling을 진행할 수 있도록 하는 방법이다.

우리가 sampling을 실행하기 희망하는 target distribution $$P(\Theta) = P(\theta_{1},...,\theta_{d})$$ 로부터 sampling을 진행하기 어려울 때, Gibbs Sampler를 활용할 수 있다.

다음과 같이 모수가 존재한다고 하자.

$$\Theta=(\theta_{1},\theta_{2},\theta_{3})$$

모수들의 결합 사후분포(joint posterior distribution)를 아는 것이 좋겠지만, 이 joint posterior distribution $$ p(\theta_{1},\theta_{2},\theta_{3}\mid \mathbf{X})$$ 가 계산하기 어려운 형태로 주어지는 반면, 완전 조건부 사후분포(full conditional posterior distribution)는 오히려 계산하기 쉬운 형태(즉, 우리가 알고 있는 분포의 형태)로 주어질 때가 있다.

full conditional posterior distribution 이란, 관심 parameter를 제외한 나머지가 모두 주어진 조건인 분포를 말하며 아래와 같다.

$$p(\theta_{1}\mid\theta_{2},\theta_{3},\mathbf{X})$$

$$p(\theta_{2}\mid\theta_{1},\theta_{3},\mathbf{X})$$

$$p(\theta_{3}\mid\theta_{1},\theta_{2},\mathbf{X})$$

Gibbs Sampler의 각 과정은 다음과 같이 진행된다. 우선 $$\theta_{1},\theta_{2},\theta_{3}$$ 의 initial value인 $$\theta_{1}^{(0)},\theta_{2}^{(0)},\theta_{3}^{(0)}$$ 를 사용자 임의로 정한다.

첫번째 Step은 다음과 같다.

$$
\begin{align}
\theta_{1}^{(1)} &\sim p(\theta_{1}\mid\theta_{2}^{(0)},\theta_{3}^{(0)}|\bf{X}) \nonumber \\
\theta_{2}^{(1)} &\sim p(\theta_{2}\mid\theta_{1}^{(1)},\theta_{3}^{(0)}|\bf{X}) \nonumber \\
\theta_{3}^{(1)} &\sim p(\theta_{3}\mid\theta_{1}^{(1)},\theta_{2}^{(1)}|\bf{X})
\nonumber
\end{align}
$$


두번째 Step은 다음과 같다. 첫번째 Step에서 구한 나머지 parameter의 값을 이용하여 Markov Chain을 형성한다.

$$
\begin{align}
\theta_{1}^{(2)} &\sim p(\theta_{1}\mid\theta_{2}^{(1)},\theta_{3}^{(1)}|\bf{X}) \nonumber \\
\theta_{2}^{(2)} &\sim p(\theta_{2}\mid\theta_{1}^{(2)},\theta_{3}^{(1)}|\bf{X}) \nonumber \\
\theta_{3}^{(2)} &\sim p(\theta_{3}\mid\theta_{1}^{(2)},\theta_{2}^{(2)}|\bf{X})
\nonumber
\end{align}
$$

이처럼 여러 Step을 반복하면 $$m$$번째 Step은 다음과 같을 것이다.

$$
\begin{align}
\theta_{1}^{(m)} &\sim p(\theta_{1}\mid\theta_{2}^{(m-1)},\theta_{3}^{(m-1)}|\bf{X}) \nonumber \\
\theta_{2}^{(m)} &\sim p(\theta_{2}\mid\theta_{1}^{(m)},\theta_{3}^{(m-1)}|\bf{X}) \nonumber \\
\theta_{3}^{(m)} &\sim p(\theta_{3}\mid\theta_{1}^{(m)},\theta_{2}^{(m)}|\bf{X})
\nonumber
\end{align}
$$

이를 일반적인 방식으로 표현하면, 즉 관심모수가 다음과 같이 $$d$$차원인 경우를 생각해보자.

$$\Theta=(\theta_{1},\theta_{2},\theta_{3},....,\theta_{d})$$

그리고 $$\theta_{-k}^{(t)}=(\theta_{1}^{(t+1)},\theta_{2}^{(t+1)},\theta_{3}^{(t+1)},...,\theta_{k-1}^{(t+1)},\theta_{k+1}^{(t)},....\theta_{d}^{(t)})$$ 라고 정의를 하면 매 Step의 각각의 단계는 다음과 같이 표기할 수 있다.

$$\theta_{k}^{(t+1)} \sim p(\theta_{k}|\theta_{-k}^{(t)})$$

이렇게 분포상 given 조건에 들어가는 parameter 값을 최신값으로 대체하면서 차례대로 $$\theta_{1},\theta_{2},\theta_{3}$$를 각각의 full conditional distribution으로부터 추출해낸다.

지금까지의 내용을 통해 확인할 수 있는 사항은 다음과 같다.

$$\theta_{1}^{(m)}$$은 가장 최근의 $$\theta_{2},\theta_{3}$$값인 $$(\theta_{2}^{(m-1)},\theta_{3}^{(m-1)})$$ 에만 의존할 뿐, 그 이전의 $$\theta_{2},\theta_{3}$$ 값에는 의존하지 않는다. 그러므로 이는 Markov Chain이라 할 수 있다.

우리는 $$t$$시점의 parameter $$\Theta^{(t)} = (\theta_{1}^{(t)},\theta_{2}^{(t)},\theta_{3}^{(t)})$$ 을 이용하여 posterior distribution의 sampling을 시행한다.

Gibb Sampler의 특징은 $$\Theta^{(t)} = (\theta_{1}^{(t)},\theta_{2}^{(t)},\theta_{3}^{(t)})$$ 가 Markov Chain이기 때문에 독립적인 표본이 아니라는 것이다. 독립적인 표본을 얻기 위해 아래 2가지 방법 중 하나를 활용한다.

1. Gibbs Sampler를 N번 독립적으로 시행하여 $$\Theta_{1}^{(m)},...,\Theta_{N}^{(m)}$$을 얻는다. 여기서 $$\Theta_{k}^{(m)}$$ 이란 k번째 독립된 Gibbs Sampler에서 만들어낸 $$\Theta^{(m)}$$ 이다. 이 방법을 통하서 독립적인 표본을 구할 수 있지만, 시간이 많이 걸린다는 단점이 있다.

2. Gibbs Sampler를 충분히 많이 돌리고 $$m$$번째 iteration 이후에서는 크기 $$\mathit{l}$$ 만큼의 간격으로 표본을 추출해낸다. $$\mathit{l}$$의 크기만 적당하다면, 각각은 서로의 연관성이 약해져 독립적인 표본이라 간주할 수 있다. 적당한 크기의 $$\mathit{l}$$을 구하기 위해서는 parameter의 ACF Plot을 통해 확인해볼 수 있다.

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

따라서 Gibbs Sampler는 항상 새롭게 proposed되는 parameter의 값을 accept하는 M-H 알고리즘이라 할 수 있다.

#### 예제

$$
\begin{align}
\mathbf{X} &\sim \mathcal{N}(\mu,\sigma^{2}) \nonumber \\
\mu &\sim \mathcal{N}(\mu_{0},\sigma^{2}_{0}) \nonumber \\
\sigma^{2} &\sim \mathcal{IG}(\alpha,\beta) \nonumber \\
(x_{1},...,x_{10}) &= (10,13,15,11,9,18,20,17,23,21) \nonumber
\end{align}
$$

이 때, $$\Theta = (\mu,\sigma^{2})$$ 에 대한 Gibbs Sampler를 진행하라.

#### Sol)

Step 1.) target posterior distribution을 도출하기

$$
\begin{align}
p(\mu,\sigma^{2}\mid \mathbf{X}) &\propto p(\mathbf{X}\mid\mu,\sigma^{2})p(\mu\mid\sigma^{2})p(\sigma^{2}) \nonumber \\
&\propto p(\mathbf{X}\mid\mu,\sigma^{2})p(\mu\mid\mu_{0},\sigma^{2}_{0})p(\sigma^{2}\mid\alpha,\beta) \nonumber
\end{align}
$$

Step 2.) $$\mu$$에 대한 Gibbs Sampler sampling step을 구하기

$$
\begin{align}
p(\mu\mid\mathbf{X},\sigma^{2}) &\propto p(\mathbf{X}\mid\mu,\sigma^{2})p(\mu\mid\mu_{0},\sigma^{2}_{0}) \nonumber \\
&\propto \text{exp}\left[\frac{-n}{2\sigma^{2}}(\mu^{2}-2\bar{x}\mu)+\frac{-1}{2\sigma^{2}_{0}}(\mu^{2}-2\mu_{0}\mu)\right] \nonumber \\
&\propto \text{exp}\left[\frac{-1}{2}\mu^{2}\left(\frac{1}{\sigma^{2}/n+\frac{1}{\sigma_{0}^{2}}} \right)-2\mu\left(\frac{\bar{x}}{\sigma^{2}/n}+\frac{\mu_{0}}{\sigma^{2}_{0}}\right)  \right] \nonumber \\
p(\mu\mid\sigma^{2},\mathbf{X}) &\sim \mathcal{N}\left(\frac{ \frac{\bar{x}}{\sigma^{2}/n} + \frac{\mu_{0}}{\sigma^{2}_{0}} }{ \frac{n}{\sigma^{2}} + \frac{1}{\sigma^{2}_{0}} } , \frac{1}{ \frac{n}{\sigma^{2}} + \frac{1}{\sigma^{2}_{0}} }  \right) \nonumber
\end{align}
$$

Step 3.) $$\sigma^{2}$$에 대한 Gibbs Sampler sampling step을 구하기

$$
\begin{align}
p(\sigma^{2}\mid\mu,\mathbf{X}) &\propto p(\mathbf{X}\mid\mu,\sigma^{2})p(\sigma^{2}) \nonumber \\
&\propto (\sigma^{2})^{-n/2}\text{exp}\left[\frac{-1}{2\sigma^{2}}\sum_{i=1}^{n}(\mu-x_{i})^{2}\right] (\sigma^{2})^{-\alpha-1}\text{exp}(-b/\sigma^{2}) \nonumber \\
&\propto (\sigma^{2})^{-n/2-\alpha-1}\text{exp}\left[\frac{-1}{\sigma^{2}}\left(\frac{1}{2}\sum_{i=1}^{n}(\mu-x_{i})^{2}\right) + \beta  \right] \nonumber \\

p(\sigma^{2}\mid\mu,\mathbf{X}) &\sim \mathcal{IG}\left(\frac{n}{2}+\alpha, \frac{1}{2}\sum_{i=1}^{n}(\mu-x_{i})^{2}+\beta \right)
\end{align}
$$

##### 상기 예제에 관련한 코드는 다음의 링크 1. [R코드](https://github.com/seolbluewings/rcode/blob/master/Gibbs%20Sampler.ipynb) 2. [Python코드](https://github.com/seolbluewings/pythoncode/blob/master/5.Gibbs%20Sampler.ipynb) 에서 확인할 수 있습니다.



#### 참조 문헌

1. [BDA](http://www.stat.columbia.edu/~gelman/book/)