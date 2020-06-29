---
layout: post
title:  "EM 알고리즘(EM Algorithm)"
date: 2020-06-27
author: YoungHwan Seol
categories: Statistics
---

EM알고리즘은 잠재변수(latent variable)를 갖는 확률 모델의 MLE 값을 찾는 과정에서 활용되는 기법이다. 잠재변수를 활용하는 가우시안 혼합 모델에 관한 추정에 자주 활용되는 기법이기도 하다.

우선 관측변수 $$\mathbf{X}=\{x_{1},...x_{N}\}$$ 이라 하자. 잠재변수 $$\mathbf{Z}=\{z_{1},z_{2},...z_{n}\}$$ 는 이 논의를 진행하는 과정에서 이산형 변수라고 가정하자. 만약 $$\mathbf{Z}$$가 연속형이라면, 아래의 과정에서 합표기가 되어있는 것을 적분으로 바꾸면 된다. 그리고 모델에 활용되는 Parameter들을 $$\Theta$$ 라고 표현하자.

우리의 목표는 $$P(\mathbf{X} \mid \Theta)$$ 혹은 $$l(\Theta \mid \mathbf{X})$$ 를 최대화시키는 $$\Theta$$ 값을 구하는 것이다. 이 $$P(\mathbf{X} \mid \Theta)$$에 대한 log-likelihood는 다음과 같이 표현할 수 있다.

$$
\text{ln}P(\mathbf{X}\mid\Theta) = \text{ln}\{\sum_{\mathbf{Z}}P(\mathbf{X},\mathbf{Z}\mid\Theta)\}
$$

$$\mathbf{X}$$ 의 관측값에 따른 잠재변수 $$\mathbf{Z}$$를 알게 되었다고 한다면, $$\{\mathbf{X},\mathbf{Z}\}$$ 를 완전한(complete) 데이터 집합이라 표현할 수 있다. $$\mathbf{Z}$$는 기존에 알지 못했던 것이기 때문에 missing value, $$\mathbf{X}$$는 기존에 알고 있었기 때문에 observed value라고 할 수 있다.

EM알고리즘은 observed variable $$\mathbf{X}$$를 이용한 log-likelihood의 최대화보다 complete data $$\{\mathbf{X},\mathbf{Z}\}$$ 를 활용한 log-likelihood의 최대화가 보다 쉽다는 가정에서 시작한다.

실제 상황에서는 완전한 데이터 집합 $$\{\mathbf{X},\mathbf{Z}\}$$이 주어지지 않고 불완전한 $$\mathbf{X}$$만 주어질 가능성이 아주 높다. 잠재변수 $$\mathbf{Z}$$에 대해서는 $$\mathbf{Z}$$에 대한 posterior distribution인 $$P(\mathbf{Z}\mid\mathbf{X},\Theta)$$를 통해서만 확인할 수 있다. 그래서 우리는 잠재 변수의 Posterior distribution을 통해 기대값을 구하고 이를 활용하여 완전한 데이터셋 $$\{\mathbf{X},\mathbf{Z}\}$$에 대한 log-likelihood를 구할 수 있다.

여기서 잠재변수의 Posterior distribution을 활용해 기대값을 구하는 것이 EM 알고리즘의 E(Expectation) 단계이다. M(Maximization) 단계는 E단계에서 구한 기대값을 최대화시키는 $$\Theta$$ 값을 추정하는 단계이다. 각 parameter에 대한 현재값을 $$\Theta^{(t)}$$라 하고 E,M 단계를 거쳐서 수정된 parameter값을 $$\Theta^{(t+1)}$$ 라고 표기한다.

E단계에서는 현재의 parameter $$\Theta^{(t)}$$를 이용하여, $$P(\mathbf{Z}\mid\mathbf{X},\Theta^{(t)})$$ 형태의 $$\mathbf{Z}$$ 의 Posterior distribution을 먼저 구한다. 그리고 이 Posterior distribution을 이용하여 다음의 값을 구한다.

$$
\begin{align}
Q(\Theta\mid\Theta^{(t)}) &= \mathbb{E}_{\mathbf{Z}}[l(\Theta\mid \mathbf{X},\mathbf{Z})\mid \mathbf{X},\Theta^{(t)}] \\ \nonumber
&= \sum_{\mathbf{Z}}P(\mathbf{Z}\mid\mathbf{X},\Theta^{(t)})\text{ln}P(\mathbf{X},\mathbf{Z}\mid\Theta) \nonumber
\end{align}
$$

M단계에서는 E단계에서 구한 $$Q$$ 함수를 최대화시키는 parameter 값을 찾는다.

$$\Theta^{(t+1)} = \text{argmax}_{\Theta}Q(\Theta\mid\Theta^{(t)})$$

전반적인 과정을 살펴본다면 EM 알고리즘은 다음의 4가지 단계로 구성된다.

1. $$P(\mathbf{X},\mathbf{Z}\mid \Theta)$$ 의 log-likelihood를 구한다.
2. 1단계에서 구한 log-likelihood를 이용하여 $$Q$$함수를 찾는다. (E-step)
3. 2단계에서 구한 $$Q$$함수를 최대화시키는 parameter 값을 구한다. (M-step)
4. parameter 값이 수렴할 때까지 해당 과정을 반복하고 parameter의 변동이 없다고 생각될 때, 반복 시행을 중단한다.


#### EM알고리즘의 Ascent Property

EM알고리즘은 매 반복마다 $$l(\Theta\mid \mathbf{X})$$가 증가한다는 성질을 갖는다. 이 말을 좀 더 수식으로 이야기하자면, $$l(\Theta^{(t+1)}\mid\mathbf{X}) \geq l(\Theta^{t}\mid\mathbf{X})$$ 라는 것이다. 이는 매 반복시행마다 log-likelihood를 증가시키는 $$\Theta^{(t+1)}$$를 찾는다는 것으로 이 알고리즘이 MLE를 찾는다는 것과 같은 의미이다.

완전한 데이터 $$\{\mathbf{X},\mathbf{Z}\}$$ 의 joint distribution은 다음과 같이 표현된다.

$$
\begin{align}
P(\mathbf{X},\mathbf{Z}\mid \Theta) &= P(\mathbf{X}\mid \Theta)P(\mathbf{Z}\mid\mathbf{X},\Theta) \\ \nonumber
\text{ln}P(\mathbf{X},\mathbf{Z}\mid\Theta) &= \text{ln}P(\mathbf{X}\mid\Theta) + \text{ln}P(\mathbf{Z}\mid\mathbf{X},\Theta) \nonumber
\end{align}
$$

아래의 식 양변에 기대값을 취해보자.

$$
\begin{align}
\mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{X}\mid\Theta)\mid \mathbf{X},\Theta^{(t)}] &= \mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{X},\mathbf{Z}\mid\Theta)\mid \mathbf{X},\Theta^{(t)}] - \mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{Z}\mid \mathbf{X},\Theta)\mid \mathbf{X},\Theta^{(t)}] \\ \nonumber

\text{ln}P(\mathbf{X}\mid\Theta) &= Q(\Theta\mid\Theta^{(t)})-\mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{Z}\mid \mathbf{X},\Theta)\mid \mathbf{X},\Theta^{(t)}] \nonumber
\end{align}
$$

이제 이 식을 활용하여 $$l(\Theta^{(t+1)}\mid\mathbf{X})- l(\Theta^{t}\mid\mathbf{X}) \geq 0$$ 임을 보일 것이다. 수식 표기의 간결성을 위해 $$\mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{Z}\mid \mathbf{X},\Theta)\mid \mathbf{X},\Theta^{(t)}]$$ 를 $$H(\Theta\mid\Theta^{(t)})$$ 로 표기한다.

$$
l(\Theta^{(t+1)}\mid\mathbf{X})- l(\Theta^{t}\mid\mathbf{X}) = Q(\Theta^{(t+1)}\mid\Theta^{(t)})-Q(\Theta^{(t)}\mid\Theta^{(t)}) - H(\Theta^{(t+1)}\mid\Theta^{(t)}) + H(\Theta^{(t+1)}\mid\Theta^{(t)})
$$

이며 $$\Theta^{(t+1)} = \text{argmax}_{\Theta}Q(\Theta\mid\Theta^{(t)}$$ 임을 고려하면 $$ Q(\Theta^{(t+1)}\mid\Theta^{(t)})-Q(\Theta^{(t)}\mid\Theta^{(t)}) \geq 0$$ 인것은 자명하다. 그렇다면, 우리는 $$ H(\Theta^{(t+1)}\mid\Theta^{(t)}) - H(\Theta^{(t+1)}\mid\Theta^{(t)}) \leq 0$$ 임을 보이기만 하면 된다.

여기서 다시 $$H$$ 함수를 원래의 $$\mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{Z}\mid \mathbf{X},\Theta)\mid \mathbf{X},\Theta^{(t)}]$$ 로 바꾼다.

$$
\begin{align}
H(\Theta^{(t+1)}\mid\Theta^{(t)}) - H(\Theta^{(t+1)}\mid\Theta^{(t)}) &= \mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{Z}\mid \mathbf{X},\Theta^{(t+1)})\mid \mathbf{X},\Theta^{(t)}] - \mathbb{E}_{\mathbf{Z}}[\text{ln}P(\mathbf{Z}\mid \mathbf{X},\Theta^{(t)})\mid \mathbf{X},\Theta^{(t)}] \\ \nonumber
&= \mathbb{E}_{\mathbf{Z}}\left[\frac{\text{ln}P(\mathbf{Z}\mid\mathbf{X},\Theta^{(t+1)})}{\text{ln}P(\mathbf{Z}\mid\mathbf{X},\Theta^{(t)})} \mid \mathbf{X},\Theta^{(t)}\right] \\ \nonumber
&\leq \text{ln}\left[\mathbb{E}_{\mathbf{Z}}\left[\frac{P(\mathbf{Z}\mid\mathbf{X},\Theta^{(t+1)})}{P(\mathbf{Z}\mid \mathbf{X},\Theta^{(t)})}\right]\mathbf{X},\Theta^{(t)} \right]=0 \nonumber
\end{align}
$$

로그함수와 같이 concave한 함수에서는 $$\mathbb{E}[g(X)] \leq g[\mathbb{E}(X)]$$ 라는 Jensen's Inequality 공식을 활용하였다.

#### 예제

혼합 가우시안 분포(Gaussian Mixture)에 대한 EM 알고리즘 적용 예제를 살펴보도록 하자. 이 예제에서는 2개의 가우시안 분포가 혼합된 형태를 가정한다. 또한 각각의 데이터가 가우시안 분포로부터 서로 독립적으로 n개 추출되는 것으로 가정하자.

$$
X_{i} \sim (1-\pi)\mathcal{N}(x_{i}\mid \mu_{1},\sigma^{2}_{1}) + \pi\mathcal{N}(x_{i}\mid\mu_{2},\sigma^{2}_{2})
$$

먼저 이 분포에 대한 likelihood와 log-likelihood를 구한다. 편의상 $$\mu$$와 $$\sigma^{2}$$ 로 parameter를 묶어서 표현하겠다.

$$
\begin{align}
\text{L}(\pi,\mu,\sigma^{2}\mid \mathbf{X}) &= \prod_{i=1}^{n}\left[(1-\pi)\mathcal{N}(x_{i}\mid \mu_{1},\sigma^{2}_{1}) + \pi\mathcal{N}(x_{i}\mid\mu_{2},\sigma^{2}_{2})\right] \\ \nonumber
l(\pi,\mu,\sigma^{2}\mid \mathbf{X}) &= \sum_{i=1}^{n}\text{ln}\left[(1-\pi)\mathcal{N}(x_{i}\mid \mu_{1},\sigma^{2}_{1})+\pi\mathcal{N}(x_{i}\mid\mu_{2},\sigma^{2}_{2}) \right] \nonumber
\end{align}
$$

데이터 추출과정은 다음과 같다고 할 수 있다. 아래와 같은 확률로 각각의 데이터 포인트가 어느 가우시안 분포에서 추출될 것인지가 결정되고

$$
z_{i} =
\begin{cases}
1 \quad \text{with probability} \quad \pi \\
0 \quad \text{with probability} \quad 1-\pi
\end{cases}
$$

이제 $$\mathbf{Z}$$ 가 주어진 상태이기 때문에 데이터 포인트의 추출은 조건부 분포를 따를 것이다.

$$
X_{i}\mid Z_{i} =
\begin{cases}
\mathcal{N}(x_{i}\mid \mu_{1},\sigma^{2}_{1}) \quad \text{if} \quad z_{i}=0 \\
\mathcal{N}(x_{i}\mid \mu_{2},\sigma^{2}_{2}) \quad \text{if} \quad z_{i}=1
\end{cases}
$$

더불어 $$P(\mathbf{X},\mathbf{Z}\ mid \Theta) = P(\mathbf{X}\mid\mathbf{Z},\Theta)P(\mathbf{Z}\mid\Theta)$$ 임을 고려한다면, 다음과 같은 likelihood 식을 구할 수 있다.

$$
\prod_{i=1}^{n}\left\{\pi\mathcal{N}(x_{i}\mid\mu_{1},\sigma^{2}_{1})\right\}^{z_{i}}\left\{(1-\pi)\mathcal{N}(x_{i}\mid\mu_{2},\sigma^{2}_{2})\right\}^{1-z_{i}}
$$

그래서 최종적인 log-likelihood는 다음과 같이 표현할 수 있다.

$$
l(\pi,\mu,\sigma^{2}\mid \mathbf{X},\mathbf{Z}) = \sum_{i=1}^{n}z_{i}\{\text{ln}\pi +\text{ln}\mathcal{N}(x_{i}\mid \mu_{1},\sigma^{2}_{1})\} \sum_{i=1}^{n}(1-z_{i})\{\text{ln}(1-\pi) +\text{ln}\mathcal{N}(x_{i}\mid \mu_{2},\sigma^{2}_{2})\}
$$

이제 E-step으로 들어가자. E-step에서는 잠재변수에 대한 조건부 기대값과 $$Q$$ 함수를 도출해야 한다.

잠재변수에 대한 조건부 기대값은 다음의 과정을 통해 구할 수 있다.

$$
\begin{align}
\mathbb{E}_{\mathbf{Z}}[z_{i}\mid x_{i},\Theta^{(t)}] &= 1\times P(z_{i}=1 \mid x_{i},\Theta^{(t)}) + 0\times P(z_{i}=0 \mid x_{i},\Theta^{(t)}) \\ \nonumber
&= \frac{P(x_{i}\mid z_{i}=1,\Theta^{(t)})P(z_{i}=1\mid \Theta^{(t)})}{P(x_{i}\mid \Theta^{(t)})} \\ \nonumber
&= \frac{P(x_{i}\mid z_{i}=1,\Theta^{(t)})P(z_{i}=1\mid \Theta^{(t)})}{P(x_{i}\mid z_{i}=1,\Theta^{(t)})P(z_{i}=1\mid \Theta^{(t)})+P(x_{i}\mid z_{i}=0,\Theta^{(t)})P(z_{i}=0\mid \Theta^{(t)})} \\ \nonumber
&= \frac{\pi^{(t)}\mathcal{N}(x_{i}\mid \mu_{1}^{(t)},(\sigma^{2}_{1})^{2})}{\pi^{(t)}\mathcal{N}(x_{i}\mid \mu_{1}^{(t)},(\sigma^{2}_{1})^{2}) + (1-\pi^{(t)})\mathcal{N}(x_{i}\mid \mu_{2}^{(t)},(\sigma^{2}_{2})^{2})} \nonumber
\end{align}
$$

즉 $$z_{i}\mid x_{i},\Theta^{(t)}$$는 다음의 베르누이 분포를 따른다고 할 수 있으며 이 확률이 $$z_{i}\mid x_{i},\Theta^{(t)}$$의 기대값 $$\hat{z_{i}}$$라 할 수 있다.

$$
z_{i} \mid x_{i},\Theta^{(t)} \sim \text{Ber}\left( \frac{\pi^{(t)}\mathcal{N}(x_{i}\mid \mu_{1}^{(t)},(\sigma^{2}_{1})^{2})}{\pi^{(t)}\mathcal{N}(x_{i}\mid \mu_{1}^{(t)},(\sigma^{2}_{1})^{2}) + (1-\pi^{(t)})\mathcal{N}(x_{i}\mid \mu_{2}^{(t)},(\sigma^{2}_{2})^{2})} \right)
$$

$$Q$$함수 $$Q(\Theta\mid\Theta^{(t)}$$ 는 $$\mathbb{E}_{\mathbf{Z}}[l(\Theta\mid\mathbf{X},\mathbf{Z})\mid \mathbf{X},\Theta^{(t)}]$$ 이며 다음과 같다.

$$
Q(\Theta\mid\Theta^{(t)}) = \sum_{i=1}^{n}\hat{z_{i}}(\text{ln}\pi+\text{ln}\mathcal{N}(x_{i}\mid \mu_{1},\sigma^{2}_{1})) + \sum_{i=1}^{n}(1-\hat{z_{i}})(\text{ln}(1-\pi)+\text{ln}\mathcal{N}(x_{i}\mid \mu_{2},\sigma^{2}_{2}))
$$

마지막으로 $$Q$$함수를 최대화시키는 5가지 모수에 대한 값을 찾는 것이다. 각 paramter별로 미분하여 최대값이 나오는 값을 찾아 업데이트 한다.

$$
\begin{align}
\frac{\partial Q(\Theta\mid\Theta^{(t)})}{\partial\pi} &= \sum_{i=1}^{n}\hat{z_{i}}\frac{1}{\pi} + \sum_{i=1}^{n}(1-\hat{z_{i}})\left(\frac{-1}{1-\pi}\right)=0 \\ \nonumber
\pi^{(t+1)} = \frac{\sum_{i=1}^{n}\hat{z_{i}}}{n} \nonumber
\end{align}
$$

나머지 parameter에 대해서는 업데이트 되는 값이 다음과 같다.

$$
\begin{align}
\mu_{1}^{(t+1)} &= \frac{\sum_{i=1}^{n}\hat{z_{i}}x_{i}}{\sum_{i=1}^{n}\hat{z_{i}}} \\ \nonumber
\mu_{2}^{(t+1)} &= \frac{\sum_{i=1}^{n}(1-\hat{z_{i}})x_{i}}{\sum_{i=1}^{n}(1-\hat{z_{i}})} \\ \nonumber
(\sigma_{1}^{2})^{(t+1)} &= \frac{\sum_{i=1}^{n}\hat{z_{i}}(x_{i}-\mu_{1})^{2}}{\sum_{i=1}^{n}\hat{z_{i}}} \\ \nonumber
(\sigma_{2}^{2})^{(t+1)} &= \frac{\sum_{i=1}^{n}(1-\hat{z_{i}})(x_{i}-\mu_{2})^{2}}{\sum_{i=1}^{n}(1-\hat{z_{i}})} \nonumber
\end{align}
$$

이 과정을 각각의 parameter에 대해 수렴하는 시점까지(t+1시점 값과 t시점 값의 차이가 일정 수준 이하가 될 때까지) 시행하여 최종적인 값을 구해낼 수 있다.
