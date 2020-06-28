---
layout: post
title:  "EM 알고리즘(EM Algorithm)"
date: 2020-06-27
author: YoungHwan Seol
categories: Statistics
---

EM알고리즘은 잠재변수(latent variable)를 갖는 확률 모델의 MLE 값을 찾는 과정에서 활용되는 기법이다. 잠재변수를 활용하는 가우시안 혼합 모델에 관한 추정에 자주 활용되는 기법이기도 하다.

우선 관측변수 $$\mathbf{X}=\{x_{1},...x_{N}\}$$ 이라 하자. 잠재변수 $$\mathbf{Z}=\{z_{1},z_{2},...z_{n}\}$$ 는 이 논의를 진행하는 과정에서 이산형 변수라고 가정하자. 만약 $$\mathbf{Z}$$가 연속형이라면, 아래의 과정에서 합표기가 되어있는 것을 적분으로 바꾸면 된다. 그리고 모델에 활용되는 Parameter들을 $$\Theta$$ 라고 표현하자.

우리의 목표는 $$P(\mathbf{X} \mid \Theta)$$ 혹은 $$l(\Theta \mid \mathbf{X})$$ 를 최대화시키는 $$\Theta$$ 값을 구하는 것이다. 이 $$P(\mathbf{X} \mid \Theta)$$에 대한 log likelihood는 다음과 같이 표현할 수 있다.

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
&\leq \text{ln}\left[\mathbb{E}_{\mathbf{Z}}\left[\frac{P(\mathbf{Z}\mid\mathbf{X},\Theta^{(t+1)})}{P(\mathbf{Z}\mid \mathbf{X},\Theta^{(t)})}\right]\mathbf{X},\Theta^{(t)} \right] \nonumber
\end{align}
$$

로그함수와 같이 concave한 함수에서는 $$\mathbb{E}[g(X)] \leq g[\mathbb{E}(X)]$$ 라는 Jensen's Inequality 공식을 활용하였다.

#### 예제




