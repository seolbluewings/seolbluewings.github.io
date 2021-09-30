---
layout: post
title:  "Concept of Hidden Markov Model 2"
date: 2021-09-17
author: seolbluewings
categories: Statistics
---

[작성중...]

앞선 포스팅에서는 HMM 모델에서 추정하게 될 항목들을 계산하기 위해서 EM 알고리즘을 활용하게 되며, E Step에서 정의한 notation을 계산하는 방법에 대해 추가적인 논의가 필요하다고 소개하였다.

$$\gamma,\xi$$에 대한 정의를 했는데 이를 계산하기 위한 대표적인 방법으로는 foward-backward 알고리즘이란 것이 있다. $$\gamma, \xi$$ 모두 latent variable $$\mathbf{z}$$에 대한 분포를 표현하는 식이므로 모든 n에 대한 $$p(x_{n}\vert z_{n},\theta^{(t)})$$ 에 대한 정보가 필요할 것이다. 다만 EM 알고리즘의 각 Step에서 $$\theta^{(t)}$$ 값은 고정이 되므로 향후 수식 표기에서는 제외하기로 한다.

HMM 모델이 아래와 같이 주어졌다고 하였을 때, 다음의 조건부 독립성을 적극적으로 이용하여 추정해야할 $$\gamma, \xi$$를 구할 수 있게 된다.

$$ p(\mathbf{x}\vert z_{n}) = p(x_{1},...,x_{n}\vert z_{n})p(x_{n+1},...,x_{N}\vert z_{n})  $$

![HMM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/HMM1.PNG?raw=true){:width="70%" height="70%"}{: .aligncenter}

우선 latent variable $$z_{n}$$ 이 K개의 class 중 하나에 할당될 확률을 의미하는 $$\gamma(z_{n}=k) = p(z_{n}=k \vert \mathbf{x}) $$ 에 대해 계산 해보자.

$$
\begin{align}
\gamma(z_{n}) &= p(z_{n}\vert \mathbf{x}) = \frac{p(\mathbf{x}\vert z_{n})p(z_{n})}{p(\mathbf{x})} \nonumber \\
&= \frac{p(x_{1},...,x_{n}\vert z_{n})p(x_{n+1},...,x_{N}\vert z_{n})}{p(\mathbf{x})} \nonumber \\
&= \frac{p(x_{1},...,x_{n},z_{n})p(x_{n+1},...,x_{N}\vert z_{n})}{p(\mathbf{x})} = \frac{\alpha(z_{n})\beta(z_{n})}{p(\mathbf{x})}
\end{align}
$$

$$\alpha(z_{n}),\beta(z_{n})$$ 이라는 2가지 notation을 추가적으로 도입하였으니 이를 계산할 수 있는 방법에 대해서도 고민이 필요하다.

우선 $$\alpha(z_{n})$$ 을 계산하는 과정은 다음과 같다.

$$
\begin{align}
\alpha(z_{n}) &= p(x_{1},...,x_{n},z_{n}) = p(x_{1},...,x_{n}\vert z_{n})p(z_{n}) \nonumber \\
&= p(x_{n}\vert z_{n})p(x_{1},...,x_{n-1}\vert z_{n})p(z_{n}) \nonumber \\
&= p(x_{n}\vert z_{n})p(x_{1},...,x_{n-1},z_{n}) \nonumber \\
&= p(x_{n}\vert z_{n})\sum_{z_{n-1}}p(x_{1},...,x_{n-1},z_{n-1},z_{n}) \nonumber \\
&= p(x_{n}\vert z_{n})\sum_{z_{n-1}}p(x_{1},...,x_{n-1},z_{n} \vert z_{n-1})p(z_{n-1}) \nonumber
&= p(x_{n}\vert z_{n})\sum_{z_{n-1}}p(x_{1},...,x_{n-1}\vert z_{n-1})p(z_{n}\vert z_{n-1})p(z_{n-1}) \nonumber
&= p(x_{n}\vert z_{n})\sum_{z_{n-1}}p(x_{1},...,x_{n-1},z_{n-1})p(z_{n}\vert z_{n-1})
\end{align}
$$

식의 마지막 형태를 보면, $$\alpha(z_{n-1})$$로 표현 가능한 부분이 보이게 된다. 따라서 $$\alpha(z_{n})$$에서는 다음과 같은 재귀적인 수식 표현이 가능하다. 이는 이전 상태의 $$\alpha$$ 값을 통해 다음 $$\alpha$$의 값을 구하는 forward 형식이다.

$$ \alpha(z_{n}) = p(x_{n}\vert z_{n})\sum_{z_{n-1}}\alpha(z_{n-1})p(z_{n}\vert z_{n-1}) $$

그렇다면 모든 것의 출발이 되는 포인트는 $$\alpha(z_{1})$$이 되겠다. 이 $$\alpha(z_{1})$$는 다음과 같이 계산할 수 있다.

$$ \alpha(z_{1}) = p(x_{1},z_{1}) = p(x_{1}\vert z_{1})p(z_{1}) = \prod_{k=1}^{K}\left\{\pi_{k}p(x_{1}\vert \phi_{k})  \right\}^{I(z_{1}=k)} $$

Chain 구조에서 첫번째 latent variable node부터 시작하여 각각의 latent variable node에 대해 $$\alpha(z_{n})$$ 계산이 가능해진다.

$$\beta(z_{n})$$ 에 대해서도 마찬가지다. 대신 $$\beta(z_{n})$$에 대해서는 $$\beta(z_{n+1})$$을 바탕으로 $$\beta(z_{n})$$을 계산하는 backward 방식이다.

$$
\begin{align}
\beta(z_{n}) &= p(x_{n+1},...,x_{N}\vert z_{n}) \nonumber \\
&= \sum_{z_{n+1}}p(x_{n+1},...,x_{N},z_{n+1}\vert z_{n}) \nonumber \\
&= \sum_{z_{n+1}}p(x_{n+1},...,x_{N}\vert z_{n+1})p(z_{n+1}\vert z_{n}) \nonumber \\
&= \sum_{z_{n+1}}p(x_{n+2},...,x_{N}\vert z_{n+1})p(x_{n+1}\vert z_{n+1})p(z_{n+1}\vert z_{n}) \nonumber \\
&= \sum_{z_{n+1}}\beta(z_{n+1})p(x_{n+1}\vert z_{n+1})p(z_{n+1}\vert z_{n}) \nonumber
\end{align}
$$

forward 방식이 최초 latent variable $$z_{1}$$에서 출발했다면, backward 방식은 최종 latent variable $$z_{N}$$으로부터 출발하게 된다.

$$n=N$$으로 설정하면 $$\beta(z_{N})$$에 대해서는 구하기 용이하다. 아래의 식을 고려한다면, $$\beta(z_{N})$$은 K개의 1로 구성된 벡터라는 사실을 알 수 있다.

$$ p(z_{N}\vert \mathbf{x}) = \frac{p(\mathbf{x},z_{N})\beta(z_{N})}{p(\mathbf{x})} $$

그리고 $$\beta(z_{N})$$은 K개의 1로 구성된 벡터라는 사실을 통해서 $$\mathbf{x}$$ 에 대한 likelihood 역시 간소화시킬 수 있다.

$$ p(\mathbf{x}) = \sum_{z_{n}}\alpha(z_{n})\beta(z_{n}) = \sum_{z_{N}}\alpha(z_{N}) $$




[to be continued...]

#### 참조 문헌
1. [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
2. [인공지능 및 기계학습 개론II](https://www.edwith.org/machinelearning2__17/lecture/10868?isDesc=false)
