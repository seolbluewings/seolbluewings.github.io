---
layout: post
title:  "Bayesian Network(1)"
date: 2019-07-25
author: YoungHwan Seol
categories: Bayesian
---

베이지안 네트워크(Bayesian Network)는 확률변수 간의 관계를 노드(node)와 링크(link) 혹은 엣지(edge)를 사용해 그래프 모델로 표현하는 것이다. 이후의 논의를 진행하기에 앞서 다음의 용어들에 대하여 정리하고 가도록 한다.

1. 노드(node) : 확률 변수 1개를 1개의 노드로 표현
2. 링크(link) : 엣지(edge)라고 불리기도 하며 이는 확률변수들 사이의 확률적 관계를 나타낸다. 화살표로 표시된다.

베이지안 네트워크(Bayesian Network)는 다음의 특징을 갖는다.

1. 베이지안 네트워크는 방향성 그래프 모델(directed graphical-model)이다.
2. 방향성이란, link가 양방향이 아닌 한쪽 방향으로만 관계가 성립하는 것이다.
3. 방향성 그래프의 중요한 제약은 방향성 link들이 사이클(cycle)을 형성하지 않는다는 것이다. 이를 다르게 표현하면 닫힌 연결(closed path)이라 하고 이런 그래프를 DAG(directed acyclic graph)라 표현한다.

방향성 그래프를 통해 확률변수 사이의 관계를 표현하는 것이 편리하며, 비방향성 그래프 모델(undirected graphical model)의 경우는 확률변수들 사이의 제약을 표현하는데 편리하다. 비방향성 그래프 모델로는 마르코프 랜덤 필드(Markov Random Field)가 있다.

여러 확률변수에 대한 문제를 해결하는 과정에서 결합 확률분포(이하 Joint distribution)을 구해야하는 일이 있는데 확률변수의 수가 많을수록 Joint distribution을 구하기가 어려워지며, 이 때 조건부 독립(이하 Conditional Independence)을 사용하여 문제를 해결할 수 있다.

우리는 Conditional Independence보다 다음의 Marginal Independence에 익숙하다.
$$
\begin{align}
	X &\perp Y \; \text{iff} \\
    P(X) &= P(X\mid Y) = \frac{P(X,Y)}{P(Y)} \\
    P(X,Y) &= P(X)P(Y)
\end{align}
$$

Conditional Independence는 다음과 같이 설명할 수 있다. 3개의 변수 a,b,c가 있다고 하자. 그리고 b,c가 given일 때, a의 조건부 분포(이하 Condtional distribution)은 b에 대해 종속적이지 않다고(independent)하자. 이는 다음과 같이 표현될 것이다.

$$ p(a\mid b,c) = p(a\mid c) $$

이 경우 a는 c가 given인 상태에서 b에 대하여 Conditional Independent 하다고 표현하며 이 경우 다음과 같이 a와 b의 조건부 독립을 표시한다.

$$ a \perp b \mid c$$

c가 given인 상태에서의 a,b의 joint distribution은 다음과 같이 표현될 수 있다.

$$
\begin{align}
	p(a,b \mid c) &= p(a \mid b,c)p(b \mid c) \\
    &= p(a \mid c)p(b \mid c)
\end{align}
$$

따라서 c가 given인 상황에서 a와 b의 joint distribution은 c가 given인 상황에서의 a의 marginal, b의 marginal distribution으로 분해할 수 있다.

#### Bayes Ball Algorithm

방향성 그래프에서 conditional independence 성질에 대한 논의를 위해 몇가지 정형화된 형태의 그래프를 살펴보도록 한다.

- Common Parent (노드가 공통 parent를 가질 때)

![BN](/images/common_parent.png)

위의 예시에 대한 joint distribution은 다음과 같이 적을 수 있다.

$$ p(a,b,c) = p(a \mid c)p(b \mid c)p(c) $$

만약 3가지 a,b,c 중에서 어떠한 변수도 관측되지 않았다면 위의 식은 양변을 c에 대해 marginalized하여 a,b가 독립적인지 확인할 수 있다.

$$ p(a,b)= \sum_{c}p(a \mid c)p(b \mid c)p(c) \neq p(a)p(b) $$

따라서 다음과 같은 결론 $$(a \notperp b | \phi)$$ 을 내릴 수 있다.

![BN](/images/common_parent2.png)

그러나 다음과 같이 이번에는 parent node인 c에 대해서 관측되어 알고 있다고 하자.

이 경우에는 a,b에 대한 조건부 분포를 다음과 같이 표현할 수 있다.

$$
\begin{align}
	p(a,b \mid c) &= \frac{p(a,b,c)}{p(c)} \\
    &= \frac{p(a \mid c)p(b \mid c)p(c)}{p(c)} \\
    &= p(a \mid c)p(b \mid c)
\end{align}
$$

따라서 c에 대해 알고 있으면 a와 b는 conditional independent $$(a \perp b \mid c)$$ 하다.






