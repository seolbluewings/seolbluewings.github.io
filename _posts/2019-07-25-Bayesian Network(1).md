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
	X &\perp\!\!\!\perp Y \; \text{iff} \\
    P(X) &= P(X\mid Y) = \frac{P(X,Y)}{P(Y)} \\
    P(X,Y) &= P(X)P(Y)
\end{align}
$$

Conditional Independence는 다음과 같이 설명할 수 있다. 3개의 변수 a,b,c가 있다고 하자. 그리고 b,c가 given일 때, a의 조건부 분포(이하 Condtional distribution)은 b에 대해 종속적이지 않다고(independent)하자. 이는 다음과 같이 표현될 것이다.

$$ p(a\mid b,c) = p(a\mid c) $$

이 경우 a는 c가 given인 상태에서 b에 대하여 Conditional Independent 하다고 표현하며 이 경우 다음과 같이 a와 b의 조건부 독립을 표시한다.

$$ a \perp\!\!\!\perp b \mid c$$

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

![BayesianNet](C://seolbluewings.github.io/assets/images/common_parent.png)

위의 예시에 대한 joint distribution은 다음과 같이 적을 수 있다.

$$ p(a,b,c) = p(a \mid c)p(b \mid c)p(c) $$

만약 3가지 a,b,c 중에서 어떠한 변수도 관측되지 않았다면 위의 식은 양변을 c에 대해 marginalized하여 a,b가 독립적인지 확인할 수 있다.

$$ p(a,b)= \sum_{c}p(a \mid c)p(b \mid c)p(c) \neq p(a)p(b) $$

따라서 다음과 같은 결론 $$(a  \not\!\perp\!\!\!\perp b \mid \phi)$$ 을 내릴 수 있다.

![BayesianNet](C://seolbluewings.github.io/assets/images/common_parent2.png)

그러나 다음과 같이 이번에는 parent node인 c에 대해서 관측되어 알고 있다고 하자. 이 경우에는 a,b에 대한 조건부 분포를 다음과 같이 표현할 수 있다.

$$
\begin{align}
	p(a,b \mid c) &= \frac{p(a,b,c)}{p(c)} \\
    &= \frac{p(a \mid c)p(b \mid c)p(c)}{p(c)} \\
    &= p(a \mid c)p(b \mid c)
\end{align}
$$

따라서 c에 대해 알고 있으면 a와 b는 conditional independent $$(a \perp\!\!\!\perp b \mid c)$$ 하다.

- Cascading (선형적 관계)

![BayesianNet](C://seolbluewings.github.io/assets/images/cascading1.png)

다음과 같이 각 노드가 선형적인 관계를 가지는 경우, 이 관계를 Cascading이라 부른다. 이 때 a,b,c의 joint distribution은 다음과 같이 $$p(a,b,c) = p(b \mid c)p(c \mid a)p(a) $$로 표현할 수 있다.

어떠한 변수도 관측되지 않았다고 할 때, c에 대해 a,b,c의 joint distribution을 marginalized하면, a와 b가 서로 독립인지 알 수 있다.

$$
\begin{align}
	p(a,b) &= \sum_{c}p(a,b,c) = sum_{c}p(b \mid c)p(c \mid a)p(a) \\
    &= p(a) \sum_{c} p(c \mid a)p(b \mid c) \\
    &= p(a)p(b \mid a) \neq p(a)p(b)
\end{align}
$$

따라서 다음과 같은 결론 $$(a  \not\!\perp\!\!\!\perp b \mid \phi)$$ 을 내릴 수 있다.

앞선 경우와 마찬가지로 이번에도 아래 그림과 같이 c에 대한 정보가 given되었다고 하자. 이 경우에는 a,b에 대한 조건부 분포를 다음과 같이 표현할 수 있다.

![BayesianNet](C://seolbluewings.github.io/assets/images/cascading2.png)

$$
\begin{align}
	p(a,b \mid c) &= \frac{p(a,b,c)}{p(c)} \\
    &= \frac{p(a)p(c \mid a)p(b \mid c)}{p(c)} \\
    p(c \mid a) &= \frac{p(a \mid c)p(c)}{p(a)} \quad \text{이므로} \\
    &= p(a \mid c)p(b \mid c)
\end{align}
$$

따라서 c에 대해 알고 있으면 a와 b는 conditional independent $$(a \perp\!\!\!\perp b \mid c)$$ 하다.

- V-structure (Common Child)

V-structure는 아래의 그림과 같다.

![BayesianNet](C://seolbluewings.github.io/assets/images/VS1.png)

V-structure는 앞서 소개한 2가지 구조와 다른 특징을 갖는다. 앞서 소개한 Common Parent와 Cascading의 경우에는 a,b,c 중 어떤 1개의 정보가 주어지지 않은 상황에서 서로 dependence하고 하나의 정보가 주어지면 independent한 관계를 갖는다.

반면 V-structure의 경우 a,b,c 중 어떠한 노드의 정보를 알지 못할 때, 서로 independent하며 오히려 a,b,c 중 어느 하나의 정보를 알게되는 순간 서로 dependence한 구조를 갖는다.

현재의 구조에서 a,b,c에 대한 joint distribution은 다음과 같이 표현될 수 있다.

$$ p(a,b,c) = p(a)p(b)p(c \mid a,b) $$

a,b에 대한 joint distribution은 위의 식을 c에 대하여 marginalized하여 얻을 수 있다.

$$
\begin{align}
	p(a,b) &= \sum_{c}p(a,b,c) \\
    &= \sum_{c}p(a)p(b)p(c \mid a,b) \\
    &= p(a)p(b)\sum_{c}p(c \mid a,b) = p(a)p(b)
\end{align}
$$

먼저 언급한 바와 같이 a,b,c 중 어떠한 것도 given이 아닌 상태에서 a와 b는 서로 독립적이다.

반면 아래의 그림과 같이 c가 given된 상황에서 어떻게 달라지는지 알아보자.

![BayesianNet](C://seolbluewings.github.io/assets/images/VS2.png)

$$
\begin{align}
	p(a,b \mid c) &= \frac{p(a,b,c)}{p(c)} \\
    &= \frac{p(a)p(b)p(c \mid a,b)}{p(c)} \\
    &\neq p(a \mid c)p(b \mid c)
\end{align}
$$

이 경우에는 a와 b가 conditional independent하지 않는다. $$ a \not\!\perp\!\!\!\perp b \mid \phi $$ 먼저 언급한 바와 같이 a,b,c 중 하나의 정보가 주어질 때, 오히려 독립성이 깨진다.

앞선(Common parent, Cascading)경우에는 특정 정보를 알면 두 변수 사이의 독립성이 생겼으나 V-Structure의 경우에는 특정 정보를 알면 두 변수 사이의 독립성이 사라지게 되었다.

#### Bayes Ball Example

![BayesianNet](C://seolbluewings.github.io/assets/images/bayes_ball.png)

다음 4가지 경우에 대해 체크해보도록 한다.

- $$ X_{1} \perp\!\!\!\perp X_{4} \mid \{X_{2}\} $$

$$X_{2}$$가 given이면, $$X_{1}$$와 $$X_{4}$$의 관계는 Cascading에서 소개한 케이스와 같다. 따라서 두 변수는 서로 독립이다.

- $$ X_{2} \perp\!\!\!\perp X_{5} \mid \{X_{1}\} $$

주어진 네트워크에서 $$X_{2}$$와 $$X_{5}$$ 그리고 $$X_{6}$$ 부분을 확인해보자. 이는 V-Structure이다. V-structure를 구성하는 변수들에 대해 어떠한 것도 given이 아니므로 $$X_{2}$$ 와 $$X_{5}$$는 서로 독립이다.

- $$ X_{1} \perp\!\!\!\perp X_{6} \mid \{X_{2},X_{3}\} $$

$$X_{1}$$ 에서 $$X_{6}$$로 가는 길은 각 Cascading 형태를 가지는데 $$X_{2}$$와 $$X_{3}$$가 모두 given이므로 $$X_{1}$$과 $$X_{6}$$는 서로 독립이다.

- $$ X_{2} \perp\!\!\!\perp X_{3} \mid \{X_{1},X_{6}\} $$

$$X_{1}$$가 given이면 좌측의 parent node가 given이 된 경우다. 경로가 이 하나만 존재한다면, 독립적이겠지만, $$X_{6}$$가 given인 상황을 고려할 필요가 있다. $$X_{6}$$가 given일 때 우측의 V-structure가 독립성을 잃어버리며 이후의 $$X_{3}$$부터 $$X_{6}$$의 관계는 Cascading 형태이고 결국 이 경우에는 $$X_{2}$$와 $$X_{3}$$는 독립이 아니게 된다.

지금까지의 논의를 거치면 Bayes Ball Algorithm에 대해서는 어느 정도 익숙해졌을 것이다.

마지막으로 2가지 사항에 대해 추가적으로 논의하고 Part1을 마무리 짓고자 한다.





