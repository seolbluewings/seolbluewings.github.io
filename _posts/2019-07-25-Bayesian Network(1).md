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

앞서 계속 conditional independence를 언급해왔는데 어떤 면에서 condtional independence가 중요한 것일까? 여러가지 확률변수가 있는 상황에서 최선은 각 확률변수들의 joint distribution을 아는 것이나 joint distribution의모수(parameter)의 수가 급격하게 많아지는 문제를 갖는다. 이러한 문제를 해결하기 위해서 서로 독립적인 조건을 알게 되면, 모수의 수를 줄일 수 있다.

#### Bayes Ball Algorithm

방향성 그래프에서 conditional independence 성질에 대한 논의를 위해 몇가지 정형화된 형태의 그래프를 살펴보도록 한다.

- Common Parent (노드가 공통 parent를 가질 때)

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/common_parent.PNG?raw=true){:width="30%" height="30%"}{: .center}

위의 예시에 대한 joint distribution은 다음과 같이 적을 수 있다.

$$ p(a,b,c) = p(a \mid c)p(b \mid c)p(c) $$

만약 3가지 a,b,c 중에서 어떠한 변수도 관측되지 않았다면 위의 식은 양변을 c에 대해 marginalized하여 a,b가 독립적인지 확인할 수 있다.

$$ p(a,b)= \sum_{c}p(a \mid c)p(b \mid c)p(c) \neq p(a)p(b) $$

따라서 다음과 같은 결론 $$(a  \not\!\perp\!\!\!\perp b \mid \phi)$$ 을 내릴 수 있다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/common_parent2.PNG?raw=true){:width="30%" height="30%"}{: .center}

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

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/cascading1.PNG?raw=true){:width="30%" height="30%"}{: .center}

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

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/cascading2.PNG?raw=true){:width="30%" height="30%"}{: .center}

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

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/VS1.PNG?raw=true){:width="30%" height="30%"}{: .center}

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

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/VS2.PNG?raw=true){:width="30%" height="30%"}{: .center}

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

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayes_ball.PNG?raw=true)

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

2가지 개념을 더 소개하고자 한다.

- D-seperation
- Markov-Blanket

D-seperation이란 directly seperated라는 표현을 줄인 것이다. "given Y일 때, X가 Z로부터 d-seperated 되어있다라"는 말은 다음의 표기와 동일한 의미를 지닌다. 사실 계속 봐왔던 표현이다.

$$ X \perp\!\!\!\perp Z \mid Y $$

Markov-Blanket을 설명하는 그림은 아래와 같다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Markov_blanket.png?raw=true){:width="30%" height="30%"}{: .center}

다음의 베이지안 네트워크에서 확률변수 A가 있다고 하자. 이 A가 베이지안 네트워크에 존재하는 다른 확률변수들과 모두 Conditional Independent하게 만드는 조건(condition)을 찾아보려고 한다. 그림에는 총 13개의 확률변수가 있고 A가 그중 하나이므로 다른 12개의 확률변수들과 conditional independent하게 만드려는 것이다.

이 때 그림의 원안에 표시된 것처럼 6개의 확률변수에 대해 알게 되면, 나머지 6개 확률변수들과 A는 conditonal independent하게 되고 이를 Markov Blanket이라 표현한다.

B를 원밖에 있는 확률변수들의 집합이라고 하면, 다음과 같이 표현할 수 있다. $$ p(A \mid blanket,B) = p(A \mid blanket) $$ 즉, $$ A \perp\!\!\!\perp B \mid blanket $$.

blanket은 다음의 3가지 파트로 구성된다.

- cascading에 의한 dependence를 막기위하여 A의 parent node가 given이어야 하고 (위의 2개)
- cascading에 의한 dependence를 막기위하여 A의 child node가 given이어야 하며 (아래 2개)
- V-struture에 의한 dependence를 막기 위하여 A의 child node의 다른 parent node가 given이어야 한다. (좌우 2개)
- child node의 parent node까지 given이어야 하는 이유는 child node를 알게되는 순간, V-structure의 특성으로 인해 cascading의 가능성이 생기며, 이를 막기 위함이다.

다음과 같은 6가지 확률변수에 대해 conditional 할 수 있다면, A는 Markov-Blanket 바깥에 위치한 확률변수들과는 condtional independent하다.

#### Example of directed Graphical models

- Bayesian Linear Regression

타깃 변수$$(\mathbf{t})$$는 다음과 같이 $$\mathbf{t}=(t_{1},...,t_{N})^{T}$$이며 회귀계수는 $$\mathbf{w}$$로 표기한다. 입력 데이터는 $$\mathbf{x} = (x_{1},...,x_{N})^{T}$$이며 오차항은 $$\mathcal{N}(0,\sigma^{2})$$를 따른다. 그래프 모델은 아래의 그림과 같이 표현할 수 있으며 $$\mathbf{t}$$와 $$\mathbf{w}$$의 joint probability는 아래와 같이 구할 수 있다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesian_linearnet.JPG?raw=true){:width="30%" height="30%"}{: .center}

$$ p(\mathbf{t},\mathbf{w}) = p(\mathbf{w})\prod_{n=1}^{N}p(t_{i} \mid \mathbf{w}) $$

위의 그림과 같이 $$t_{1}$$부터 $$t_{N}$$까지 모두 표기하는 방식은 깔끔하지 못하다. 여기서 plate라는 개념을 소개하는데, plate는 보통 하나의 그룹으로 표현되는 노드들을 박스 형태로 표기하는 방식이다. 따라서 N개의 $$t_{}$$들은 다음과 같이 하나의 박스로 표기가 가능하다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesian_linearnet2.JPG?raw=true){:width="30%" height="30%"}{: .center}

여기에 이미 값이 주어진 것으로 간주하는 변수들에 대한 정보를 추가할 수 있다. 이 경우에는 다른 노드들처럼 큰 원을 그리는 것이 아니라 작은 원(혹은 점)의 형태로 표기한다. $$\alpha$$는 베이지안 회귀분석에서 $$\mathbf{w}$$에 대해 $$\mathcal{N}(0,\alpha^{-1}I)$$라는 prior가 주어진 것을 의미한다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesian_linearnet3.JPG?raw=true){:width="30%" height="30%"}{: .center}




