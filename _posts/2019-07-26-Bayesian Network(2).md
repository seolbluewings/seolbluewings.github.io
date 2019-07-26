---
layout: post
title:  "Bayesian Network(2)"
date: 2019-07-26
author: YoungHwan Seol
categories: Bayesian
---

앞서 계속 conditional independence를 언급해왔는데 어떤 면에서 condtional independence가 중요한 것일까? 여러가지 확률변수가 있는 상황에서 최선은 각 확률변수들의 joint distribution을 아는 것이나 joint distribution의모수(parameter)의 수가 급격하게 많아지는 문제를 갖는다. 이러한 문제를 해결하기 위해서 서로 독립적인 조건을 알게 되면, 모수의 수를 줄일 수 있다.

일반적으로 K개의 변수를 갖는 joint distribution, $$ p(x_{1},...,x_{K}) $$은 다음과 같이 전개(factorization)이 가능하다.

$$
\begin{align}
	p(x_{1},...,x_{K}) &= p(x_{K}\mid x_{1},...x_{k-1})p(x_{1},...,x_{k-1}) \\
    &= p(x_{K}\mid x_{1},...x_{k-1})p(x_{K-1}\mid x_{1},...,x_{K-2})p(x_{1},...,x_{K-2}) \\
    &= p(x_{K}\mid x_{1},...,x_{K-1})\cdot\cdot\cdot p(x_{2}\mid x_{1})p(x_{1})
\end{align}
$$

다음과 같은 경우의 베이지안 네트워크를 fully connected 되었다고 하며, 이는 임의의 두쌍의 노드가 서로 연결되어 있음을 의미한다.

그러나 아래 그림과 같이 일부 링크가 없는 네트워크가 보다 일반적이다. 아래의 그래프는 fully connected가 아니며, $$x_{1}$$에서 $$x_{2}$$로의 링크, $$x_{3}$$에서 $$x_{7}$$으로 가는 링크가 존재하지 않는다.


![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/conditonal.JPG?raw=true){: width="180" height="232"}

7개의 확률변수 $$p(x_{1},x_{2},...,x_{7})$$은 다음과 같은 형태로 표현될 수 있다.

$$ p(x_{1},x_{2},...,x_{7})=p(x_{1})p(x_{2})p(x_{3})p(x_{4} \mid x_{1},x_{2},x_{3}) $$


베이지안 네트워크(Bayesian Network)에서 사용되는 가장 전형적인 예시를 통해 앞서 논의한 내용들에 대해 다시 한 번 점검하도록 하자.

다음의 그림과 같은 관계가 있다고 하자. 도둑의 침입을 대비하여 알람을 설치하였는데 지진이 발생하는 경우에도 알람이 작동할 가능성이 있다. 도둑이 침입할 확률은 그림에 주어진 바와 같이 0.001이고 지진이 발생할 확률은 0.002다. 알람이 작동할 확률은 $$p(A \mid B,E)$$로 값이 그림과 같이 주어져있다.

알람이 울리면 John과 Mary는 전화를 해주기로 합의하였고 알람이 울렸을 때, John이 전화할 확률 $$p(J \mid A)$$와 알람이 울렸을 때, Mary가 전화할 확률 $$P(M \mid A)$$는 그림과 같다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/burglary_example.PNG?raw=true)

위의 그림을 통해 다음과 같은 질문유형을 해결할 수 있다.

- Mary가 나에게 전화를 한 상황에서 정말로 도둑이 들었을 확률?




따라서 다음과 같은 결론 $$(a  \not\!\perp\!\!\!\perp b \mid \phi)$$ 을 내릴 수 있다.



앞선 경우와 마찬가지로 이번에도 아래 그림과 같이 c에 대한 정보가 given되었다고 하자. 이 경우에는 a,b에 대한 조건부 분포를 다음과 같이 표현할 수 있다.



![BayesianNet](C:/seolbluewings.github.io/assets/images/cascading2.png)



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



![BayesianNet](C:/seolbluewings.github.io/assets/images/VS1.png)



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



마지막으로 다음의 2가지 사항에 대해 추가적으로 논의하고 Part1을 마무리 짓고자 한다.



- D-seperation

- Markov-Blanket



D-seperation이란 directly seperated라는 표현을 줄인 것이다. "given Y일 때, X가 Z로부터 d-seperated 되어있다라"는 말은 다음의 표기와 동일한 의미를 지닌다. 사실 계속 봐왔던 표현이다.



$$ X \perp\!\!\!\perp Z \mid Y $$



Markov-Blanket을 설명하는 그림은 아래와 같다.



![BayesianNet](C://seolbluewings.github.io/assets/images/Markov_blanket.png)



다음의 베이지안 네트워크에서 확률변수 A가 있다고 하자. 이 A가 베이지안 네트워크에 존재하는 다른 확률변수들과 모두 Conditional Independent하게 만드는 조건(condition)을 찾아보려고 한다. 그림에는 총 13개의 확률변수가 있고 A가 그중 하나이므로 다른 12개의 확률변수들과 conditional independent하게 만드려는 것이다.



이 때 그림의 원안에 표시된 것처럼 6개의 확률변수에 대해 알게 되면, 나머지 6개 확률변수들과 A는 conditonal independent하게 되고 이를 Markov Blanket이라 표현한다.



B를 원밖에 있는 확률변수들의 집합이라고 하면, 다음과 같이 표현할 수 있다. $$ p(A \mid blanket,B) = p(A \mid blanket) $$ 즉, $$ A \perp\!\!\!\perp B \mid blanket $$.



blanket은 다음의 3가지 파트로 구성된다.

- cascading에 의한 dependence를 막기위하여 A의 parent node가 given이어야 하고 (위의 2개)

- cascading에 의한 dependence를 막기위하여 A의 child node가 given이어야 하며 (아래 2개)

- V-struture에 의한 dependence를 막기 위하여 A의 child node의 다른 parent node가 given이어야 한다. (좌우 2개)

- child node의 parent node까지 given이어야 하는 이유는 child node를 알게되는 순간, V-structure의 특성으로 인해 cascading의 가능성이 생기며, 이를 막기 위함이다.



다음과 같은 6가지 확률변수에 대해 conditional 할 수 있다면, A는 Markov-Blanket 바깥에 위치한 확률변수들과는 condtional independent하다.





