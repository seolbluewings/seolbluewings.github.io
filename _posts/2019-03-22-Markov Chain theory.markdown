---
layout: post
title:  "Markov Chain Theory"
date: 2019-03-22 
author: YoungHwan Seol
categories: Bayesian
---

유한의 discrete state space $${\it \chi}$$ 에 대해, 다음과 같은 확률과정(stohastic process)가 $$\{x^{(t)},t=0.1,2,...\}$$ 있다고 가정하자. 

예를 들자면, $$x^{(t)}$$는 t번째 날짜의 날씨라고 할 수 있다.

만약 이 날씨의 확률과정이 다음과 같은 식을 만족한다면, 우리는 이를 Markov Chain이라 부를 수 있다.
\\begin{center}
$$p(x^{(t+1)} | x^{(0)},...,x^{(t-1)},x^{(t)} )=p(x^{(t+1)}|x^{(t)})$$
\\end{center}

다르게 표현하자면,  $$x^{(t+1)}$$ 은 $$x^{(t)}$$가 given일 때, $$\{ x^{(0)},...,x^{(t-1)} \}$$ 과 독립이라 할 수 있고 즉, (t+1)번째 날의 날씨는 t번째 날의 날씨에만 영향을 받으며 그 이전의 날씨에는 영향을 받지 않는다고 볼 수 있다. 이는 또 다르게 표현하자면 다음과 같이 표현가능할 것이다. 

과거와 현재의 상태가 주어졌을 때, 미래 상태의 조건부 확률 분포가 과거의 상태와는 독립적으로 현재 상태에 의해서만 결정된다는걸 의미한다. 

Markov Chain 과정은 다음과 같은 특징을 가지고 있다.

1,  $$P_{ij}^{(t)}=p(x^{(t+1)}=j \| x^{(t)}=i)$$ 를 (one-step) transition probability라 부르며 이 때, $$\sum_{j \in \chi} P_{ij}^{t}=1$$ 를 만족한다. 

2, 만약 $$P_{ij}^{(t)}=P_{ij}$$ 라면, transition probability는 time-homogeneous 하다. 이는 transition probability가 시간 t에 의존하지 않는다는 것을 의미한다. 

3, 유한번의 step을 통해 한가지 상태(A state)에서 또 다른 상태(B state)로 이동할 수 있는 확률이 0이 아니라면, 이 Markov Chain은 irreducible 하다고 표현된다. irreducible한 Markov Chain은 모든 상태끼리 서로 교통 가능하다고 생각할 수 있다.

4, 초기 상태가 A일 때, 다시 A에 도달할 확률이 양수고 이에 해당되는 모든 시간의 최대 공약수(maximum common divider)가 1이면, 이 때 Markov Chain은 aperiodic(비주기적) 이라 한다. 

5, 상태 A를 떠나 '유한' 시간 안에 다시 A로 돌아오는 경우 이를 Markov Chain이 positive recurrent 하다고 표현한다.

6, Aperiodic 하고 Positive recurrent한 상태를 ergodic 하다고 표현한다. Markov Chain의 모든 상태(all states)가 ergodic 하다면, Markov Chain은 positive recurrent, aperiodic, irreducible 할 것이며 Markov Chain이 ergodic 하다고 할 수 있다.

7, $$ \pi(j)=\sum_{i \in \chi} \pi(i)P_{ij} $$를 만족한다면, $$\{\pi(x), x \in \chi \} $$ 는 stationary distribution 이라 부른다. 

8, 만약 stochastic process $$\{x^{(t)},t=0,1,2,...\}$$가 ergodic Markov Chain with stationary distribution $$\pi(x)$$ 라면, 다음과 같은 관계를 얻을 수 있다.
\\begin{center}
$$ \frac{1}{T}\sum_{t=1}^{T}h(x^{(t)}) \rightarrow \int h(x)\pi(x)dx $$ as $$ T \rightarrow \infty $$
\\end{center}
9, Markov Chain이 다음의 조건을 만족하면, time reversible 하다고 말한다. 모든 i,j에 대하여
\\begin{center}
$$ P(x^{(t+1)}=j | x^{(t)}=i)=P(x^{(t)}=j|x^{(t+1)}=i) $$
\\end{center}

이 때 우리는 베이즈 정리(Bayes Theorm)를 사용하여

\\begin{align}
$$ P(x^{(t)}=j | x^{(t+1)}=i) &=  \frac{P(x^{(t)}=j)P(x^{(t+1)}=i|x^{(t)}=j)}{P(x^{(t+1)}=i)} $$ \\
$$ P_{ij} &= \frac{\pi(j)P_{ji}}{\pi(i)} $$
\\end{align}

과 같은 식을 구할 수 있고, 위의 식을 정리하면 detailed balanced condition을 구할 수 있다.

\\begin{center}
$$ \pi(i)P_{ij}=\pi(j)P_{ji} $$
\\end{center}

detailed balanced condition을 i에 대하여 sum 한다면, 다음과 같은 식을 구할 수 있다.

\\begin{center}
$$ \sum_{i \in \chi} \pi(i)P_{ij} = \sum_{i \in \chi} \pi(j)P_{ji} = \pi(j) $$
\\end{center}









