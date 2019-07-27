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


![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/conditonal.JPG?raw=true){:height="200px" width="100px"}

7개의 확률변수 $$p(x_{1},x_{2},...,x_{7})$$은 다음과 같은 형태로 표현될 수 있다.

$$ p(x_{1},x_{2},...,x_{7})=p(x_{1})p(x_{2})p(x_{3})p(x_{4} \mid x_{1},x_{2},x_{3})p(x_{5}\mid x_{1},x_{3})p(x_{6} \mid x_{4})p(x_{7} \mid x_{4},x_{5}) $$

이를 일반화한 표현은 다음과 같으며, $$pa_{k}$$는 변수 $$x_{k}$$의 parent node 집합을 의미한다.

$$ p(\mathbf{x}) = \prod_{k=1}^{K}p(x_{k}\mid pa_{k})$$

conditional probability를 이용하여 우리는 몇가지 evidence가 주어진 상황에서 관측되지 않은 hidden variable의 conditional probability를 알아낼 수 있다. 베이지안 네트워크에서 확인할 수 있는 전체 확률변수를 $$\mathbf{X}$$라고 하자. 이 때, 우리가 관찰한 확률변수는 $$X_{obs}$$라 표기하고 관측하지 못한 변수들에 대해서는 $$X_{H}$$이라 표현하자. 또한 $$X_{H}$$은 우리가 관심을 가지고 있는(혹은 conditional probability를 알아내고자 하는) 변수 Y와 그마저도 관심이 없는 변수 Z로 나뉠 수 있다. $$ X_{H} = \{Y,Z\} $$

현재 알지 못하고 있으나 관심있는 변수 Y에 대한 conditional distribution은 다음과 같이 구할 수 있다. 우리는 conditional independence를 활용해 joint distribution을 얻을 수 있으므로 conditional distribution을 joint distribution을 향한 방향으로 풀어갈 필요가 있다.

$$
\begin{align}
	p(Y \mid X_{obs}) &= \sum_{z} p(Y,Z=z \mid  X_{obs}) \\
    &= \sum_{z} \frac{p(Y,Z,X_{obs})}{p(X_{obs})} \\
    &= \sum_{z} \frac{p(Y,Z,X_{obs})}{\sum_{y,z}p(Y=y,Z=z,X_{obs})}
\end{align}
$$

이렇게 구한 condtional probability를 활용하여 우리는 관심있는 Hidden Variable Y에 대해 가장 발생 가능한 사건(most probable assignment)을 얻을 수 있다.

베이지안 네트워크(Bayesian Network)에서 사용되는 가장 전형적인 예시를 통해 앞서 논의한 내용들에 대해 다시 한 번 점검하도록 하자.

다음의 그림과 같은 관계가 있다고 하자. 도둑의 침입을 대비하여 알람을 설치하였는데 지진이 발생하는 경우에도 알람이 작동할 가능성이 있다. 도둑이 침입할 확률은 그림에 주어진 바와 같이 0.001이고 지진이 발생할 확률은 0.002다. 알람이 작동할 확률은 $$p(A \mid B,E)$$로 값이 그림과 같이 주어져있다.

알람이 울리면 John과 Mary는 전화를 해주기로 합의하였고 알람이 울렸을 때, John이 전화할 확률 $$p(J \mid A)$$와 알람이 울렸을 때, Mary가 전화할 확률 $$P(M \mid A)$$는 그림과 같다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/burglary_example.PNG?raw=true)

앞서 언급한 바처럼, 우리는 먼저 full joint probability를 계산하고 이제 marginalized하여 일부 변수에 대한 partial joint probability를 얻는다. 이후 관심변수에 대한 conditional probability를 계산하여 값을 결정한다.

다음의 partial joint probability는 어떻게 구할 수 있을까?

$$ p(A=True, B=True, MC= True)$$

먼저 full joint probability를 구하고 이를 marginalized한다.

$$
\begin{align}
	p(A,B,MC) &= \sum_{JC}\sum_{E} p(A,B,E,MC,JC) \\
    &= \sum_{JC}\sum_{E} p(JC \mid A)p(MC \mid A)p(A \mid B,E)p(E)p(B) \\
    &= p(B)p(MC \mid a) \sum_{JC}p(JC \mid A) \sum_{E}p(A \mid B,E)p(E)
\end{align}
$$

따라서 $$MC=True, A=True$$일 때, 도둑이 들었을 확률은 다음과 같이 구할 수 있을 것이다.

$$
\begin{align}
	p(B \mid A,MC) &= \frac{p(A,B,MC)}{p(A,MC)} \\
    &= \frac{p(B)p(MC \mid a) \sum_{JC}p(JC \mid A) \sum_{E}p(A \mid B,E)p(E)}{p(MC\mid A)p(A)}
\end{align}
$$





