---
layout: post
title:  "Bayesian Network(3)"
date: 2019-07-28
author: YoungHwan Seol
categories: Bayesian
---

분류에 사용되는 모델은 크게 2가지로 나눌 수 있다.

- 생성 모델(Generative Model)
- 판별 모델(Discriminative model)

로지스틱 회귀분석과 같이 우리가 잘 알고있는 분류 문제 해결방법은 판별 모델(discriminative model)이며, 이러한 판별 모델들은 데이터 X와 label Y가 주어진 상황에서 sample data를 생성하지 않고 직접 $$p(y \mid x)$$를 구하여 클래스 분류에 집중한다.

반면 생성 모델은 $$p(x\mid y)$$와 $$p(y)$$ 의 분포를 학습하고 이를 바탕으로 $$p(y \mid x) \propto p(y)p(x \mid y)$$를 계산해 sample data set을 생성할 수 있다. likelihood와 posterior probability를 이용하여 클래스를 결정짓는 decision boundary를 생성하는 것이다.

따라서 우리는 어떤 확률 분포로부터 임의의 sample을 만들어내는 방법을 알아야 하며, 지금부터 방향성 그래프 모델과 관련이 있는 ancestral sampling이라는 방법에 대해 소개해보고자 한다.






![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/conditonal.JPG?raw=true){:width="30%" height="30%"}{: .center}

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

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/burglary_example.PNG?raw=true){: .center}

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


#### Example of directed Graphical models

- Bayesian Linear Regression

타깃 변수$$(\mathbf{t})$$는 다음과 같이 $$\mathbf{t}=(t_{1},...,t_{N})^{T}$$이며 회귀계수는 $$\mathbf{w}$$로 표기한다. 입력 데이터는 $$\mathbf{x} = (x_{1},...,x_{N})^{T}$$이며 오차항은 $$\mathcal{N}(0,\sigma^{2})$$를 따른다. 그래프 모델은 아래의 그림과 같이 표현할 수 있으며 $$\mathbf{t}$$와 $$\mathbf{w}$$의 joint probability는 아래와 같이 구할 수 있다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesian_linearnet.JPG?raw=true){:width="30%" height="30%"}{: .center}

$$ p(\mathbf{t},\mathbf{w}) = p(\mathbf{w})\prod_{n=1}^{N}p(t_{i} \mid \mathbf{w}) $$

위의 그림과 같이 $$t_{1}$$부터 $$t_{N}$$까지 모두 표기하는 방식은 깔끔하지 못하다. 여기서 plate라는 개념을 소개하는데, plate는 보통 하나의 그룹으로 표현되는 노드들을 박스 형태로 표기하는 방식이다. 따라서 N개의 $$t_{}$$들은 다음과 같이 하나의 박스로 표기가 가능하다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesian_linearnet2.JPG?raw=true){:width="30%" height="30%"}{: .center}

여기에 이미 값이 주어진 것으로 간주하는 변수들에 대한 정보를 추가할 수 있다. 이 경우에는 다른 노드들처럼 큰 원을 그리는 것이 아니라 작은 원(혹은 점)의 형태로 표기한다. $$\alpha$$는 베이지안 회귀분석에서 $$\mathbf{w}$$에 대해 $$\mathcal{N}(0,\alpha^{-1}I)$$라는 prior가 주어진 것을 의미한다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesian_linearnet3.JPG?raw=true){:width="30%" height="30%"}{: .center}

- Naive Bayes

앞선 글에서 먼저 소개했던 Common Parent 구조는 Naive Bayes의 가장 전형적인 예시라고 할 수 있다. Naive Bayes는 conditional probability를 이용하여 k개의 가능한 확률적 결과(분류)를 다음과 같이 할당한다.

$$ p(C_{k} \mid x_{1},...,x_{n}) = p(C_{k}\mid \mathbf{x}) = \frac{p(C_{k})p(\mathbf{x}\mid C_{k})}{p(\mathbf{x})} $$

분자 부분은 factorization하여 다음과 같이 표현할 수 있다.

$$
\begin{align}
	p(C_{k}, \mathbf{x}) &= p(C_{k})p(x_{1},...,x_{k} \mid C_{k}) \\
	&= p(C_{k})p(x_{1} \mid C_{k})p(x_{2} \mid C_{k},x_{1})\cdot\cdot\cdot p(x_{n}\mid C_{k},x_{1},....,x_{n-1})
\end{align}
$$

Naive Bayes에서는 $$C_{k}$$가 주어진 경우, $$x_{i}$$와 $$x_{j}$$가 독립이라는 가정을 한다. 즉 조건부 독립에 대한 가정이 있는 셈이다. Naive Bayes에서 조건부 독립은 다음과 같이 표기할 수 있다.

$$
\begin{align}
	p(x_{i} \mid C_{k}, x_{j}) &= p(x_{i} \mid C_{k}) \\
	p(x_{i} \mid C_{k}, x_{j},x_{k}) &= p(x_{i} \mid C_{k})
\end{align}
$$

따라서 결국 Naive Bayes 모델은 다음과 같이 표현될 수 있다.

$$
\begin{align}
	p(C_{k}\mid \mathbf{x}) &\propto p(C_{k},\mathbf{x}) \\
    &\propto p(C_{k})p(x_{1}\mid C_{k})p(x_{2}\mid C_{k}) \cdot\cdot\cdot p(x_{n}\mid C_{k}) \\
    &\propto p(C_{k})\prod_{i=1}^{n}p(x_{i}\mid C_{k})
\end{align}
$$

결국 Naive Bayes 모델은 가장 가능성 높은 class를 찾아내는 것으로 다음과 같이 $$\hat{y} = argmax_{k \in \{1,...,K\}} p(C_{k})\prod_{i=1}^{n}p(x_{i}\mid C_{k})$$ 로 표현할 수 있다.