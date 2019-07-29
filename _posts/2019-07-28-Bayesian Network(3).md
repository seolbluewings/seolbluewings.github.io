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

반면 생성 모델은 $$p(x\mid y)$$와 $$p(y)$$ 의 분포를 학습하고 이를 바탕으로 $$p(y \mid x) \propto p(y)p(x \mid y)$$를 간접적으로 계산해낸다. likelihood와 posterior probability를 이용하여 클래스를 결정짓는 decision boundary를 생성하는 것이다. 생성 모델의 경우 $$p(x \mid y)$$를 구축하기 때문에 이 모델을 활용하여 x에 대한 sampling을 진행할 수 있다.

따라서 우리는 어떤 확률 분포로부터 임의의 sample을 만들어내는 방법을 알아야 하며, 지금부터 방향성 그래프 모델과 관련이 있는 ancestral sampling이라는 방법에 대해 소개해보고자 한다.

k개의 확률변수로 이루어진 joint probability $$p(x_{1},...,x_{k})$$가 있다고 하자.

모델에 대한 그래프가 다음과 같이 주어진 상황에서 각 노드에 번호를 붙인다. 이 때, 자식 노드에는 부모 노드보다 더 큰 번호를 부여한다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/conditonal.JPG?raw=true){:width="30%" height="30%"}{: .center}

ancestral sampling 기법은 가장 작은 노드에서 시작하여 그래프 아래 노드로 이동하며 sampling하는 방식을 취한다. 먼저 $$p(x_{1})$$에서 sample 하나를 생성해 $$\hat{x_{1}}$$이라 하며, 노드 순서대로 sample을 생성한다.

위 그림에서 $$x_{4}$$의 경우, $$p(x_{4} \mid x_{1},x_{2},x_{3})$$ 로부터 sample을 생성하며 이렇게 k번째 노드에서 sample을 생성할 때는 $$p(x_{k} \mid pa_{k})$$ 분포를 활용한다. 부모 노드에 대한 sample은 이미 주어지기 때문에 conditional probability를 바탕으로 sample을 얻을 수 있다.

이렇게 $$x_{k}$$까지 sample을 생성하면 joint probability로부터 하나의 샘플 $$(x_{1},...,x_{k})$$을 얻게 된다. 만약 일부 확률변수 $$x_{1},x_{3}$$에 대한 결합 분포에서 sampling하길 희망한다면, 전체 sampling 후 $$\hat{x_{1}},\hat{x_{3}}$$ 값만 사용하면 된다.

그래프 모델을 통해 우리는 데이터가 발생하게 된 인과 과정을 알아볼 수 있다. 관찰 데이터가 왜 발생하였는지를 모델을 통해 설명할 수 있어 이러한 이유로 데이터를 생성할 수 있는 인과 모델을 생성 모델(generative model)이라 부른다.

#### Naive Bayes

앞선 글에서 먼저 소개했던 Common Parent 구조는 Naive Bayes의 가장 전형적인 예시이며 동시에 바로 위에서 언급한 생성 모델이기도 하다. Naive Bayes는 conditional probability를 이용하여 k개의 가능한 확률적 결과(분류)를 다음과 같이 할당한다.

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

따라서 Naive Bayes 모델은 다음과 같이 표현될 수 있다.

$$
\begin{align}
	p(C_{k}\mid \mathbf{x}) &\propto p(C_{k},\mathbf{x}) \\
    &\propto p(C_{k})p(x_{1}\mid C_{k})p(x_{2}\mid C_{k}) \cdot\cdot\cdot p(x_{n}\mid C_{k}) \\
    &\propto p(C_{k})\prod_{i=1}^{n}p(x_{i}\mid C_{k})
\end{align}
$$

결국 Naive Bayes 모델은 가장 가능성 높은 class를 찾아내는 것으로 다음과 같이 $$\hat{y} = argmax_{k \in \{1,...,K\}} p(C_{k})\prod_{i=1}^{n}p(x_{i}\mid C_{k})$$ 로 표현할 수 있다.

#### Using Bayes Classifier as a Generative Model

이번에는 Bayes Classifier를 이용하여 새로운 데이터를 generate 하는 과정을 진행해보고자 한다.

다음의 notation을 따르기로 하자.

- x : input data
- y : data label
- $$p(x \mid y)$$ : label y가 given일 때, x의 확률
- $$p(x \mid y=1)$$ : label $$y=1$$일 때, x의 확률
- $$p(y \mid x)$$ : data x가 주어진 상황에서 y의 확률 (분류 문제의 목표이기도 하다)

생성 모델(generative model)에 관한 논의를 진행하고자 하는 과정에서 우리는 $$p(y \mid x)$$ 대신 $$p(x \mid y)$$를 학습하는 것에 주목한다. 우리의 목표는 $$y=y_{i}$$로 class가 주어진 상황에서 $$p(x \mid y=y_{i})$$의 값을 구하는 것이다.

문제를 간단하게 만들기 위해 지금부터 다루는 input data가 모두 가우시안 분포를 따를 것이라고 가정하며 우리는 y의 label이 주어진 상황에서 $$p(x \mid y=y_{i})$$가 어떤 가우시안 분포를 따르는지 알아내고자 한다.

$$p(x \ mid y=y_{i})$$의 분포를 알아내기 위한 과정은 다음과 같다.

1. class $$y_{i}$$에 해당하는 모든 data point $$x_{i}$$를 찾아낸다.
2. 이 때의 $$x_{i}$$ 값들을 가지고 $$\mu_{y_{i}}$$와 $$\sigma_{y_{i}}^{2}$$를 계산한다.

데이터를 생성하는 과정은 [Kaggle](https://www.kaggle.com/c/digit-recognizer/data)에 업로드 되어진 MNIST 데이터를 활용할 것이다.

우선 필요한 라이브러리와 데이터를 불러온다.

~~~
import os
from builtins import range, input
import numpy as np
import pandas as pd
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt

def get_mnist():
    df = pd.read_csv(r'C:\Users\user\Python\train.csv')
    data = df.as_matrix()

    # data를 random하게 섞는다.
    np.random.shuffle(data)
    # 데이터 X와 label Y 2가지 파트로 구분한다.
    X = data[:, 1:] / 255.0 # pixels values are in [0, 255] => Normalize the data
    Y = data[:, 0]
    return X, Y
~~~

BayesClassifier란 class를 만든다. 

