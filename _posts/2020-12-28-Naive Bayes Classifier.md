---
layout: post
title:  "Naive Bayes Classifier"
date: 2020-12-28
author: seolbluewings
comments : true
categories: Bayesian
---

[작성중...]

나이브 베이즈 분류기(Naive Bayes Classifier)는 베이즈 정리에 기반하여, 즉 조건부 확률을 이용해 각 Class에 속할 확률을 계산하여 분류를 진행하는 학습 모델을 의미한다.

Naive Bayes Classifier는 Naive란 단어가 포함되는 명칭에서 유추할 수 있듯이 가장 단순한 형태의 가정에서 출발한 모델이다. Naive Bayes Classifier는 데이터셋의 모든 컬럼이 동등한 조건으로 영향력을 행사하고 서로 독립적이라는 가정을 바탕으로 시작하는 모델이다.

그런데 사실 모델을 만드는 과정에서 모든 변수의 중요성이 동등하고 상호 독립적인 경우는 거의 없다. 우리가 채무 불이행을 예측하는 모델을 만든다고 생각해보자. 상식적으로 개개인의 소득수준이 성별보다 더 중요한 변수일 것이고 각자의 소득수준은 개인의 학력과 연관이 있을 가능성이 크다. 아닌 사람들도 있겠지만, 대다수 사람들이 생각하기에는 지금 이야기한 관계들이 성립한다고 자연스럽게 받아들인다.

앞서 말했듯이 Naive Bayes Classifier는 이러한 관계를 다 무시한다. 즉, 가장 단순한 가정으로 모델을 생성하기 때문에 필자는 강의시간에 분류 모델을 만들고자 할 때, 무조건 Naive Bayes Classifier보다는 좋은 모델을 만들어야 한다는 이야기를 들은 적이 있다. 상식적으로 받아들일 수 있는 분류기 효율성의 하한(lower bound)을 구하는 모델로 학습과정에서 이해하였고 그런 의미에서 Naive Bayes Classifier가 어떤 원리로 만들어지는지 알 필요가 있다고 생각 된다.

#### Bayesian Decision Theory

베이지안 결정 이론(Bayesian Decision Theory)은 확률을 이용해 의사결정을 수행하는 기본적인 방법 중 하나다. Bayesian Decision Theory가 어떻게 활용되는지를 다중 클래스(Multi-Class) 구분 문제를 통해 확인해보자. 먼저 다음과 같이 K개의 Class가 존재한다고 하자.

$$ \mathcal{y} = \{c_{1},c_{2},...c_{K}\}$$

그리고 $$\lambda_{ij}$$는 실제 Class가 $$c_{j}$$인 샘플이 $$c_{i}$$로 오분류될 때의 손실을 표현하는 것으로 가정하자. 그렇다면 조건부 확률 $$\it{p}(c_{i}\vert x)$$ 를 이용해서 샘플 $$x$$를 Class $$c_{i}$$로 잘못 분류할 때의 기대 손실(expected loss)를 다음과 같은 수식으로 표현할 수 있다. 이를 조건부 리스크(conditional risk)라고 표현한다.

$$ R(c_{i}\vert x) = \sum_{j=1}^{K}\lambda_{ij}\it{p}(c_{j}\vert x) $$

분류 문제에 있어서 우리는 항상 정확한 분류를 해내고자 한다. 따라서 각 샘플 데이터에서 조건부 리스크를 최소화시키는 방향으로 Class를 선택하는 것이 옳다. 이를 수식으로 표현하면 다음과 같으며, 베이즈 최적 분류기(Bayes Optimal Classifier)라고 부른다.

$$ h^{*}(x) = \text{argmin}_{c \in \mathcal{y}} R(c\vert x)$$

만약에 우리의 목표가 분류기의 오차율을 최소화시키는 것이라면 손실 $$\lambda_{ij}$$는 다음과 같이 표현될 수 있다.

$$
\lambda_{ij} =
\begin{cases}
0 \quad i=j \\
1 \quad \text{otherwise}
\end{cases}
$$

이를 반영하면 조건부 리스크에 대한 식은 $$R(c\vert x) = 1- \it{p}(c \vert x)$$ 이고 베이즈 최적 분류기에 대한 식은 다음과 같을 것이다.

$$
\begin{align}
h^{*}(x) &=\text{argmin}_{c \in \mathcal{y}}R(c \vert x) \nonumber \\
&= \text{argmax}_{c \in \mathcal{y}}\it{p}(c \vert x) \nonumber
\end{align}
$$

우리는 여기서 아래의 표기에 주목할 필요가 있다. 베이즈 최적 분류기는 곧 각 샘플 $$x$$에 대해 사후확률 $$\it{p}(c\vert x)$$를 최대화시킬 수 있는 클래스로 결정을 내리게 된다. 확률에 근거하여 판단한다고 했을 때, 가장 가능성이 높은 클래스로, 즉 확률이 높은 클래스로 할당되는 것은 우리의 상식과도 부합하는 결과라고 할 수 있다.

#### Naive Bayes Classifier 


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)