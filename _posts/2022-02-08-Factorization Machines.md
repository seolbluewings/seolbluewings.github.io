---
layout: post
title:  "Factorization Machines"
date: 2022-02-08
author: seolbluewings
categories: Statistics
---


Factorization Machines(이하 FM)은 실수값으로 이루어진 input 벡터에 대해 범용적으로 적용할 수 있는 모델이다. FM 모델은 데이터가 sparse한 상태에서 SVM이 적절한 비선형 hyper-plane을 만들지 못하는 단점을 보완하기 위해 생성된 모델로 Factorization을 통한 데이터 Sparsity 이슈를 해결한다.

모델 개발을 위해 데이터셋을 생성하다보면 sparse한 데이터셋이 빈번하게 생성된다. 추천 시스템을 만드는 과정에서도 sparse한 데이터가 만들어지지만, 일반적인 고객성향 예측 모델에서도 데이터가 sparse한 경우는 흔하다.

Categorical 변수가 있고 이를 one-hot encoding을 하면 할수록 데이터는 점점 sparse해져간다. 이러한 상황에서 FM은 적절한 대안이 될 수 있다.

타깃변수 $$\mathbf{y}$$를 예측하는 상황에서 선형 모형 $$\mathbf{w}^{T}\mathbf{x}$$ 형태의 모델을 생성하면, 변수간 interaction을 무시하게 된다.

만약 Polynomial 형태나 변수들을 조합해서 비선형 hyer-plane을 만든다고 했을 때는 interaction 자체는 고려할 수 있지만 계산량이 급격하게 증가하며 데이터가 더욱 심하게 sparse해질 것이라는 단점이 있다.

이러한 이슈를 FM은 변수(feature)간의 interaction을 나타내는 matrix $$\mathbf{W}$$를 factorization 함으로써 해결한다.

#### Factorization Model

FM 모델은 다음과 같은 수식으로 표현할 수 있다.

$$ \hat{y}(\mathbf{x}) = w_{0} + \sum_{i=1}^{n}w_{i}x_{i} + \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i},v_{j}>x_{i}x_{j} $$

여기서 $$x_{i}$$는 $$\mathbf{x}$$에 대한 하나의 행 벡터이다.

FM 모델은 bias term($$w_{0}$$)과 각 변수에 대한 term, 변수들 간의 interaction에 대한 term으로 구분된다. 데이터가 sparse한 환경에서 변수간 interaction을 직접적이며 독립적으로 추정하는 것은 어려운 일이다. 그러나 k개의 latent vector로 interaction을 표현함으로써 sparse한 경우에도 변수 간 interaction 추정이 가능해진다.

$$ <v_{i},v_{j}> = \sum_{f=1}^{k}v_{i,f}v_{j,f} $$

변수의 interaction을 의미하는 $$\mathbf{W}$$는 positive definite일 때 $$\mathbf{W} = \mathbf{V}\mathbf{V}^{T}$$로 표현할 수 있는 $$\mathbf{V}$$가 항상 존재한다. 충분한 크기의 latent space dimension이 정의되었을 때, FM 모델은 어떠한 interaction term $$\mathbf{W}$$도 표현 가능하다.

FM의 장점 중 하나는 이 interaction term을 linear한 형태로 변경시킬 수 있다는 것이다.

$$
\begin{align}
\sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_{i},v_{j}>x_{i}x_{j} &= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}< v_{i},v_{j}>x_{i}x_{j} - \frac{1}{2}\sum_{i=1}^{n}<v_{i},v_{i}>x_{i}x_{i} \nonumber \\
&= \frac{1}{2}\left(\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{j,f}x_{i}x_{j} - \sum_{i=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{i,f}x_{i}x_{i}   \right) \nonumber \\
&= \frac{1}{2}\sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n}v_{i,f}x_{i}  \right)\left(\sum_{j=1}^{n}v_{j,f}x_{j}\right)-\sum_{i=1}^{n}v_{i,f}^{2}x_{i}^{2} \right) \nonumber \\
&= \frac{1}{2}\sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n}v_{i,f}x_{i}  \right)^{2}-\sum_{i=1}^{n}v_{i,f}^{2}x_{i}^{2} \right)
\end{align}
$$

따라서 FM 모델은 계산의 복잡성이 $$\mathcal{O}(kn^{2})$$에서 $$\mathcal{O}(kn)$$ 으로 줄어든다.

이 결과 FM 모델은 SGD와 같은 방식으로 쉽게 계산이 가능해진다.

$$
\frac{\partial}{\partial \theta}\hat{y}(\mathbf{x}) = \begin{cases}
1 \quad \theta = w_{0} \\
x_{i} \quad \\
x_{i}\sum_{j=1}^{n}v_{j,f}x_{j}-v_{i,f}x_{i}^{2} \quad \theta = v_{i,f}
\end{cases}
$$

#### FFM

FM모델에서 1단계 향상된 버전이 FFM(Field-aware Factorization Machines)이다. 기존 FM 모델에서는 각 feature에 대응하는 latent vector는 feature 당 1개였다.

A라는 변수와 B,C 변수 사이의 latent effect를 추정하기 위해서 latent vector를 정의했지만, B와 C의 성격이다를 수 있기 때문에 하나의 vector로 표현하는 것은 무리가 있지 않나? 라는 생각에서 만들어진 것이 FFM 이다.

따라서 FFM에서는 각 feature 마다 latent vector가 여러개 생성될 수 있고 feature 간의 interaction term $$\phi(\mathbf{w},\mathbf{x})$$ 는 다음과 같이 표현 가능하다.

$$ \phi(\mathbf{w},\mathbf{x}) = \sum_{j_{1}=1}^{n}\sum_{j_{2}=j_{1}+1}^{n}<w_{j_{1},f_{2}}, w_{j_{2},f_{1}}> x_{j_{1}}x_{j_{2}} $$

여기서 $$j$$는 개별 변수를 의마하고 $$f$$는 latent space 크기를 의미한다.

FFM은 FM에서 interaction을 구하는 부분이 달라진 것이며 이에 따라 계산량은 증가할 수 밖에 없다. 그러나 일반적으로 FFM이 FM보다 더 좋은 성능을 보이는 것으로 알려진다.

각 변수에 대응하는 parameter에 대한 update는 FM과 마찬가지로 SGD를 활용한다. 이 방식에 대한 실질적 활용은 xlearn 과 같은 라이브러리를 이용해서 이루어질 것이니 구체적인 수식은 생략하기로 한다.


#### 참고문헌

1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
2. [Factorization Machines (FM) 설명 및 Tensorflow 구현](https://greeksharifa.github.io/machine_learning/2019/12/21/FM/)
3. [Field-aware Factorization Machines (FFM) 설명 및 xlearn 실습](https://greeksharifa.github.io/machine_learning/2020/04/05/FFM/)
