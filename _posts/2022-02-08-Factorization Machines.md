---
layout: post
title:  "Factorization Machines"
date: 2022-02-08
author: seolbluewings
categories: Statistics
---

[작성중... ]

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

FM 모델은 bias term($$\w_{0}$$)과 각 변수에 대한 term, 변수들 간의 interaction에 대한 term으로 구분된다.

(이어서 작성...)

.....






포스팅 관련된 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Python%20Class.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [나도 코딩 유투브 강의](https://www.youtube.com/watch?v=kWiCuklohdY)
2. [점프 투 파이썬](https://wikidocs.net/book/1)