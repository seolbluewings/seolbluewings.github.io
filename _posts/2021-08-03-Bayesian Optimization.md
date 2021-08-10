---
layout: post
title:  "Bayesian Optimization"
date: 2021-08-03
author: seolbluewings
categories: Statistics
---

Gaussian Process를 활용하는 대표적인 사례가 바로 Bayesian Optimization이다. 모델을 생성하는 과정에서 Grid Search와 Random Search 대안으로 Bayesian Optimization이 언급되므로 Bayesian Optimization에 대한 정리를 진행해보고자 한다.

Optimization이란 기본적으로 input 값들을 입력받아 어떠한 함수 $$f(\mathbf{x})$$ 가 반환하는 결과를 maximize(또는 minimize) 하는 과정에서 발생한다.

$$ x^{*} = \text{argmax}_{x \in \mathbf{X}} f(x) $$

이는 우리가 알아내고자 하는 어떠한 underlying function $$f(x)$$가 있고 domain $$\mathbf{X}$$ 에 속한 데이터 포인트 $$x$$ 를 순차적으로 $$f(x)$$ 에 대입하여 최대값(최소값)이 나오는 데이터 포인트 $$x^{*}$$ 를 찾는걸 의미한다. 따라서 Optimization을 위해서는 underlying function 에 대한 지식과 input data setting이 필요하다.

분석 과정에서 어떠한 모델을 만들었고 모델에 대한 hyperparameter tuning을 진행하는 상황을 가정하자. F1 Score나 Accuarcy, MSE 같은 모델의 성능을 평가할 지표를 바탕으로 hyperparameter를 선택하기로 결정했다면, hyperparameter $$\lambda$$ 가 모델의 성능 $$y$$ 에 미치는 영향을 표현하는 함수는 $$ y = f(\lambda) $$ 로 표현할 수 있다. 여기서 함수 $$f$$ 는 hyperparameter와 모델 성능 사이의 관계를 나타내는 objective function이자 우리가 정확하게 알지 못하는 underlying function이기도 하다.

기존 Grid Search와 Random Search는 underlying function을 알아내는 것에 관심이 없다. hyperparameter에 대한 Range를 설정하고 그에 대해 일정 간격 또는 무작위로 값을 넣어서 최대값이 나온 지점을 찾는다.

그러나 Bayesian Optimization은 underlying function $$f$$ 에 대해 학습을 진행하고 $$f$$에 대한 학습 결과를 바탕으로 다음에 탐색해볼 hyperparameter를 결정한다. underlying function에 대한 학습의지 여부가 가장 큰 차이점이며 $$f$$에 대한 학습을 바탕으로 parameter를 탐색할 구간을 좁힌다.

Gaussian Process는 현재까지 관측된 hyperparameter 데이터를 바탕으로 최적의 underlying function을 추정하기 위한 방법이며 Gaussian Process를 바탕으로 다음 차례에 테스트할 hyperparameter를 결정짓는 메카니즘은 Acquisition Function이라 불린다.

![GP](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Bayesian_Optimization.png?raw=true){:width="80%" height="80%"}{: .aligncenter}

그림을 통해서 Bayesian Optimization에 대해 더욱 쉽게 이해할 수 있다. 먼저 2개 데이터를 바탕으로 Gaussian Process를 활용해 underlying function $$f$$를 추정했다. 이후 Acquisition function $$u$$를 최대화하는 점을 구하여 다음에 샘플링할 hyperparameter의 데이터 포인트를 결정한다.

샘플링이 결정된 점 $$x_{i}$$에서의 $$f(x_{i})$$ 값을 구할 수 있고 데이터 관측을 통해 해당 데이터 지점에서의 불확실성이 줄어든 것을 확인할 수 있다. 다시 Acquisition function을 최대화하는 점을 찾고 해당 지점에서의 함수값을 구하면서 Bayesian Optimization이 진행된다.


#### Acquisition Function

Acquisition Function은 샘플링 가능한 공간에서 다음 샘플링 포인트를 제안하는 함수이다. 따라서 hyperparameter tuning 시 다음번 탐색할 hyperparameter 조합은 Acquisition Function이 결정한다고 볼 수 있다. Acquisition Function은 2가지 관점에 의해 다음 데이터 포인트를 결정 짓는다.

하나의 관점은 exploitation 이고 다른 하나의 관점은 exploration이다. exploitation은 현재까지 관측된 $$(x_{i},f(x_{i})$$ 데이터보다 더 큰 값을 가질만한 곳을 우선시하여 탐색한다. 더 큰 값을 가진다는 것은 Gaussian Process의 평균값으로 판단한다. 아래의 그림과 같이 현재까지 관측된 데이터들 중에서 최대값을 갖는 조합이 $$(x^{+}, f(x^{+}))$$ 라면, Gaussian Process의 평균(검정 실선)이 $$f(x^{+})$$ 보다 큰 곳만 탐색한다.

한편 exploration은 현재까지 관측되지 않은 불확실성이 높은 데이터 포인트 구간을 탐험하는 것이다. Gaussian Process를 참고하여 Variance가 큰 곳부터 데이터를 샘플링하는 것이다.

![GP](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Bayesian_Optimization2.png?raw=true){:width="80%" height="80%"}{: .aligncenter}

현재까지의 관측된 데이터가 $$\mathcal{D}_{1:t-1} = \{(x_{1},f(x_{1}),...,(x_{t-1},f(x_{t-1}))\}$$ 이고 Acquisition Function을 $$u$$ 라고 표현한다면 Acquisition Function이 제안하는 다음 샘플링 포인트는 다음과 같다.

$$ x_{t} = \text{argmax}_{\mathbf{x}}u(\mathbf{x}\vert \mathcal{D}_{1:t-1})$$

Acquisition Function 을 통해 제안된 $$x_{t}$$에서의 함수값 $$f(x_{t})$$를 구하고 $$\mathcal{D}_{1:t} = \mathcal{D}_{1:t-1},(x_{t},f(x_{t}))$$ 를 활용해 Gaussian Process를 업데이트 한다. 그리고 이 과정을 반복하여 최적의 값 $$x^{*}$$ 를 발견하게 된다.

여러 종류의 Acquisition Function에 대한 소개와 각 Acquisition Function 마다의 원리는 이 [포스팅](https://mambo-coding-note.tistory.com/284)을 참고하면 좋다.

#### 결론

실질적으로 Bayesian Optimization은 hyperparameter tuning 과정에서 많이 사용하게 된다. hyperparameter와 모델 성능간의 관계는 우리가 알지 못하는 함수 관계를 지니기 때문에 이 함수에 대해서는 Gaussian Process로 추정한다.

기존의 Grid Search가 hyperparameter space를 격자로 나누어 지정된 몇가지 값을 직접 대입해보는 식으로 hyperparameter를 결정한다. 격자를 촘촘하게 나눌수록 연산량이 증가하며 격자점이 아닌 곳은 탐색하지 않는다는 단점이 있다.

Random Search의 경우 hyperparameter space에서 hyperparameter를 임의 조합하여 최적의 hyperparameter를 발견하는 방식이다. 두가지 방법은 포스팅의 앞부분에서 언급한 바와 같이 hyperparameter와 모델의 성능간의 관계 함수 $$f(\lambda)$$에 대해 관심이 없는 방식이다.

함수 $$f(\lambda)$$ 에 대한 추정과 동시에 보다 효율적인 방식으로 hyperparameter space를 탐색한다는 점을 고려했을 때, 모델링 과정에서 Bayesian Optimization을 사용하는 것은 아주 바람직해 보인다.




#### 참조 문헌
1. [Gaussian Process](https://kaist.edwith.org/aiml-adv/lecture/21300) <br>
2. [Bayesian Optimization](http://krasserm.github.io/2018/03/21/bayesian-optimization/) <br>
3. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)