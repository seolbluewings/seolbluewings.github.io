---
layout: post
title:  "Bayesian Optimization"
date: 2021-08-03
author: seolbluewings
categories: Statistics
---

[작성중...]

Gaussian Process를 활용하는 대표적인 사례가 바로 Bayesian Optimization이다. 모델을 생성하는 과정에서 Grid Search와 Random Search 대안으로 Bayesian Optimization이 언급되므로 Bayesian Optimization에 대한 정리를 진행해보고자 한다.

Optimization이란 기본적으로 input 값들을 입력받아 어떠한 함수 $$f(\mathbf{x})$$ 가 반환하는 결과를 maximize(또는 minimize) 하는 과정에서 발생한다.

$$ x^{*} = \text{argmax}_{x \in \mathbf{X}} f(x) $$

이는 우리가 알아내고자 하는 어떠한 underlying function $$f(x)$$가 있고 domain $$\mathbf{X}$$ 에 속한 데이터 포인트 $$x$$ 를 순차적으로 $$f(x)$$ 에 대입하여 최대값(최소값)이 나오는 데이터 포인트 $$x^{*}$$ 를 찾는걸 의미한다. 따라서 Optimization을 위해서는 underlying function 에 대한 지식과 input data setting이 필요하다.

분석 과정에서 어떠한 모델을 만들었고 모델에 대한 hyperparameter tuning을 진행하는 상황을 가정하자. F1 Score나 Accuarcy, MSE 같은 모델의 성능을 평가할 지표를 바탕으로 hyperparameter를 선택하기로 결정했다면, hyperparameter $$\lambda$$ 가 모델의 성능 $$y$$ 에 미치는 영향을 표현하는 함수는 $$ y = f(\lambda) $$ 로 표현할 수 있다. 여기서 함수 $$f$$ 는 hyperparameter와 모델 성능 사이의 관계를 나타내는 objective function이자 우리가 정확하게 알지 못하는 underlying function이기도 하다.

기존 Grid Search와 Random Search는 underlying function을 알아내는 것에 관심이 없다. hyperparameter에 대한 Range를 설정하고 그에 대해 일정 간격 또는 무작위로 값을 넣어서 최대값이 나온 지점을 찾는다.

그러나 Bayesian Optimization은 underlying function $$f$$ 에 대해 학습을 진행하고 $$f$$에 대한 학습 결과를 바탕으로 다음에 탐색해볼 hyperparameter를 결정한다. underlying function에 대한 학습의지 여부가 가장 큰 차이점이며 $$f$$에 대한 학습을 바탕으로 parameter를 탐색할 구간을 좁힌다.

Gaussian Process는 현재까지 관측된 hyperparameter 데이터를 바탕으로 최적의 underlying function을 추정하기 위한 방법이며 Gaussian Process를 바탕으로 다음 차례에 테스트할 hyperparameter를 결정짓는 메카니즘은 Acquisition Function이라 불린다.

![GP](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Bayesian_Optimization.png?raw=true){:width="80%" height="80%"}{: .aligncenter}

그림을 통해서 Bayesian Optimization에 대해 더욱 쉽게 이해할 수 있다. 먼저 2개 데이터를 바탕으로 Gaussian Process를 통해 underlying function을 추정했다. 

#### 참조 문헌
1. [Gaussian Process](https://kaist.edwith.org/aiml-adv/lecture/21300) <br>
2. [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/#Posterior)
3. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)