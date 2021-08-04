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

$$ x^{*} = \argmax_{x \in \mathbf{X}} f(x) $$

이는 우리가 알아내고자 하는 어떠한 underlying function $$f(x)$$가 있고 domain $$\mathbf{X}$$ 에 속한 데이터 포인트 $$x$$ 를 순차적으로 $$f(x)$$ 에 대입하여 최대값(최소값)이 나오는 데이터 포인트 $$x^{*}$$ 를 찾는걸 의미한다. 따라서 Optimization을 위해서는 underlying function 에 대한 지식과 input data setting이 필요하다.

분석 과정에서 어떠한 모델을 만들었고 모델에 대한 hyperparameter tuning을 진행하는 상황을 가정하자. 또한 hyperparameter를 결정짓기 위해서 F1 Score를 최소화시키는 것을 평가 지표로 삼았다고 가정하자.

기존 Grid Search와 Random Search는 underlying function을 알아내는 것에 관심이 없다.

![GP](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Bayesian_Optimization.png?raw=true){:width="70%" height="70%"}{: .aligncenter}



#### 참조 문헌
1. [Gaussian Process](https://kaist.edwith.org/aiml-adv/lecture/21300) <br>
2. [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/#Posterior)
3. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)