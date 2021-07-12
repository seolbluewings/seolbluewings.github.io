---
layout: post
title:  "Gaussian Process"
date: 2021-07-11
author: seolbluewings
categories: Statistics
---

[작성중...]

Input 데이터 $$x_{i}$$에 대한 Target Output $$t_{i}$$는 일반적으로 $$t_{i}=y(x_{i})+\epsilon_{i}$$ 로 표현 가능하다. 모델은 어떠한 현상을 수식으로 표현하는 것인데 이는 결국 $$t_{i}$$를 가장 잘 설명할 수 있는 최적의 함수 $$y(x_{i})$$를 구하는 것이라고 볼 수 있다.

함수 $$y(x_{i})$$가 선형회귀 식이라고 할 때, 지금까지는 함수 $$y(x_{i})$$의 parameter인 $$\theta$$의 분포 $$p(\theta\vert x,y)$$를 찾는 것을 목표로 했다. 그러나 이제는 함수 자체에 대한 추론을 해보고 싶다. 이 함수 자체에 대한 추론을 진행하는 것이 가우시안 과정(Gaussian Process)이다.

Gaussian Process는 다른 베이지안 추론과 유사하게 함수에 대한 Prior 분포를 설정하고 데이터를 통해 관찰한 뒤, 함수에 대한 Posterior 분포를 생성한다.
 
#### Gaussian Process를 활용한 Regression

##### 1. Linear Regression에서 시작





#### 참조 문헌
1. [Permutation Importance](https://www.kaggle.com/dansbecker/permutation-importance) <br>
2. [Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)
