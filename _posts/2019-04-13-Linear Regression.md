---
layout: post
title:  "선형 회귀 모델(Linear Regression)"
date: 2019-04-13 
author: YoungHwan Seol
categories: Bayesian
---

회귀분석이란 P차원 벡터($$\bf{X}$$)가 입력(input)될 때, 그에 대응하는 연속형 타깃 변수(target) $$\bf{y}$$를 예측하는 것이다. N개의 관측값 \(\bf{X}_{N}\)이 있다고 하자. 이에 대응하는 변수 \(\bf{y}_{N}\)이 훈련 집합으로 존재한다고 하자. 이 때, 선형회귀모델의 목표는 새로운 변수 $$X_{\text{new}}$$의 종속변수인 $$y_{\text{new}}$$를 예측하는 것이며 최종적으로 $$\text{P}(y_{\text{new}}\mid X_{\text{new}})$$ 의 분포를 모델링하는 것이다. 우리는 이 예측분포를 통해서 $$X_{\text{new}}$$에 대한 $$y_{\text{new}}$$의 불확실성을 표현할 수 있다.

#### '선형' 회귀(Linear Regression)

가장 단순한 형태의 선형 회귀 모델은 입력 변수들의 선형 결합을 바탕으로 한 모델로 다음과 같이 표현할 수 있다.

$$\bf{Y} = \bf{X}\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\sigma^{2})$$

여기서 짚고 넘어가야할 부분은 무엇에 관한 '선형'이냐는 것이다. 로지스틱 회귀의 경우는 위와 달리 $$\text{logit}(\bf{y}) = \bf{X}\beta $$ 로 표기되는데 이 역시도 선형 모델이라고 불린다. 그래서 입력 변수 $$\bf{X}$$에 대한 선형 함수(결합)가 아닌 $$\beta$$에 대한 선형 함수라고 할 수 있다. 그래서 선형 모델인 것이다. 물론 $$\bf{X}\beta$$의 경우는 입력변수 $$\bf{X}$$의 선형 함수이기도 하다.
















