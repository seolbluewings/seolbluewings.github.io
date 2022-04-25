---
layout: post
title:  "Granger Causality"
date: 2022-04-24
author: seolbluewings
categories: Statistics
---

서로 시차가 존재하는 데이터간의 선/후행을 따져 인과관계를 알아보고자 할 때, 사용할 수 있는 방법 중 하나가 Granger Causality이다.

다음과 같이 시계열 데이터 $$\{x_{t}\}_{t=1}^{T}$$와 $$\{y_{t}\}^{T}_{t=1}$$ 가 존재할 때, $$y_{t}$$가 $$x_{t}$$의 과거 데이터 linear regression 형태로 적합되며 이 linear regression이 통계적으로 유의미할 때, $$\{x_{t}\}_{t=1}^{T}$$와 $$\{y_{t}\}^{T}_{t=1}$$는 Granger Causality 관계에 있다고 말한다.





![Spline](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/spline.png?raw=true){:width="80%" height="30%"}{: .aligncenter}

(1,2)구역에 존재하는 함수는 연속성 제약조건을 추가하였지만, 연결이 부자연스러운 점이 있다. 그래서 불연속 지점에 대해 연속성 제약 뿐만 아니라 해당 지점에서의 1,2차 미분이 가능하다는 제약을 추가하면, (2,1)구역에 존재하는 이미지와 같이 선이 매끄럽게 연결된다. 이를 Regression Spline이라 부른다.

일반화 하면, d차 Spline을 만들기 위해서 각 Piecewise 구역에서 d차 Polynomial Regression을 적합시키고 각 knot에서 연속이며 (d-1)차 미분까지 가능해야한다는 제약을 더 추가해주면 된다.



#### 참고문헌

1. [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)
2. [Regression Splines](https://cdm98.tistory.com/26)
3. [Smoothing Splines](https://cdm98.tistory.com/27?category=749235)
4. [Multivariate Adaptive Regression Spline](https://asbates.rbind.io/2019/03/02/multivariate-adaptive-regression-splines/)