---
layout: post
title:  "Regression Spline"
date: 2022-02-28
author: seolbluewings
categories: Statistics
---

[작성중... ]

Modeling 과정에서 Linear Regression을 활용한 케이스가 많다. 이는 Linear Model이 결과에 대한 해석이 용이하다는 점에서 기인한 것이다. 그러나 Linear Model은 예측력이 다소 떨어진다는 단점이 있고 특히 현실의 문제를 해결하는 과정에서 $$f(\mathbf{x})$$의 함수 $$f$$가 선형함수인 경우는 드물다고 봐야한다.

그래서 Linear Regression에 대한 변형을 주게 되는데 이번 포스팅에서는 Regression Spline이라는 것을 살펴보고자 한다.

#### Basis Function

기존의 Linear Regression이 $$y_{i} = \beta_{0}+\beta_{1}x_{i} + \epsilon_{i} $$ 형태였다면, Basis Function을 적용한 Linear Model은 다음과 같이 표현할 수 있다.

$$ y_{i} = \beta_{0} + \beta_{1}b_{1}(x_{i}) + \cdots + \beta_{k}b_{k}(x_{i}) + \epsilon_{i} $$

이는 $$\mathbf{x}$$에 대해 어떠한 기저 함수(basis function)에 대한 Linear Combination으로 $$\mathbf{y}$$를 표현하는 셈이 된다.

$$b_{k}(x_{i})$$는 분석가가 지정하기 나름이다. 만약 $$b_{k}(x_{i}) = x_{i}^{k}$$ 라면, 이는 Polynomial Regression 이 되고 $$b_{k}(x_{i}) = \mathcal{I}(c_{j} < x_{i} < c_{j+1})$$ 이면 Piecewise Constant Regression이 된다.

Regression Spline은 Basis Function을 사용하여 Piecewise Constant Regression과 Polynomial Regression을 적절히 조합한 모델이라 할 수 있다.

#### Regression Spline

Spline에 앞서 Piecewise Polynomial Regression에 대해 살펴볼 필요가 있다. 변수 $$\mathbf{x}$$에 대한 전체 범위가 아니라 구역을 나누어 구역 별로 Polynomial Regression을 수행하는 것을 의미한다. 아래 이미지에서 (1,1) 구역에 있는 이미지가 Piecewise Polynomial Regression을 의미한다. 그러나 이러한 경우, 적합된 모델에서의 불연속 지점이 발생하기 때문에 하나의 데이터 포인트에서 2개의 값을 가지게 되는 비합리적인 문제가 발생하여 추가 조치가 필요하다.

![Spline](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/spline.png?raw=true){:width="70%" height="30%"}{: .aligncenter}



#### 참고문헌

1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
2. [Factorization Machines (FM) 설명 및 Tensorflow 구현](https://greeksharifa.github.io/machine_learning/2019/12/21/FM/)
3. [Field-aware Factorization Machines (FFM) 설명 및 xlearn 실습](https://greeksharifa.github.io/machine_learning/2020/04/05/FFM/)
