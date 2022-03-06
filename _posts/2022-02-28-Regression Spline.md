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

Spline에 앞서 Piecewise Polynomial Regression에 대해 살펴볼 필요가 있다. 변수 $$\mathbf{x}$$에 대한 전체 범위가 아니라 구역을 나누어 구역 별로 Polynomial Regression을 수행하는 것을 의미한다.

Piecewise 3차 Polynomial을 적합하려는 상황을 가정하자. 아래 이미지에서 (1,1) 구역에 있는 이미지가 Piecewise Polynomial Regression을 의미한다. 그러나 이러한 경우, 적합된 모델에서의 불연속 지점이 발생하기 때문에 하나의 데이터 포인트에서 2개의 값을 가지게 되는 비합리적인 문제가 발생하여 추가 조치가 필요하다.

![Spline](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/spline.png?raw=true){:width="70%" height="30%"}{: .aligncenter}

(1,2)구역에 존재하는 함수는 연속성 제약조건을 추가하였지만, 연결이 부자연스러운 점이 있다. 그래서 불연속 지점에 대해 연속성 제약 뿐만 아니라 해당 지점에서의 1,2차 미분이 가능하다는 제약을 추가하면, (2,1)구역에 존재하는 이미지와 같이 선이 매끄럽게 연결된다. 이를 Regression Spline이라 부른다.

일반화 하면, d차 Spline을 만들기 위해서 각 Piecewise 구역에서 d차 Polynomial Regression을 적합시키고 각 knot에서 연속이며 (d-1)차 미분까지 가능해야한다는 제약을 더 추가해주면 된다.

이러한 제약 조건을 구현하여 매끄러운 Regression Spline을 만들기 위해서 Basis Function이 사용되는데 가장 대표적으로 사용되는 방법이 바로 각 knot마다 truncated power basis function을 적용하는 것이다.

$$
h(x,\xi) = (x-\xi)^{3}_{+} = \begin{cases}
(x-\xi)^{3} \quad \text{if} x>\xi \\
0 \quad \text{otherwise}
\end{cases}
$$

truncated power basis function을 적용한 Cubic Spline(3차 Regression Spline)은 다음과 같이 표현할 수 있다.

$$ y_{i} = \beta_{0} + \beta_{1}x_{i} + \beta_{2}x_{i}^{2} + \beta_{3}x_{i}^{3} + \beta_{4}h(x_{i},\xi_{i}) + \cdots + \beta_{k+3}h(x_{i},\xi_{i}) + \epsilon_{i} $$

그리고 이 Cubic Spline에 대한 parameter($$\beta$$) 추정은 $$\sum_{i=1}^{n}(y_{i}-f(x_{i}))^{2}$$ 을 최소화시키는 방향으로 즉, 기존의 Regression과 마찬가지 방식으로 이루어진다.

그러나 Cubic Spline은 다항회귀를 적합시키기 때문에 양 끝단의 knot에서 모형이 지나치게 급변한다는 단점이 있다. 가장 작은 knot보다 작은 범위, 가장 큰 knot보다는 큰 범위에서는 linear regression을 적용시키는 Natural Cubic Spline이 대안으로 제시되기도 한다.

#### Smoothing Spline


#### 참고문헌

1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
2. [Factorization Machines (FM) 설명 및 Tensorflow 구현](https://greeksharifa.github.io/machine_learning/2019/12/21/FM/)
3. [Field-aware Factorization Machines (FFM) 설명 및 xlearn 실습](https://greeksharifa.github.io/machine_learning/2020/04/05/FFM/)
