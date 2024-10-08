---
layout: post
title:  "Logistic Regression"
date: 2020-10-19
author: seolbluewings
categories: Statistics
---

로지스틱 회귀는 프로빗 모델처럼 반응변수 범주형인 케이스에서 활용할 수 있는 방법이다. 로지스틱 회귀는 새로운 변수(X)가 주어졌을 때, 반응 변수가 각 범주에 속할 확률이 얼마인지를 추정하며, 추정된 확률에 따라 반응 변수의 Class를 분류하게 된다.

일단 반응 변수의 클래스가 2가지인 경우, 즉 $$ y \in \{0,1\}$$ 로 표현되는 경우를 생각해보자.

![LR](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Logistic.png?raw=true){:width="70%" height="70%"}

그림처럼 성공/실패가 명확하게 구분되는 케이스에서 기존의 선형 모델을 그대로 적용하는 것은 무리가 있다. 비선형 함수를 활용하여 성공/실패, $$y=0$$ 또는 $$y=1$$을 구분하는 것이 바람직하다고 할 수 있다. 그림의 우측에서 활용되는 비선형 함수는 시그모이드 함수이며, 이처럼 시그모이드 함수를 연결함수로 활용하여 데이터를 적합시키는 것을 로지스틱 회귀라고 부른다.

$$
\begin{align}
\pi(\mathbf{X}) &= p(y=1\mid\mathbf{X}) \nonumber \\
\log{\frac{\pi(\mathbf{X})}{1-\pi(\mathbf{X})}} &= \mathbf{X}\beta \nonumber \\
\pi(\mathbf{X}) &= \frac{e^{\mathbf{X}\beta}}{1+e^{\mathbf{X}\beta}} \nonumber
\end{align}
$$

여기서 로지스틱 회귀의 중요한 특징을 하나 발견할 수 있다. 수식의 형태를 고려해볼 때, 로지스틱 회귀는 Odds 관점에서 해석할 수 있다는 것이다.

> **Odds 란?** 어떤 시행에 대한 성공의 확률이 $$\pi$$라 하자. 이 때 성공에 대한 Odds는 $$\frac{\pi}{1-\pi}$$ 이다. 성공할 확률이 실패의 확률보다 몇배에 해당하는지를 보여주는 값이라 할 수 있으며 0 이상의 실수값을 갖는다. Odds가 1 이상의 값을 가지면 성공의 확률이 실패의 확률보다 크다고 할 수 있다. 만약 $$\pi=0.75$$라면, Odds는 3이다.

모델의 결과에 대한 해석이 가능하다는 점은 로지스틱 회귀가 갖는 장점으로 만약 $$\mathbf{X} = (x_{1},...,x_{k})$$ 이고 $$ d \leq k $$일 때, $$\text{exp}(\beta_{d})$$의 값은 나머지 변수가 고정인 상황에서 $$x_{d}$$가 1단위 변화할 때, Odds가 어떻게 변화하는지를 나타내는 값이라고 해석할 수 있다.

따라서 로지스틱 회귀는 선형 모델의 예측 결과를 활용하여 데이터의 로그 Odds에 근사하는 것이다. 여기서 주의할 부분은 이름은 회귀지만, 실제로 행하는 작업은 분류라는 것이다. 더불어 로지스틱 회귀는 단순하게 결과를 0,1로만 return해준다기 보다는 각 Class로 할당될 확률(0,1로 결정될 확률)을 근사적으로 보여준다고 할 수 있다.

앞서 나열했던 수식을 활용하면 다음과 같은 관계가 나타남을 확인할 수 있다.

$$
\begin{align}
\pi(\mathbf{X}) &= p(y=1\mid\mathbf{X}) \nonumber \\
p(y=1\mid\mathbf{X}) &= \pi(\mathbf{X}) = \frac{e^{\mathbf{X}\beta}}{1+e^{\mathbf{X}\beta}} \nonumber \\
p(y=0\mid\mathbf{X}) &= 1-\pi(\mathbf{X}) = \frac{1}{1+e^{\mathbf{X}\beta}} \nonumber
\end{align}
$$

#### 로지스틱 회귀 추정

로지스틱 회귀문제에서 parameter $$\beta$$에 대한 추정은 로지스틱 회귀의 중요한 이슈이다. 기존의 선형 모델에서는 오차제곱합을 최소화시키는 parameter $$\beta$$를 구했다. 이는 Convex 함수이기 때문에 Global한 minimal을 비교적 쉽게 찾을 수 있다.

$$\text{argmin}_{\beta} (Y-X\beta)^{T}(Y-X\beta)$$

비슷한 방식을 취한다고 한다면, 로지스틱 회귀의 경우는 아래와 같을 것이다. h는 연결함수인 것으로 받아들이자.

$$\text{argmin}_{\beta} (Y-h(X\beta))^{T}(Y-h(X\beta))$$

그런데 이 수식은 앞선 단순 선형회귀 모델의 경우와 달리 Convex하지 않게 된다. non-Convex한 함수가 되어 Global minimal을 찾기가 어려워진다. 따라서 로지스틱 회귀에서 parameter $$\beta$$ 를 추정하는 것이 중요한 이슈이다.

기존 선형회귀 모델에서 최소제곱추정량(Least Square Estimator, LSE)말고도 최대우도추정량(Maximum Likelihood Estimator, MLE)가 존재했던 것을 생각해보자. 로지스틱 회귀에서는 MLE을 사용하여 회귀계수 $$\beta$$를 추정한다.

데이터셋을 $$ \{(\mathbf{X}_{i},y_{i})\}_{i=1}^{n}$$ 으로 표현한다고 하자. 로지스틱 회귀는 0 또는 1 (성공 또는 실패)의 값을 갖기 때문에 베르누이 시행을 전제로 한다. 그래서 우리는 확률변수 $$y$$의 분포를 고려할 필요가 있다.

$$ p(y_{i}=1\mid\mathbf{X},\beta) = \pi(\mathbf{X}_{i})^{y_{i}}\left(1-\pi(\mathbf{X}_{i})\right)^{1-y_{i}}$$

따라서 Likelihhod는 다음과 깉이 표현할 수 있을 것이다.

$$
\begin{align}

\text{L} &= \prod_{i=1}^{n} \pi(\mathbf{X}_{i})^{y_{i}}\left(1-\pi(\mathbf{X}_{i})\right)^{1-y_{i}} \nonumber \\
\textit{l} &= \sum_{i=1}^{n}\left((1-y_{i})(-x_{i}^{T}\beta)-\log{(1+\text{exp}(-x_{i}^{T}\beta))}  \right) \nonumber
\end{align}
$$

이를 이용해서 우리는 $$\beta$$를 최대화시키는 값을 구해야하고, $$\beta^{*} = \text{argmin}_{\beta}\textit{l}(\beta)$$ 이러한 상황에서 일반적으로 활용되는 방법은 Gradient Descent, Newton's Method, Metropolis-Hastings Algorithm 이 있다.

이번 포스팅에서는 Metropolis-Hastings Algorithm을 이용하여 $$\beta$$에 대한 추정을 시도하는 것을 살펴보자.

#### Metropolis-Hastings Algorithm

parameter $$\beta$$에 대한 true Posterior distribution을 구하는 식을 먼저 얻어야 할 것이다.

$$ p(\beta\mid y) \propto p(y \mid \beta)p(\beta) $$

여기서 $$\beta$$에 대한 prior를 $$p(\beta) \propto C$$ 로 non-informative prior를 주는 것으로 가정하자. 그렇다면 Posterior는 곧 likelihood에 비례하게 된다.

$$\beta$$에 대한 transition kernel을 $$\beta^{*} \sim \mathcal{N}(\beta,(X^{T}X)^{-1})$$ 라고 설정하면 (t+1)번째 시행에서의 $$\beta$$값은 $$\beta^{(t+1)} \sim \mathcal{N}(\beta^{(t)},(X^{T}X)^{-1})$$ 일 것이다. 따라서 M-H Algorithm에서 활용되는 채택확률 $$\alpha$$는 다음과 같이 계산될 것이다. Transition Kernel의 경우, 대칭 형태인 정규분포이기 때문에 삭제해도 무방하여 계산과정에서는 제외하였다. 

$$
\alpha = \frac{ p(\beta^{(t+1)}\mid y) }{  p(\beta^{(t)}\mid y) }
$$

그리고 새롭게 proposed된 $$\beta^{(t+1)}$$ 값을 확률 $$p=\text{min}(\alpha,1)$$ 로 accept한다. 충분한 iteration을 진행한 이후 $$\beta$$의 Posterior Mode나 Mean 값을 $$\beta$$의 추정값으로 결정 짓는다.


##### 상기 예제에 관련한 코드는 다음의 링크 1. [Python코드](https://github.com/seolbluewings/pythoncode/blob/master/8.Logistic%20Regression.ipynb) 2. [R코드](https://github.com/seolbluewings/R_code/blob/master/Logistic%20Regression.ipynb) 에서 확인할 수 있습니다.


#### 참조 문헌
1. [Categorical Data Analysis](https://d1wqtxts1xzle7.cloudfront.net/45095661/AGRESTI.PDF?1461669052=&response-content-disposition=inline%3B+filename%3DCategorical_Data_Analysis.pdf&Expires=1604326876&Signature=ahH43ZxLJmWSbmVkF3r4fESFrVbC18~TAeNGjspg46h5mIhn7I-XbXcfFBsyGQjcxZb-T0SpluXdBk-yJ1sghNkuWq-8BN3ubrWEbsrbiOS4z~ovVT0vrhW6QfGy7WpbIpeODrq2RBk9FF~taaehvE6YYZAYRBWAoUK9JBfG5ES1qzeEhyZcrmMn-vtbRAYaVSURzBbhtI77jwe~u2CWtcVctTPfptggb4aEfqnVFFXVZeRelBsovgVaaQvg54n-o7xdX-UZFR~ZiTRUsz~rHO~l3HaA6jobD54TL6DVncwfhHEiOD5nOpXj9~2CTNCwzjRxgTW63HCFEIGWYrRzEQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)




