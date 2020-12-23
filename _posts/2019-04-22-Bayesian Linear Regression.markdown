---
layout: post
title:  "Bayesian Linear Regression"
date: 2019-04-22
author: seolbluewings
categories: Bayesian
---

일반적으로 선형 회귀(linear regression)는 다음과 같이 표현된다.

$$ y = {\bf X} {\bf \beta}+\epsilon $$

여기서 $$\epsilon_{i} \sim \mathcal{N}(0,\sigma^{2})$$ 이라 가정한다면, 다음과 같이 표현할 수 있다.

$$ y\vert \mathbf{X},\beta,\sigma^{2} \sim \mathcal{N}(\mathbf{X}\beta, \sigma^{2}I) $$

입력 데이터 집합 $${\bf X}=\{ {\bf X_{1}},...,{\bf X_{N}} \}$$ 과 타깃 변수 $${\bf y}=\{ y_1,...,y_{N}\}$$ 이 있다고 하자. 각 데이터가 먼저 언급한 분포로부터 독립적으로 추출되었다는 가정 하에 다음과 같은 Likelihood 함수를 얻을 수 있다.

$$
\begin{align}
{\it p}({\bf y}\vert{\bf X},{\bf \beta},\sigma^{2}) &= \prod_{i=1}^{N} \mathcal{N}(y_{i}\vert{\bf x^{T}_{i}}{\bf \beta},\sigma^{2}) \nonumber \\
\log{\it p}({\bf y}\vert{\bf X},{\bf \beta},\sigma^{2}) &= \sum_{i=1}^{N}\log\mathcal{N}(y_{i}\vert{\bf x^{T}_{i}}{\bf \beta},\sigma^{2}) \nonumber \\
\log{\it p}({\bf y}\vert{\bf X},{\bf \beta},\sigma^{2}) &= -\frac{N}{2}\log(2\pi)-\frac{N}{2}\log(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i=1}^{n}(y_{i}-\mathbf{x_{i}}^{T}\beta)^{2} \nonumber
\end{align}
$$

노이즈가 가우시안 분포를 따르면, 선형 모델에 대해 Maximum Likelihood를 최대화하는 방식은 LSE 방식과 동일하며, 그 내용은 다음의 [링크](https://seolbluewings.github.io/%EC%84%A0%ED%98%95%EB%AA%A8%EB%8D%B8/2019/04/13/Linear-Regression.html)에서 확인할 수 있다.

다만 MLE를 구하는 것이 반드시 좋은건 아니다. MLE 방식으로 해를 구하면 언제나 overfitting하는 복잡한 모델을 선택하게 되는 것으로 알려진다. 대신 베이지안 방법론으로 선형 회귀를 시행하면 MLE에서 발생하는 overfitting 문제를 피할 수 있으며 훈련 데이터만 가지고 모델의 복잡도를 자동적으로 결정할 수 있다고 알려진다. 이에 대해서 알아보도록 하자.

#### Posterior Distiribution 구하기

Parameter에 대한 Posterior Distribution을 구하기에 앞서 우선 $$\beta$$와 $$\sigma^{2}$$에 대한 Prior Distribution을 다음과 같이 Conjugate prior 형태인 $$\beta \sim \mathcal{N}(\beta_{0},\Sigma_{0})$$, $$\sigma^{2} \sim IG(a,b)$$ 로 설정한다.

prior distribution을 위와 같이 설정하면, Posterior Distribution은 다음과 같이 계산된다.

$$
\begin{align}
{\it p}(\beta,\sigma^{2}\vert y) &\propto (\sigma^{2})^{-n/2} \times \text{exp}\{ \frac{1}{2\sigma^{2}}(y-X\beta)^{T}(y-X\beta)\} \nonumber \\
&\times \text{exp}\{ -\frac{1}{2}(\beta-\beta_0)^{T}\Sigma^{-1}_{0}(\beta-\beta_{0})\} \times (\sigma^{2})^{-a-1}\cdot exp\{-b/\sigma^{2}\} \nonumber
\end{align}
$$

지금 우리가 구한 것은 $$(\beta,\sigma^{2})$$에 대한 Joint Posterior Distribution이다. Parameter에 대한 값을 Sampling 하고 싶은데 이 Joint Posterior Distribution 분포는 우리가 알지 못하는 분포다. 그래서 데이터를 추출해내기 어렵다. 그래서 우리는 각 Parameter에 대한 Full Conditional Posterior Distribution을 구하고 여기서 각 Parameter에 대한 Sampling을 수행한다. 그렇다면, 우리가 그 다음 수행할 과정은 각 Parameter에 대한 Full Conditional Posterior Distribution이다.

먼저 $$\beta$$에 대한 Full Conditional Posterior Distribution 을 구하는 과정은 다음과 같다. 기존의 Joint Posterior Distribution에서 $$\beta$$가 들어간 부분을 다 가져와서 써보고 우리가 아는 형태의 분포의 형태로 바꿔가는 작업을 수행한다.

$$
\begin{align}
{\it p}(\beta\vert\sigma^{2},y) &\propto \text{exp}\left[-\frac{1}{2}\{\frac{1}{\sigma^{2}}\beta^{T}X^{T}X\beta-\frac{2}{\sigma^{2}}\beta^{T}X^{T}y\}-\frac{1}{2}\{\beta^{T}\Sigma^{-1}_{0}\beta-2\beta^{T}\Sigma^{-1}_{0}\beta_{0}\}\right] \nonumber \\
&\propto \text{exp}\left[-\frac{1}{2}\{\frac{1}{\sigma^{2}}\beta^{T}X^{T}X\beta-\frac{2}{\sigma^{2}}\beta^{T}X^{T}y\}-\frac{1}{2}\{\beta^{T}\Sigma^{-1}_{0}\beta-2\beta^{T}\Sigma^{-1}_{0}\beta_{0}\}\right] \nonumber
\end{align}
$$

이 수식을 잘 살펴보면, 정규분포의 형태와 닮았다. 따라서 $$\beta$$에 대한 Full Conditional Posterior Distribution은 다음과 같이 정의할 수 있다.

$$
{\it p}(\beta\vert\sigma^{2},y) \sim \mathcal{N}(\mu_{\beta},\Sigma_{\beta})
$$

여기서 $$\Sigma_{\beta}=\bigg( \frac{1}{\sigma^2}X^{T}X+\Sigma^{-1}_{0}\bigg)^{-1}$$ 이며 $$\mu_{\beta}=\Sigma_{\beta}^{-1}\cdot\bigg(\frac{1}{\sigma^2}X^{T}y+\Sigma^{-1}_{0}\beta_{0}\bigg)$$ 이다.

마찬가지 방식으로 $$\sigma^{2}$$에 대한 Full Conditional Posterior Distribution을 구하면 다음과 같다.

$$
\begin{align}
{\it p}(\sigma^{2}\vert\beta,y) &\propto (\sigma^{2})^{-n/2}\times(\sigma^{2})^{-a-1}\times \text{exp}\left[-\frac{1}{\sigma^{2}}\{ \frac{1}{2}(y-X\beta)^{T}(y-X\beta)+b\}\right] \nonumber \\
\therefore {\it p}(\sigma^{2}|\beta,y) &\sim IG(\frac{n}{2}+a,\frac{1}{2}(y-X\beta)^{T}(y-X\beta)+b)
\end{align}
$$

Parameter에 대한 Full Conditional Posterior Distribution으로부터 Sampling을 진행하는 것은 [Gibb Sampler 포스팅](https://seolbluewings.github.io/bayesian/2019/05/22/Gibbs-Sampler.html) 을 통해 확인할 수 있으며, 이에 대한 [Python 코드](https://github.com/seolbluewings/pythoncode/blob/master/5.Gibbs%20Sampler.ipynb) 와 [R 코드](https://github.com/seolbluewings/R_code/blob/master/Gibbs%20Sampler.ipynb) 를 통해서도 확인 가능하다.


#### Bayesian Linear Model 학습 과정

![BLR](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Figure3.7.png?raw=true){:width="50%" height="50%"}{: .center}

위의 그림은 1차원 데이터 x에 대한 회귀모델을 나타낸 것으로 $$y=\beta_{0}+\beta_{1}x$$ 형태이며 이 모델 상에서 데이터 집합의 크기가 커짐에 따른 베이지안 학습 결과를 보여준다. 현재의 Posterior 분포는 새로운 데이터 포인트가 관측된 후, 새로운 Prior distribution이 되는 베이지안 학습의 순차적인 모습을 보여준다.

1행은 데이터를 하나도 관측하지 못한 상황. 가운데 열에 있는 Prior distribution 공간에 있는 $${\bf \beta}$$ 를 추출하여, 마지막열에 6개의 모델을 그려냈다.

2행은 첫번째 데이터(파란 원)를 관측한 이후의 상황. 가장 왼쪽열은 Likelihood $${\it p}(y\|x,\beta)$$ 를 $$\beta$$ 에 대한 함수로 그린 것이다. 이 Likelihood를 1행의 2번째열인 Prior distribution과 곱하면, 2행의 2번째 열값인 Posterior distribution을 구할 수 있다. 그리고 이 Posterior distribution에서 추출한 $$\beta$$를 바탕으로 마지막열에 모델을 그려낸다.

마찬가지로 3번째 행은 두번째 데이터를 관측한 이후의 상황이며, 두번째 데이터에 대한 Likelihood와 이전의 Posterior가 Prior의 역할을 하여 3행의 2번째 열에 있는 $$\beta$$의 Posterior 분포를 얻는다. 이 Posterior로부터 추출한 $$\beta$$를 바탕으로 한 모델이 3행 3열에 위치해있다.

이와같이 베이지안 방식으로 업데이트 모델을 만들어낼 수 있다. sample이 하나 추가 될 때, 기존의 Posterior distribution은 Prior distribution으로 활용될 수 있으며 데이터가 추가 될수록 Posterior Distribution이 점차 특정지어지는 것을 확인 가능하다.

#### 참고문헌

1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)






