---
layout: post
title:  "Probit model"
date: 2019-07-06
author: seolbluewings
categories: ML
---

기존의 회귀분석 모델 $$\mathbf{y} = \mathbf{X}\beta + \epsilon$$ 모형은 보통 1. $$\mathbf{X}$$와 $$\mathbf{y}$$ 사이의 선형 관계가 있고  2. $$\mathbf{y}$$가 정규분포를 따른다고 볼 수 있을 때, 활용하는 것이 적절하다.

그러나 현실 세계에서는 이러한 조건에 부합하지 않는 데이터가 많다. $$\mathbf{X}$$와 $$\mathbf{y}$$가 선형관계가 아닌 S자 형태의 관계를 보일 수도 있고 $$\mathbf{y}$$가 정규분포가 아닌 이항분포, 다항분포 등을 따를 수도 있다. 이러한 상황에서는 기존의 선형모형을 활용하는 것이 적절하지 않다.

$$\mathbf{X}$$와 $$\mathbf{y}$$ 사이의 다양한 관계를 표현할 수 있는 일반화 선형 모형(Generalized Linear Model)을 이러한 상황에 활용할 수 있다.

$$\mathbf{y} = h(\mathbf{X}\beta)$$

위와 같은 형태로 연결 함수(Link Function) $$h$$ 를 이용해 일반화 선형 모형을 표현할 수 있다. 가장 빈번하게 활용하는 Link Function은 $$\mathbf{y}$$가 이항분포를 따를 때 사용하는 프로빗(probit) 함수와 로지스틱(logistic) 함수가 있다.

$$
\begin{align}
y &\sim \text{Ber}(p) \nonumber \\
h(p) &= log\left(\frac{p}{1-p}\right) \quad \text{if logistic function} \nonumber \\
h(p) &= \Phi^{-1}(p) \quad \text{if probit function}
\end{align}
$$

Link Function을 활용해 Probit Model을 계산하는 과정에서 잠재변수(Latent Variable) $$\mathbf{u}$$ 를 이용하게 된다. 그리고 베이지안 방법론은 Latent Variable이 있는 상황에서 강점을 갖는다.

### 프로빗 모형(Probit Model)

Probit Model은 반응변수 $$\mathbf{y}$$가 0 또는 1의 값을 갖는 이항분포를 따를 때 사용할 수 있다. n개의 반응변수 $$y_{i},\; i=1,...,n$$ 각각에 대하여 독립적으로 $$\text{Ber}(p_{i})$$를 가정하여 진행하는 것을 바탕으로 한다.

이 때, $$y_{i}$$의 기대값인 $$p_{i}$$와 설명변수의 선형결합인 $$x_{i}^{T}\beta$$의 관계를 살펴볼 필요가 있다.

$$p_{i}$$는 0과 1사이에서의 값을 가져야 한다. 그런데 $$x_{i}^{T}\beta$$의 경우는 (0,1)로 값을 한정지을 수 없다. 또한 이 관계는 S자 형태의 변화를 설명할 수 없다. 따라서 $$p_{i}=x_{i}^{T}\beta$$의 관계식을 가정하는 것은 적절하지 않다.

이 경우 범위를 (0,1)로 한정시키면서 S자 형태의 변화를 보이는 함수로 정규분포의 누적분포함수(CDF)인 $$\Phi$$를 사용할 수 있고 이를 활용한 것을 Probit Model이라 한다.

$$
\begin{align}
y_{i} &\sim Ber(p_{i}) \\
p_{i} &= \Phi(x_{i}^{T}\beta) \\
\Phi^{-1}(p_{i}) &= x_{i}^{T}\beta
\end{align}
$$

Probit Model에서 우리의 관심을 끄는 parameter는 $$\beta$$이며 이에 대한 likelihood는 다음과 같이 표현할 수 있다.

$$
\begin{align}
l(\beta|y) &= \prod_{i=1}^{n}p_{i}^{y_{i}}(1-p_{i})^{1-y_{i}} \nonumber \\
&= \prod_{i=1}^{n}\Phi(x_{i}^{T}\beta)^{y_{i}}(1-\Phi(x_{i}^{T}\beta))^{1-y_{i}} \nonumber
\end{align}
$$

우리의 관심 parameter $$\beta$$는 (0,1)로 범위를 한정할 이유가 없어 다음과 같이 정규분포 형태의 사전분포(Prior)를 가정한다.

$$\beta \sim \mathcal{N}(\beta_{0},\Sigma_{0})$$

다만, $$\beta$$에 대한 Prior와 likelihood의 형태가 서로 Conjugate하지 않아 posterior가 편리한 형태로 주어지지 않아 베이지안 추론이 쉽지 않게 된다. 이 문제를 해결하기 위해 Latent Variable을 활용하게 된다.

다음과 같이 Latent Variable($$\mathbf{u}$$)를 고려하면 Gibbs Sampler 등을 이용해 베이지안 추론을 비교적 간단하게 이끌어낼 수 있다. 잠재변수 $$\mathbf{u}$$를 다음과 같이 정의하자.

$$ u_{i} \sim \mathcal{N}(x_{i}^{T}\beta,1) $$

그리고 다음과 같이 $$u_{i}$$ 값에 따라 $$y_{i}$$의 값이 정해진다고 하자.

$$
y_{i} =
\begin{cases}
1 \quad \text{if} \quad u_{i} > 0 \\
0 \quad \text{if} \quad u_{i} \leq 0
\end{cases}
$$

따라서 이항변수 $$\mathbf{y}$$에 대하여 $$y_{i}=1$$인 사건은 연속형 변수 $$u_{i}>0$$인 사건과 동일하며 이를 다음과 같이 정리할 수 있다.

$$p(y_{i}=1)=p(u_{i}>0)=1-\Phi(-x_{i}^{T}\beta)=\Phi(x_{i}^{T}\beta)$$

이 식이 성립하는 이유는 다음과 같다.

$$
\begin{align}
p(y_{i}=1|\mathbf{X}) &= p(u_{i}>0) \\
&= p(x_{i}^{T}\beta+\epsilon > 0) \\
&= p(\epsilon > - x_{i}^{T}\beta) \\
&= p(\epsilon < x_{i}^{T}\beta) \quad \text{by the Symmetry of the Normal distribution}\\
& =\Phi(x_{i}^{T}\beta) \\
& =1-\Phi(-x_{i}^{T}\beta)
\end{align}
$$

$$\mathbf{u}$$는 latent variable이므로 또 다른 parameter로 취급할 수 있고 이 latent variable을 표현하는 likelihood는 다음과 같다.

$$l(\mathbf{u}|\beta,\mathbf{y}) = \prod_{i=1}^{n} p(u_{i}|x_{i}^{T}\beta,1)[I(u_{i}>0,y_{i}=1)+I(u_{i}\leq 0,y_{i}=0)]$$

따라서 latent variable $$\mathbf{u}$$의 Conditional posterior distribution은 다음과 같이 Truncated Normal distribution을 따른다

$$
u_{i}\mid\beta,y_{i} =
\begin{cases}
\mathcal{N}(x_{i}^{T}\beta,1)\cdot\mathcal{I}(u_{i}>0) \quad \text{if} \quad y_{i}=1 \\
\mathcal{N}(x_{i}^{T}\beta,1)\cdot\mathcal{I}(u_{i}\leq 0) \quad \text{if} \quad y_{i}=0
\end{cases}
$$

더불어, $$\beta$$에 대한 conditional posterior distribution을 구할 수 있다.

$$\mathbf{y}=(y_{1},...,y_{n}), \mathbf{u}=(u_{1},...,u_{n}), \mathbf{X}=(x_{1},...,x_{n})$$ 일 때,

$$
\begin{align}
p(\beta|\mathbf{y},\mathbf{u}) &\propto exp[-\frac{1}{2}\{(\mathbf{u}-\mathbf{X}\beta)^{T}(\mathbf{u}-\mathbf{X}\beta)+(\beta-\beta_{0})^{T}\Sigma^{-1}_{0}(\beta-\beta_{0})\}] \\
  &\propto exp[-\frac{1}{2}\{\beta^{T}(\mathbf{X}^{T}\mathbf{X}+\Sigma^{-1}_{0})\beta-2\beta^{T}(\mathbf{X}^{T}\mathbf{u}+\Sigma^{-1}_{0}\beta_{0})\}]
\end{align}
$$

다음과 같이 $$\beta$$에 대한 conditional posterior를 구할 수 있고 그 결과는 아래와 같다.

$$
\begin{align}
\beta|\mathbf{y},\mathbf{u} &\sim \mathcal{N}(\mu_{\pi},\Sigma_{\pi}) \\
\Sigma_{\pi} &= (\mathbf{X}^{T}\mathbf{X}+\Sigma^{-1}_{0})^{-1} \\
\mu_{\pi} &= \Sigma_{\pi}(\mathbf{X}^{T}\mathbf{u}+\Sigma^{-1}_{0}\beta_{0})
\end{align}
$$

이처럼 $$\beta$$와 $$\mathbf{u}$$의 Conditional Posterior distribution은 Sampling하기 편한 형태로 주어지기 때문에 이후의 과정에서 Gibbs Sampler 등을 이용하여 $$\beta$$와 $$\mathbf{u}$$에 대한 Sampling을 수행할 수 있게 된다.

##### 상기 예제에 관련한 코드는 다음의 링크 1. [R코드](https://github.com/seolbluewings/R_code/blob/master/Probit%20Regression.ipynb) 2. [Python코드](https://github.com/seolbluewings/pythoncode/blob/master/6.Probit%20Regression.ipynb) 에서 확인할 수 있습니다.


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [BDA](http://www.stat.columbia.edu/~gelman/book/)


