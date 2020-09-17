---
layout: post
title:  "프로빗 모델(Probit model)"
date: 2019-07-06
author: seolbluewings
categories: Statistics
---

기존의 회귀분석 모델 $$\mathbf{y} = \mathbf{X}\beta + \epsilon$$ 모형은 보통 1. $$\mathbf{X}$$와 $$\mathbf{y}$$ 사이의 선형 관계가 있고  2. $$\mathbf{y}$$가 정규분포를 따른다고 볼 수 있을 때, 활용하는 것이 적절하다.

그러나 현실 세계에서는 이러한 조건에 부합하지 않는 데이터가 많다. $$\mathbf{X}$$와 $$\mathbf{y}$$가 선형관계가 아닌 S자 형태의 관계를 보일 수도 있고 $$\mathbf{y}$$가 정규분포가 아닌 이항분포, 다항분포 등을 따를 수도 있다. 이러한 상황에서는 기존의 선형모형을 활용하는 것이 적절하지 않다.

$$\mathbf{X}$$와 $$\mathbf{y}$$ 사이의 다양한 관계를 표현할 수 있는 일반화 선형 모형(Generalized Linear Model)을 이러한 상황에 활용할 수 있다.

$$\mathbf{y} = h(\mathbf{X}\beta)$$

위와 같은 형태로 연결 함수(Link Function) $$h$$ 를 이용해 일반화 선형 모형을 표현할 수 있다. 가장 빈번하게 활용하는 Link Function은 $$\mathbf{y}$$가 이항분포를 따를 때 사용하는 프로빗(probit) 함수와 로지스틱(logistic) 함수가 있다.

$$
\begin{align}
y &\sim \text{Ber}(p) \nonumber \\
h(p) &= log\left(\frac{p}{1-p}\right) \quad \text{logistic function} \nonumber \\
h(p) &= \Phi^{-1}(p) \quad \text{probit function}
\end{align}
$$

Link Function을 활용해 Probit Model을 계산하는 과정에서 잠재변수(Latent Variable) $$\mathbf{U}$$ 를 이용하게 된다. 그리고 베이지안 방법론은 Latent Variable이 있는 상황에서 강점을 갖는다.

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

다음과 같이 Latent Variable($$\mathbf{U}$$)를 고려하면 Gibbs Sampler 등을 이용해 베이지안 추론을 비교적 간단하게 이끌어낼 수 있다. 잠재변수 $$\mathbf{U}$$를 다음과 같이 정의하자.

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
	Pr(Y=1|X) &= Pr(Y^{*}>0) \\
    &= Pr(X^{T}\beta+\epsilon > 0) \\
    &= Pr(\epsilon > - X^{T}\beta) \\
    &= Pr(\epsilon < X^{T}\beta) \text{  by Symmetry of the normal-dist}\\
    & =\Phi(X^{T}\beta) \\
    & =1-\Phi(-X^{T}\beta)
\end{align}
$$

$$Y_{i}^{*}$$는 관측변수가 아니므로 모수로 취급하여 새로운 likelihood를 구할 수 있으며 이를 표현하면 다음과 같을 것이다.

$$l(y^{*}|\beta,y) = \prod_{i=1}^{n} \phi(y_{i}^{*}|x_{i}^{T}\beta,1)[I(y_{i}^{*}>0,y_{i}=1)+I(y_{i}^{*}\leq 0,y_{i}=0)]$$

여기서 $$\phi(y_{i}^{*}\|x_{i}^{*}\beta,1)$$ 은 $$\mathcal{N}(x_{i}^{T}\beta,1)$$ 분포의 $$y_{i}^{*}$$에서의 pdf 값이다.

다음과 같이 정의하면, $$\beta$$에 대한 conditional posterior distribution을 구할 수 있다.

$$\mathbf{y}=(y_{1},...,y_{n}), \mathbf{y}^{*}=(y^{*}_{1},...,y^{*}_{n}), \mathbf{X}=(x_{1},...,x_{n})$$ 일 때,

$$
\begin{align}
	\pi(\beta|\mathbf{y},\mathbf{y}^{*}) &\propto exp[-\frac{1}{2}\{(\mathbf{y}^{*}-\mathbf{X}\beta)^{T}(\mathbf{y}^{*}-\mathbf{X}\beta)+(\beta-\beta_{0})^{T}\Sigma^{-1}_{0}(\beta-\beta_{0})\}] \\
    &\propto exp[-\frac{1}{2}\{\beta^{T}(\mathbf{X}^{T}\mathbf{X}+\Sigma^{-1}_{0})\beta-2\beta^{T}(\mathbf{X}^{T}\mathbf{y}^{*}+\Sigma^{-1}_{0}\beta_{0})\}]
\end{align}
$$

다음과 같이 $$\beta$$에 대한 conditional posterior를 구할 수 있고 그 결과는 아래와 같다.

$$
\begin{align}
	\beta|\mathbf{y},\mathbf{y}^{*} &\sim \mathcal{N}(\mu_{\pi},\Sigma_{\pi}) \\
    \Sigma_{\pi} &= (\mathbf{X}^{T}\mathbf{X}+\Sigma^{-1}_{0})^{-1} \\
    \mu_{\pi} &= \Sigma_{\pi}(\mathbf{X}^{T}y^{*}+\Sigma^{-1}_{0}\beta_{0})
\end{align}
$$

따라서 잠재변수(Latent Variable) $$Y_{i}^{*}$$의 conditional posterior distribution은 다음과 같이 truncated Normal distribution을 따른다.

$$
y^{*}_{i}|y_{i},x_{i},\beta \sim
\begin{cases}
	\mathcal{N}(x_{i}^{T}\beta,1)\mathcal{I}(y_{i}^{*} \leq 0)\quad \text{if\$$y_{i}=1$$} \\
	\mathcal{N}(x_{i}^{T}\beta,1)\mathcal{I}(y_{i}^{*} > 0) \quad \text{if\$$y_{i}=0$$}
\end{cases}
$$

Truncated Normal distribution에서의 Sampling은 distribution이 얼마나 truncated 되어있는가에 따라 다르며 제한없는 $$\mathcal{N}(\mu,\sigma^{2})$$로부터 난수를 생성한 다음, 구간 (a,b)에 속하는 난수만 취하고 나머지는 버리는 rejection-method를 사용하므로 구간(a,b)의 확률이 작을 경우 비효율적일 수 있다. 그러나 역누적분포기법(Inverse CDF)을 사용하여 그러한 비효율을 막을 수 있다.

임의의 누적분포함수 F에 대하여

$$F(X) \sim U(0,1)$$

임을 이용하여, $$U \sim U(0,1)$$을 생성한 다음, $$X=F^{-1}(U)$$ 변환을 이용해 누적분포함수 F를 갖는 난수 X를 얻는 방법이다. 이를 truncated-Normal distribution, $$\mathcal{N}(\mu,\sigma^{2})\mathcal{I}(a,b)$$에 적용하면, 다음과 같다.

$$
\begin{align}
	\frac{\Phi\left(\frac{X-\mu}{\sigma}\right)-\Phi\left(\frac{a-\mu}{\sigma}\right)}{\Phi\left(\frac{b-\mu}{\sigma}\right)-\Phi\left(\frac{a-\mu}{\sigma}\right)} &= U \\
    U &\sim U(0,1)
\end{align}
$$

$$ X = \Phi^{-1}\left(U\times\Phi\left(\frac{b-\mu}{\sigma}\right)+(1-U)\times \Phi\left(\frac{a-\mu}{\sigma}\right) \right)
$$

#### 예시

30개의 실험대상에 대하여 관측한 결과, X값으로 첫 10개 대상은 0, 다음 10개 대상은 1, 나머지 10개 대상은 2값을 갖는다. y는 0 또는 1값을 갖고 $$ y_{1} \sim y_{3}=1, y_{4}\sim y_{10}=0, y_{11}\sim y_{15}=1, y_{16}\sim y_{20}=0, y_{21}\sim y_{22}=1, y_{23}\sim y_{30}=0 $$이다. 이 자료에 대해 Probit 모형을 적용하고, $$\beta_{0},\beta_{1}$$의 사전 분포로 각 $$\mathcal{N}(0,100)$$을 주자.

$$
\begin{align}
	Y_{i} &\sim Ber(p_{i}) \\
    \Phi^{-1}(p_{i}) &= \beta_{0}+\beta_{1}(x_{i}-\bar{x})
\end{align}
$$

```
library(mvtnorm)
y=rep(c(1,0,1,0,1,0),c(3,7,5,5,2,8))
x=rep(c(0,1,2),c(10,10,10))
x=x-mean(x)
X=cbind(rep(1,30),x)
p=2; n=30
beta0=rep(0,2)
Beta0=diag(100,2)

### initialize

beta=beta0
iter=3000; warm=1000
beta_mat=matrix(0,nrow=iter,ncol=p)
beta_Sig=solve(t(X)%*%X+Beta0)
ystar=rep(0,n)

### truncated normal random generator

trunc_norm=function(mu,sigma,l,u){
  unif=runif(1,0,1)
  trunc_norm=qnorm(unif*pnorm((u-mu)/sigma)+
                    (1-unif)*pnorm((l-mu)/sigma))*sigma+mu
}


### MCMC

for(i in 1:iter+warm){
  ## generate y_star
  for(j in 1:n){
    if(y[j]==1){
      ystar[j]=trunc_norm((X[j,])%*%beta,1,0,Inf)
    } else{
      ystar[j]=trunc_norm((X[j,])%*%beta,1,-Inf,0)
    }
  }
  ## generate beta
  beta_mu=beta_Sig%*%(t(X)%*%ystar+solve(Beta0)%*%beta0)
  beta=rmvnorm(1,beta_mu,beta_Sig)
  if(i>warm){
    beta_mat[i-warm,]=beta
  }
}

beta_hat=c(0,0)
for(i in 1:p){
  beta_hat[i]=mean(beta_mat[,i])
}

par(mfrow=c(2,2))
plot(beta_mat[,1],type="l",xlab="",ylab=expression(paste(beta[0])))
plot(beta_mat[,2],type="l",xlab="",ylab=expression(paste(beta[1])))

plot(density(beta_mat[,1]),xlab=expression(paste(beta[0])),ylab="posterior",main="")
abline(v=beta_hat[1],lty=2)
plot(density(beta_mat[,2]),xlab=expression(paste(beta[1])),ylab="posterior",main="")
abline(v=beta_hat[2],lty=2)
```





