---
layout: post
title:  "쌍대성과 KKT 조건(Duality & KKT Conditions)"
date: 2020-11-29
author: seolbluewings
categories: 최적화
---

라그랑주 승수법을 등식조건과 부등식 조건이 동시에 주어진 경우로 확장해보자. 함수 $$f(\mathbf{X})$$를 $$g(\mathbf{X})=0$$와 $$h(\mathbf{X})\leq 0$$ 에 대해서 최소화시키는 것을 목적으로 하자. 그렇다면 우리는 다음과 같은 라그랑주 함수를 정의하여 함수를 최소화시킬 포인트 $$\mathbf{X}$$를 발견해내고자 한다.

$$
L(\mathbf{X},\lambda,\mu) = f(\mathbf{X})+\lambda^{T}g(\mathbf{X})+\mu^{T}h(\mathbf{X})
$$

이렇게 라그랑주 함수를 정의한 상황에서 $$L(\mathbf{X},\lambda,\mu)$$ 를 최소화시킬 수 있는 점 $$\mathbf{X}$$를 구하는 방법을 알아보도록 하자. 우리는 무제약 최적화 문제를 만드는 과정에서 새로운 변수 $$\lambda$$와 $$\mu$$를 도입했고 이제 두 변수까지 고려해서 최적점을 찾아야한다. 이 때, 우리는 최적화 이론에서 활용되는 쌍대성(이하 Duality) 이란 개념을 활용할 수 있다.

#### 쌍대성 (Duality)

Duality는 어떠한 최적화 문제가 원초 문제(이하 Primal Problem)와 쌍대 문제(이하 Dual Problem) 관점으로 표현할 수 있다는 것을 의미한다. 여기서 활용할 중요한 특징은 Dual Problem의 상한이 Primal Problem의 하한이라는 것이다. 선형 계획법에서 쌍대성 이론을 적용하는 한가지 예시를 살펴보자. 선형 계획법이란 최적화 문제의 일종으로 주어진 선형조건을 만족시키면서 선형 형태의 Objective function을 최적화하는 문제를 의미한다.

$$
\begin{align}
	\text{min}_{x}c^{T}x & \nonumber \\
    \text{s.t.} \quad Ax &= b \nonumber \\
    Gx &\leq h \nonumber
\end{align}
$$

이는 2가지 제약조건 $$Ax=b$$와 $$Gx \leq h$$를 만족하면서 Objective function $$c^{T}x$$ 를 최소화하는 $$x$$를 찾는 것을 의미한다.

여기서 제약조건에 벡터 $$u$$와 $$v$$를 곱하자. 부등식에 곱하는 벡터 $$v$$는 $$v \geq 0$$으로 이는 벡터 $$v$$의 모든 요소가 0 이상의 값을 갖는걸 의미한다. 두가지 제약조건은 다음과 같이 표현될 수 있다. 부등호 방향이 변하지 않는 것은 $$v$$에 대한 제약조건 때문이다. 

$$
\begin{align}
u^{T}Ax &= u^{T}b \nonumber \\
v^{T}Gx &\leq v^{T}h \nonumber
\end{align}
$$

여기서 양변을 더하고 수식을 정리하면 다음과 같은 결론을 얻을 수 있다.

$$
\begin{align}
	u^{T}Ax + v^{T}Gx &\leq u^{T}b + v^{T}h \nonumber \\
    (u^{T}A+v^{T}G)x &\leq u^{T}b + v^{T}h \nonumber \\
    (-A^{T}u-G^{T}v)^{T}x &\geq -u^{T}b - v^{T}b \nonumber \\
    c^{T}x &\geq -u^{T}b-v^{T}h \nonumber
\end{align}
$$

$$-A^{T}u-G^{T}v = c$$ 라고 보면, 우리는 Primal Problem의 Objective function $$c^{T}x$$의 하한 $$-u^{T}b-v^{T}h $$ 을 얻을 수 있었다.

따라서 $$c^{T}x$$를 최소화시키는 Primal problem 다음의 문제를 해결하는 것과 동등하다. 즉, Primal Problem은 최적의 $$x$$를 찾는 것이며 Dual Problem은 최적의 벡터 $$u,v$$를 찾는 것이다.

$$
\begin{align}
	\text{max}_{u,v} -u^{T}b - v^{T}h & \nonumber \\
    \text{s.t.} \quad v &\geq 0 \nonumber \\
    -A^{T}u-G^{T}v &= c \nonumber
\end{align}
$$



#### KKT 조건 (KKT Conditions)



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)

3. [Duality and KKT Conditions](http://fourier.eng.hmc.edu/e176/lectures/ch3/node15.html)
