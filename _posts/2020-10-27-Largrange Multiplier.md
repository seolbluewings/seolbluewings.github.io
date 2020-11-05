---
layout: post
title:  "라그랑주 승수(Lagrange Multiplier)"
date: 2020-10-27
author: seolbluewings
categories: 최적화
---

[작성중...]

라그랑주 승수법(Lagrange Multiplier)은 최적화 문제를 풀기위해 고안한 방법이다. 직접적으로 어떠한 문제의 최적점을 찾아내는 방법은 아니고 최적점이 되기위한 조건을 찾는 방법이며 즉, 최적의 해를 갖추기 위한 필요조건을 찾는 방법이라 말할 수 있겠다.

라그랑주 승수법은 한개 이상의 제약조건이 주어진 상황에서 다변수 함수의 극점(최대 또는 최소값을 갖는 지점)을 발견하는데 활용된다.

함수 $$f(x_{1},x_{2})$$의 최대값을 찾는 문제를 가정하자. 그리고 이 수식에 적용되는 제약조건을 $$g(x_{1},x_{2})=0$$ 라고 표현하도록 하자.

이러한 문제를 마주한다면, 우리는 이러한 시도를 해볼 것이다.

$$g(x_{1},x_{2})=0$$ 을 풀어서 $$x_{2} = h(x_{1})$$ 형태로 $$x_{2}$$를 $$x_{1}$$의 함수 형태로 표현한다. 그 다음 이를 $$f(x_{1},x_{2})$$에 대입하여 $$f(x_{1},h(x_{1}))$$ 형태로 바꿀 것이고 미분을 이용해서 이 함수를 $$x_{1}$$ 에 대하여 최대화 한다. 그리고 $$x_{2}^{*} = h(x_{1}^{*})$$ 수식을 활용해서 $$x_{2}$$ 값을 구한다.

이 때, $$x_{2}$$를 $$x_{1}$$에 대한 함수로 표현한 식에 대한 해를 구하는 것이 어려워 이 방법을 활용하기 어려울 수 있다.

이러한 상황에서 라그랑주 승수라고 불리는 parameter $$\lambda$$를 도입하면 문제를 비교적 간단하게 해결할 수 있다.

#### 등식 제약 조건(equality constraint)

D차원의 변수 $$\mathbf{X} = (x_{1},...,x_{D})$$ 가 존재한다고 가정하자. 그리고 제약조건 $$g(\mathbf{X})=0$$ 이 있을 것이다. 이 때 등식이 사용하기 때문에 이를 등식 제약 조건(equality constraint)이라 부른다. 이 제약조건은 D-1차원이다.

> 왜 D-1차원인가? 고등학교 시절 벡터를 다룬 단원에서 평면에 대한 수식을 생각해보자. 3차원 공간에서의 평면에 대한 수식은  $$ ax_{1}+bx_{2}+cx_{3}+d = 0 $$ 였다. 이 수식은 $$g(\mathbf{X})=0$$ 이며, 3차원 공간에서의 데이터 $$\mathbf{X}$$ 에 대한 $$g(\mathbf{X})=0$$ 은 평면 2차원이다. 그래서 데이터 $$\mathbf{X}$$가 D차원일 때, 제약조건 $$g(\mathbf{X})=0$$은 D-1차원일 것이다.

이 때 우리는 2가지 사항을 확인할 수 있다.

- $$g(\mathbf{X})=0$$ 상의 임의의 점 $$X$$에 대하여 $$\bigtriangledown g(\mathbf{X})$$는 $$g(\mathbf{X})=0$$ 과 직교한다.

표면상의 한 점 $$\mathbf{X}$와 $$\mathbf{x+\epsilon}$$ 이 있다고 가정해보자. $$g(\mathbf{X+\epsilon})$$은 다음과 같이 Taylor Series 전개될 수 있다.

$$g(\mathbf{X+\epsilon}) \simeq g(\mathbf{X}) + \mathbf{\epsilon}^{T}\bigtriangledown g(\mathbf{X})$$

$$\mathbf{X}$$와 $$\mathbf{X+\epsilon}$$은 둘 다 제약조건 $$g(\mathbf{X})=0$$ 위에 존재하기 때문에 $$g(\mathbf{X})=g(\mathbf{X+\epsilon})$$ 이다. $$\mathbf{\epsilon} \to 0$$이면 $$ \mathbf{\epsilon}^{T}\bigtriangledown g(\mathbf{X})=0$$ 이다. $$\mathbf{\epsilon}$$은 $$g(\mathbf{X})=0$$ 에 평행하기 때문에 $$ \mathbf{\epsilon}^{T}\bigtriangledown g(\mathbf{X})=0$$ 임을 고려한다면, $$\bigtriangledown g(\mathbf{X})$$는 제약조건 평면에 수직이라는 것을 알 수 있다.

- 최적의 포인트 $$\mathbf{X}^{*}$$에서 $$f(\mathbf{X})$$에서 해당 $$\mathbf{X}^{*}$$에서의 경사 $$\bigtriangledown f(\mathbf{X}^{*})$$에서 제약조건 $$g(\mathbf{X})=0$$에 직교한다.

만약 직교하지 않는다면, $$g(\mathbf{X})=0$$을 따라 더 짧은 거리를 이동시켜 $$f(\mathbf{X})$$ 값을 더 증가시킬 수 있다.

따라서 $$\bigtriangledown f$$와 $$\bigtriangledown g$$는 평행(방향이 같거나 반대방향)하다고 할 수 있다. 따라서 다음을 만족하는 0 아닌 parameter $$\lambda$$가 존재할 것이며 이 $$\lambda$$를 라그랑주 승수라고 부른다.

$$\bigtriangledown f(\mathbf{X}^{*}) + \lambda\bigtriangledown g(\mathbf{X}^{*}) = 0$$

$$\lambda$$ 값은 양수, 음수 모두 가능하다. 여기서 라그랑주 함수를 다음과 같이 도입할 수 있다.

$$L(\mathbf{X},\lambda) \equiv f(\mathbf{X}) + \lambda g(\mathbf{X})$$

이 때, 다음의 식을 생각해볼 수 있다.

$$
\begin{align}
\bigtriangledown_{\mathbf{X}}L &= \bigtriangledown f(\mathbf{X}) + \lambda \bigtriangledown g(\mathbf{X}) = 0 \nonumber \\
\bigtriangledown_{\lambda}L &= g(\mathbf{X}) = 0 \nonumber
\end{align}
$$

즉, 우리는 라그랑주 함수 $$L(\mathbf{X},\lambda)$$를 정의함으로써 제약조건 하의 최적화 문제를 라그랑주 함수의 무제약 최적화 문제로 변환시킬 수 있다. 이제는 $$L(\mathbf{X},\lambda)$$의 임계점을 찾는 문제로 바뀌었고 $$\mathbf{X}$$가 $$D$$차원이라면, 임계점 $$\mathbf{X}^{*}$$와 $$\lambda$$를 찾는 $$D+1$$개의 공식을 얻게 된다.

#### 부등식 제약 조건(inequality constraint)

앞서 우리는 제약조건이 등식인 형태, $$g(\mathbf{X})=0$$ 에서의 함수 $$f$$의 최적화를 살펴보았는데 이번에는 $$g(\mathbf{X}) \geq 0$$ 조건에서 함수 $$f$$를 최대화시키는 경우를 생각해볼 것이다. 이 상황에서 경우의 수는 다음과 같이 2가지다.

1. 제약 조건 하의 함수 $$f$$의 임계점이 $$g(\mathbf{X})>0$$ 지역에 존재하는 경우

2. 제약 조건 하의 함수 $$f$$의 임계점이 $$g(\mathbf{X})=0$$ 지역에 존재하는 경우

1의 경우를 제약 조건이 비활성화(inactive)되었다고 말하며 2의 경우 제약 조건이 활성(active) 되었다고 말한다.

우선 1번 케이스부터 살펴보도록 하자. 이 때, 함수 $$g(\mathbf{X})$$는 어떠한 역할도 하지 않아 단순히 $$\bigtriangledown f(\mathbf{X})=0$$ 만으로 최적의 임계점을 찾게 된다. 이는 기존의 라그랑주 함수에서 $$\lambda=0$$ 인 케이스라고 볼 수 있겠다.

2번 케이스는 앞서 등식 제약 조건에서 맞이했던 상황과 동등하다. $$\lambda \neq 0$$ 인 상황에서 라그랑주 함수의 임계점을 구하는 것이다. 이제는 라그랑주 승수 $$\lambda$$의 부호가 중요하다.



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)


