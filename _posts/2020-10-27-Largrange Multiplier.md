---
layout: post
title:  "라그랑주 승수(Lagrange Multiplier)"
date: 2020-10-27
author: seolbluewings
categories: 최적화
---

라그랑주 승수법(Lagrange Multiplier)은 최적화 문제를 풀기위한 방법이다. 직접적으로 어떠한 문제의 최적점을 바로 찾아내는 방법은 아니라 최적점이 되기위한 조건을 찾는 방법이다. 즉, 최적의 해를 갖추기 위한 필요조건을 찾는 방법이라 말할 수 있겠다.

라그랑주 승수법은 한개 이상의 제약조건이 주어진 상황에서 다변수 함수의 극점(최대 또는 최소값을 갖는 지점)을 발견하는데 활용된다.

라그랑주 승수는 $$d$$개의 변수, $$k$$개의 제약조건의 최적화 문제를 $$d+k$$개 변수의 무제약 최적화 문제로 전환시켜 해결하는 방법이다. 이렇게 문제를 전환시키는 상황에서 라그랑주 승수라고 불리는 parameter $$\lambda$$를 도입하면 문제를 비교적 간단하게 해결할 수 있다.

#### 등식 제약 조건(equality constraint)

D차원의 변수 $$\mathbf{X} = (x_{1},...,x_{D})$$ 가 존재한다고 가정하자.

그리고 제약조건 $$g(\mathbf{X})=0$$ 이 있을 것이다. 이 때 등식이 사용하기 때문에 이를 등식 제약 조건(equality constraint)이라 부른다. 이 제약조건은 D-1차원이다.

> 왜 D-1차원인가? 고등학교 시절 벡터를 다룬 단원에서 평면에 대한 수식을 생각해보자. 3차원 공간에서의 평면에 대한 수식은  $$ ax_{1}+bx_{2}+cx_{3}+d = 0 $$ 였다. 이 수식은 $$g(\mathbf{X})=0$$ 이며, 3차원 공간에서의 데이터 $$\mathbf{X}$$ 에 대한 $$g(\mathbf{X})=0$$ 은 평면 2차원이다. 비슷한 원리로 데이터 $$\mathbf{X}$$가 D차원일 때, 제약조건 $$g(\mathbf{X})=0$$은 D-1차원일 것이다.

어떠한 좌표 $$\mathbf{X}^{*}$$를 찾으려고 한다. 이 $$\mathbf{X}^{*}$$는 $$g(\mathbf{X})=0$$이면서 동시에 $$f(\mathbf{X})$$의 최대값이어야 한다.즉, $$g(\mathbf{X})=0$$ 상에서 $$f(\mathbf{X})$$의 최대값을 찾는 포인트를 찾는 것이다. 이 때 우리는 다음과 같은 2가지 사항을 확인할 수 있다.

- $$g(\mathbf{X})=0$$ 상의 임의의 점 $$X$$에 대하여 $$\bigtriangledown g(\mathbf{X})$$는 $$g(\mathbf{X})=0$$ 과 직교한다.

표면상의 한 점 $$\mathbf{X}$$와 $$\mathbf{X+\epsilon}$$ 이 있다고 가정해보자. $$g(\mathbf{X+\epsilon})$$은 다음과 같이 Taylor Series 전개될 수 있다.

$$g(\mathbf{X+\epsilon}) \simeq g(\mathbf{X}) + \mathbf{\epsilon}^{T}\bigtriangledown g(\mathbf{X})$$

$$\mathbf{X}$$와 $$\mathbf{X+\epsilon}$$은 둘 다 제약조건 $$g(\mathbf{X})=0$$ 위에 존재하기 때문에 $$g(\mathbf{X})=g(\mathbf{X+\epsilon})$$ 이다. $$\mathbf{\epsilon} \to 0$$이면 $$ \mathbf{\epsilon}^{T}\bigtriangledown g(\mathbf{X})=0$$ 이다. $$\mathbf{\epsilon}$$은 $$g(\mathbf{X})=0$$ 에 평행하기 때문에 $$ \mathbf{\epsilon}^{T}\bigtriangledown g(\mathbf{X})=0$$ 임을 고려한다면, $$\bigtriangledown g(\mathbf{X})$$는 제약조건 평면에 수직이라는 것을 알 수 있다.

- 최적의 포인트 $$\mathbf{X}^{*}$$에서 $$f(\mathbf{X})$$는 해당 포인트 $$\mathbf{X}^{*}$$에서의 경사 $$\bigtriangledown f(\mathbf{X}^{*})$$에서 제약조건 $$g(\mathbf{X})=0$$과 직교한다.

만약 직교하지 않는다면, $$g(\mathbf{X})=0$$을 따라 더 짧은 거리를 이동시켜 $$f(\mathbf{X})$$ 값을 더 증가시킬 수 있다. 이 부분에 대해서는 시각적으로 한번 확인하는 것이 확실하게 와닿는데 이 [블로그](https://m.blog.naver.com/PostView.nhn?blogId=lyb0684&logNo=221332307807&proxyReferer=https:%2F%2Fwww.google.com%2F)에서 이를 시각적으로 이 사실을 보여준다.

따라서 $$\bigtriangledown f$$와 $$\bigtriangledown g$$는 평행(방향이 같거나 반대방향)하다고 할 수 있다. 따라서 다음을 만족하는 0 아닌 parameter $$\lambda$$가 존재할 것이며 이 $$\lambda$$를 라그랑주 승수라고 부른다.

$$\bigtriangledown f(\mathbf{X}^{*}) + \lambda\bigtriangledown g(\mathbf{X}^{*}) = 0$$

$$\lambda$$ 값은 양수, 음수 모두 가능하다. 여기서 라그랑주 함수를 다음과 같이 정의할 수 있다.

$$L(\mathbf{X},\lambda) \equiv f(\mathbf{X}) + \lambda g(\mathbf{X})$$

이 때, $$\mathbf{X}$$와 $$\lambda$$에 대한 편미분을 통해 다음의 식을 도출하고 이를 계산하여 최적의 $$\mathbf{X}$$와 $$\lambda$$를 구할 수 있다.

$$
\begin{align}
\bigtriangledown_{\mathbf{X}}L &= \bigtriangledown f(\mathbf{X}) + \lambda \bigtriangledown g(\mathbf{X}) = 0 \nonumber \\
\bigtriangledown_{\lambda}L &= g(\mathbf{X}) = 0 \nonumber
\end{align}
$$

즉, 우리는 라그랑주 함수 $$L(\mathbf{X},\lambda)$$를 정의함으로써 제약조건 하의 최적화 문제를 라그랑주 함수의 무제약 최적화 문제로 변환시킬 수 있다. 이제는 $$L(\mathbf{X},\lambda)$$의 임계점을 찾는 문제로 바뀌었고 $$\mathbf{X}$$가 $$D$$차원이라면, 임계점 $$\mathbf{X}^{*}$$와 $$\lambda$$를 찾는 $$D+1$$개의 공식을 얻게 된다.

#### 부등식 제약 조건(inequality constraint)

앞서 우리는 제약조건이 등식인 형태, $$g(\mathbf{X})=0$$ 에서의 함수 $$f$$의 최적화를 살펴보았는데 이번에는 $$g(\mathbf{X}) \geq 0$$ 조건에서 함수 $$f$$를 최대화시키는 경우를 생각해볼 것이다. 이 상황에서 경우의 수는 다음과 같이 2가지다.

1. 제약 조건 하의 함수 $$f$$의 최적 포인트 $$\mathbf{X}^{*}$$ 가 $$g(\mathbf{X})>0$$ 지역에 존재하는 경우

2. 제약 조건 하의 함수 $$f$$의 최적 포인트 $$\mathbf{X}^{*}$$ 가 $$g(\mathbf{X})=0$$ 지역에 존재하는 경우

1의 경우를 제약 조건이 비활성화(inactive)되었다고 말하며 2의 경우 제약 조건이 활성(active) 되었다고 말한다.

우선 1번 케이스부터 살펴보도록 하자. 이 때, 함수 $$g(\mathbf{X})$$는 어떠한 역할도 하지 않아 단순히 $$\bigtriangledown f(\mathbf{X})=0$$ 만으로 최적의 포인트를 찾게 된다. 이는 기존의 라그랑주 함수에서 $$\lambda=0$$ 인 케이스라고 볼 수 있겠다.

$$\bigtriangledown_{\mathbf{X}}L = \bigtriangledown f(\mathbf{X}) = 0 $$

2번 케이스는 앞서 등식 제약 조건에서 맞이했던 상황과 동등하다. $$\lambda \neq 0$$ 인 상황에서 라그랑주 함수의 임계점을 구하는 것이다. 이제는 라그랑주 승수 $$\lambda$$의 부호가 중요하다. 함수 $$f(\mathbf{X})$$는 기울기 $$\bigtriangledown f(\mathbf{X})$$가 $$g(\mathbf{X})>0$$ 이 지향하는 방향으로 존재할 때만 최대값을 갖을 것이다.

![LARGRANGE](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Largrange.png?raw=true){:width="70%" height="70%"}

수식 자체는 $$\lambda \neq 0$$ 이므로 이전과 동등하지만, 주의할 사항이 생긴다. 최적의 포인트 $$\mathbf{X}^{*}$$라 할 때, $$\bigtraiangledown f(\mathbf{X}^{*})$$ 와 $$\bigtriangledown g(\mathbf{X}^{*})$$ 는 서로 방향이 반대여야한다. 이에 대해서는 위의 이미지 오른쪽을 통해 확인할 수 있다. $$\bigtriangledown g(\mathbf{X})$$는 $$g(\mathbf{X})$$가 지향하는 방향으로 존재할 것이다.

$$\bigtriangledown f(\mathbf{X}) = \lambda\bigtriangledown g(\mathbf{X})\quad \lambda < 0$$

함수 $$f(\mathbf{X})$$의 최적화 방향(최대/최소)과 제약조건의 부등식 조건에 따라 다음과 같이 케이스를 분류할 수 있다.

|CASE|$$g(\mathbf{X} \geq 0$$|$$g(\mathbf{X} \leq 0$$|
|:---:|:---:|:---:|
|$$\text{max} f(\mathbf{X})$$|$$\bigtriangledown f(\mathbf{X}) = \lambda\bigtriangledown g(\mathbf{X}),\lambda <0$$|$$\bigtriangledown f(\mathbf{X}) = \lambda\bigtriangledown g(\mathbf{X}),\lambda >0$$|
|$$\text{min} f(\mathbf{X})$$|$$\bigtriangledown f(\mathbf{X}) = \lambda\bigtriangledown g(\mathbf{X}),\lambda >0$$|$$\bigtriangledown f(\mathbf{X}) = \lambda\bigtriangledown g(\mathbf{X}),\lambda <0$$|


1,2번 조건 모두 $$\lambda g(\mathbf{X}) =0$$ 인 것은 분명하다. 따라서 $$f(\mathbf{X})$$를 $$g(\mathbf{X})\geq 0$$ 조건에서 최대화시키는 문제는 라그랑주 함수 $$L(\mathbf{X},\lambda) \equiv f(\mathbf{X}) + \lambda g(\mathbf{X})$$ 를 다음의 조건 아래에서 $$\mathbf{X}$$ 와 $$\lambda$$ 에 대해 최적화시키는 것으로 변형시킬 수 있다. 이 조건들을 KKT(Karush-Kuhn-Tucker) 조건이라 부른다.

$$
\begin{align}
g(\mathbf{X}) &\geq 0 \nonumber \\
\lambda &\geq 0 \nonumber \\
\lambda g(\mathbf{X}) &= 0 \nonumber
\end{align}
$$


#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)


