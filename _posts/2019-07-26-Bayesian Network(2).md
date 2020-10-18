---
layout: post
title:  "베이지안 네트워크 2(Bayesian Network)"
date: 2019-07-26
author: YoungHwan Seol
categories: 베이지안네트워크
---

일반적으로 K개의 변수를 갖는 joint distribution, $$ p(x_{1},...,x_{K}) $$은 다음과 같이 전개(factorization)이 가능하다.

$$
\begin{align}
	p(x_{1},...,x_{K}) &= p(x_{K}\mid x_{1},...x_{k-1})p(x_{1},...,x_{k-1}) \\
    &= p(x_{K}\mid x_{1},...x_{k-1})p(x_{K-1}\mid x_{1},...,x_{K-2})p(x_{1},...,x_{K-2}) \\
    &= p(x_{K}\mid x_{1},...,x_{K-1})\cdot\cdot\cdot p(x_{2}\mid x_{1})p(x_{1})
\end{align}
$$

위와 같은 경우의 베이지안 네트워크를 fully connected 되었다고 하며, 이는 임의의 두쌍의 노드가 서로 연결되어 있음을 의미한다.

그러나 아래 그림과 같이 일부 링크가 없는 네트워크가 보다 일반적이다. 아래의 그래프는 fully connected가 아니며, $$x_{1}$$에서 $$x_{2}$$로의 링크, $$x_{3}$$에서 $$x_{7}$$으로 가는 링크가 존재하지 않는다. 노드 x의 부모를 y라할 때, y의 값이 주어지면 x는 비후손(child node를 제외한 나머지)과 모두 conditional independence하다는 Markov Assumption에 의해 간략하게 표현할 수 있다. 


![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/conditonal.JPG?raw=true){:width="30%" height="30%"}{: .center}

7개의 확률변수 $$p(x_{1},x_{2},...,x_{7})$$은 다음과 같은 형태로 표현될 수 있다.

$$ p(x_{1},x_{2},...,x_{7})=p(x_{1})p(x_{2})p(x_{3})p(x_{4} \mid x_{1},x_{2},x_{3})p(x_{5}\mid x_{1},x_{3})p(x_{6} \mid x_{4})p(x_{7} \mid x_{4},x_{5}) $$

이를 일반화한 표현은 다음과 같으며, $$pa_{k}$$는 변수 $$x_{k}$$의 parent node 집합을 의미한다.

$$ p(\mathbf{x}) = \prod_{k=1}^{K}p(x_{k}\mid pa_{k})$$

conditional probability를 이용하여 우리는 몇가지 evidence가 주어진 상황에서 관측되지 않은 hidden variable의 conditional probability를 알아낼 수 있다. 베이지안 네트워크에서 확인할 수 있는 전체 확률변수를 $$\mathbf{X}$$라고 하자. 이 때, 우리가 관찰한 확률변수는 $$X_{obs}$$라 표기하고 관측하지 못한 변수들에 대해서는 $$X_{H}$$이라 표현하자. 또한 $$X_{H}$$은 우리가 관심을 가지고 있는(혹은 conditional probability를 알아내고자 하는) 변수 Y와 그마저도 관심이 없는 변수 Z로 나뉠 수 있다. $$ X_{H} = \{Y,Z\} $$

현재 알지 못하고 있으나 관심있는 변수 Y에 대한 conditional distribution은 다음과 같이 구할 수 있다. 우리는 conditional independence를 활용해 joint distribution을 얻을 수 있으므로 conditional distribution을 joint distribution을 향한 방향으로 풀어갈 필요가 있다.

$$
\begin{align}
	p(Y \mid X_{obs}) &= \sum_{z} p(Y,Z=z \mid  X_{obs}) \\
    &= \sum_{z} \frac{p(Y,Z,X_{obs})}{p(X_{obs})} \\
    &= \sum_{z} \frac{p(Y,Z,X_{obs})}{\sum_{y,z}p(Y=y,Z=z,X_{obs})}
\end{align}
$$

이렇게 구한 condtional probability를 활용하여 우리는 관심있는 Hidden Variable Y에 대해 가장 발생 가능한 사건(most probable assignment)을 얻을 수 있다.

베이지안 네트워크(Bayesian Network)에서 사용되는 가장 전형적인 예시를 통해 앞서 논의한 내용들에 대해 다시 한 번 점검하도록 하자.

다음의 그림과 같은 관계가 있다고 하자. 도둑의 침입을 대비하여 알람을 설치하였는데 지진이 발생하는 경우에도 알람이 작동할 가능성이 있다. 도둑이 침입할 확률은 그림에 주어진 바와 같이 0.001이고 지진이 발생할 확률은 0.002다. 알람이 작동할 확률은 $$p(A \mid B,E)$$로 값이 그림과 같이 주어져있다.

알람이 울리면 John과 Mary는 전화를 해주기로 합의하였고 알람이 울렸을 때, John이 전화할 확률 $$p(J \mid A)$$와 알람이 울렸을 때, Mary가 전화할 확률 $$P(M \mid A)$$는 그림과 같다.

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/burglary_example.PNG?raw=true){: .center}

앞서 언급한 바처럼, 우리는 먼저 full joint probability를 계산하고 이제 marginalized하여 일부 변수에 대한 partial joint probability를 얻는다. 이후 관심변수에 대한 conditional probability를 계산하여 값을 결정한다.

다음의 partial joint probability는 어떻게 구할 수 있을까?

$$ p(A=True, B=True, MC= True)$$

먼저 full joint probability를 구하고 이를 marginalized한다.

$$
\begin{align}
	p(A,B,MC) &= \sum_{JC}\sum_{E} p(A,B,E,MC,JC) \\
    &= \sum_{JC}\sum_{E} p(JC \mid A)p(MC \mid A)p(A \mid B,E)p(E)p(B) \\
    &= p(B)p(MC \mid a) \sum_{JC}p(JC \mid A) \sum_{E}p(A \mid B,E)p(E)
\end{align}
$$

따라서 $$MC=True, A=True$$일 때, 도둑이 들었을 확률은 다음과 같이 구할 수 있을 것이다.

$$
\begin{align}
	p(B \mid A,MC) &= \frac{p(A,B,MC)}{p(A,MC)} \\
    &= \frac{p(B)p(MC \mid a) \sum_{JC}p(JC \mid A) \sum_{E}p(A \mid B,E)p(E)}{p(MC\mid A)p(A)}
\end{align}
$$

#### 확률 추론

![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayes%20net.png?raw=true){:width="80%" height="80%"}

위와 같은 그림이 있다고 하자. 그림에서 parent node의 결과를 알면, child node의 값은 숫자을 확인함으로써 구할 수 있다. 폐렴이나 폐암은 아닌 상태에서 피로를 느낄 확률은 $$p(fatigue \mid bronchitis,no-cancer)=0.1$$이다.

앞서 소개했던 D-seperation이 중요한 이유는 이러한 계산을 진행하는 과정에서 계산량을 줄여준다는 장점이 있기 때문이다. 만약 우리가 $$p(fatigue, positive \mid no-cancer)$$에 대해 알고 싶어한다고 하자.

폐암에 대한 정보가 주어진 상황에서 Common-parent 모델은 conditional independent하므로 원하는 분포를 $$p(fatigue \mid no-cancer)$$와 $$p(positive \mid no-cancer)$$의 곱으로 나눌 수 있다.

그러나 보통 X레이 결과가 양성일 때, 폐암일 확률과 같은 케이스에 더 많은 관심을 갖는다. child node일수록 parent node보다 더 관측가능하기 때문에 실질적으로 필요한 추론은 아래서 위로 거꾸로 올라가는 방향이다.

X-레이 결과가 양성일 때, 폐암일 확률 $$p(cancer \mid positive)$$는 다음의 과정을 통해 구할 수 있다.

$$
\begin{align}
	p(cancer \mid positive) &= \frac{p(positive \mid cancer)p(cancer)}{p(positive)} \\
    p(cancer) &= p(cancer \mid smoking)p(smoking) + p(cancer \mid no-smoking)p(no-smoking) \\
    &= 0.03 \times 0.2 + 0.00005 \times 0.8 = 0.00604 \\
    p(positive) &= p(positive \mid cancer)p(cancer) + p(positive \mid no-cancer)p(no-cancer) \\
    &= 0.6 \times 0.00604 + 0.02 \times 0.99396 = 0.0235 \\
    p(cancer \mid positive) &= \frac{0.6 \times 0.00604}{0.0235} = 0.154
\end{align}
$$


![BN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/bayesnet%20example.png?raw=true){:width="40%" height="30%"}

다음의 예시에서 $$\mathbf{x}=x_{1}$$일 때, $$\mathbf{w}=w_{1}$$일 확률은 다음과 같이 구할 수 있다.

$$
\begin{align}
	p(z_{1} \mid x_{1}) &= p(z_{1} \mid y_{1})p(y_{1} \mid x_{1}) + p(z_{1} \mid y_{2})p(y_{2} \mid x_{1}) \\
    &= 0.7*0.9 + 0.4*0.1 = 0.67 \\
    p(z_{2} \mid )x_{1} &= 0.33 \\
    p(w_{1} \mid x_{1}) &= p(w_{1} \mid z_{1})p(z_{1}\mid x_{1}) + p(w_{1} \mid z_{2})p(z_{2}\mid x_{1}) \\
    &= 0.5*0.67 + 0.6*0.33 = 0.533
\end{align}
$$

반대로 $$\mathbf{w}=w_{1}$$ 임을 아는 상황에서 $$\mathbf{x}=x_{1}$$일 확률 $$p(x_{1}\mid w_{1})$$을 구하는 과정은 다음과 같다.

$$
\begin{align}
	p(x_{1}\mid w_{1}) &= \frac{p(w_{1} \mid x_{1})p(x_{1})}{p(w_{1})} \\
    p(y_{1}) &= p(y_{1}\mid x_{1})p(x_{1}) + p(y_{1}\mid x_{2})p(x_{2}) \\
    &= 0.9*0.4 + 0.8*0.6 \\
    p(y_{1}) &= 0.84 \\
    p(z_{1}) &= p(z_{1} \mid y_{1})p(y_{1}) + p(z_{1}\mid y_{2})p(y_{2}) \\
    &= 0.7*0.84 + 0.4 * 0.16 \\
    p(z_{1}) &= 0.652 \\
    p(w_{1}) &= p(w_{1} \mid z_{1})p(z_{1}) + p(w_{1}\mid z_{2})p(z_{2}) \\
    &= 0.5 * 0.652 + 0.6 *0.348 = 0.5348 \\
    \frac{p(w_{1} \mid x_{1})p(x_{1})}{p(w_{1})} &= \frac{0.533*0.4}{0.5348} = 0.3987
\end{align}
$$

위와 같은 해결방식은 NP-hard 하여 대안으로 정확성은 다소 낮지만 근사적인 해를 구하는 방향으로 진행할 수 있다.

만약 $$p(y_{1})$$에 대해 알고 싶다면 이 값을 근사하는 알고리즘은 다음과 같을 것이다.

~~~
import numpy as np
import pandas as pd

h=0
m=1000
x1=np.zeros(m); x2=np.zeros(m); y1=np.zeros(m); y2=np.zeros(m)
for iter in range(0,m):
    x=np.random.rand(1)
    if x <= 0.4:
        x1[iter]=x
        y=np.random.rand(1)
        if y<= 0.9:
            y1[iter]=y
        else:
            y2[iter]=y
    else:
        x2[iter]=x
        y=np.random.rand(1)
        if y<=0.8:
            y1[iter]=y
        else:
            y2[iter]=y
    if y1[iter] != 0:
        h=h+1

print(h/m)
~~~
이 결과는 m값의 크기에 따라 변동하며, m값이 커질수록 직접 계산해낸 값인 $$p(y=y_{1})=0.84$$에 가까워진다.



