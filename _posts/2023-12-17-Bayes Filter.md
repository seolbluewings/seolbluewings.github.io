---
layout: post
title:  "Bayes Filter"
date: 2023-12-17
author: seolbluewings
categories: Statistics
---

[작성중...]

Bayes Filter는 Kalman Filter, Particle Filter의 기본이 되는 개념으로 Bayes Rule을 따르는 재귀적인 필터이다. Bayes Filter는 prior와 likelihood를 통해서 posterior를 구하고 이 posterior가 다음 단계의 prior로 작용한다는 점에서 재귀적이라 표현한다.

Bayes Filter에 사용되는 notation을 활용하여 표현하면 다음과 같다.

초기시간 1부터 시간 t까지 측정센서 값의 시퀀스를 $$ \mathbf{z} =\{z_{1},z_{2},...,z_{t}\} $$, 제어입력의 시퀀스를 $$ \mathbf{u} =\{u_{1},u_{2},...,u_{t}\} $$ 로 정의하고 이 값이 주어졌을 때, 이를 조건부로 갖는 상태(state) 변수 $$ \mathbf{x} =\{x_{1},x_{2},...,x_{t}\} $$ 의 conditional pdf를 구하는 것이라 할 수 있다.

$$ \text{bel}(x_{t}) = p(x_{t}\vert z_{1:t},u_{1:t}) $$

이는 1부터 t시점까지의 센서값(z), 제어입력값(u)을 이용해서 t시점의 상태값 $$x_{t}$$를 확률적으로 추정하는 작업으로 $$ \text{bel}(x_{t}) $$ 값은 시간 t에서의 belief state 값이라고 부른다.

Bayes Filter의 알고리즘은 다음의 프로세스를 따른다.

```
\text{for all} $$x_{t}$$ do :
	\bar{\text{bel}}(x_{t}) = \int p(x_{t}\vert u_{t},x_{t-1})\text{bel}(x_{t-1})dx
    \text{bel}(x_{t}) = \eta p(z_{t}\vert x_{t})\bar{\text{bel}}(x_{t})
\text{end for}
\text{return} \text{bel}(x_{t})
```

이 Bayes Filter 프로세스는 다음과 같이 해석할 수 있다.

우선 첫번째 줄은 제어 업데이트(control update) 또는 예측(prediction)이라 불리는 단계로 제어값을 활용하여 상태값을 업데이트하는 작업이다. $$\text{bel}(x_{t-1})$$은 t-1시점의 상태값을 의미하며 $$p(x_{t}\vert u_{t},x_{t-1})$$ 값은 t-1시점의 상태값과 t시점의 제어값이 주어진 상황에서의 현재 t시점의 상태값의 확률 분포로 이 2가지 값을 활용하여 현재의 상태값을 예측하는 행위라 볼 수 있다.




두번째 줄은 측정 업데이트(measurement update) 또는 보정(correction)으로 불리며 이 작업은 센서값을 통해 측정한 값을 활용하여 prediction한 값을 보정한다.







Bayes Filter는 2가지 가정을 따른다.
1. 직전 시점의 상태값과 제어값을 이용해서 현재의 상태를 예측함
2. 측정값 $$\mathbf{z}_{t}$$는 상태변수 $$\mathbf{x}_{t}$$에만 의존함

이 2가지 가정으로 인해 다음의 식이 Bayes Filter 프로세스를 정의하는 과정에서 활용될 수 있다.

$$
\begin{align}
p(x_{t}\vert x_{1:t-1},z_{1:t},u_{1:t}) &= p(x_{t}\vert x_{t-1},u_{t-1}) \\ \newline
p(z_{t}\vert x_{1:t},z_{1:t-1},u_{1:t}) &= p(z_{t}\vert x_{t})
\end{align}
$$







[작성중....]




Label Spreading은 일부 데이터의 label이 존재하지 않는 데이터에서 semi-supervised learning 을 수행하는 모델 중 하나로 label이 주어진 일부 데이터셋의 정보만을 가지고 그래프 이론을 바탕으로 데이터에 label을 부여하는 작업을 수행한다. label이 존재하는 데이터로부터 label에 대한 정보가 label이 존재하지 않는 데이터로 퍼져가는 모델이라 볼 수 있다.

![label](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/label_spreading.png?raw=true){:width="90%" height="80%"}{: .aligncenter}

Label Spreading 모델은 유유상종이라는 굉장히 직관적인 가정에서 출발한다. 나 자신을 데이터 분포상 하나의 node로 정의할 때, 나의 node 주변에 있는 node들의 label은 나와 같은 label일 것이라 추정할 수 있다.

이는 준지도 학습(semi-supervised learning)의 기본 가정인 consistency와도 일치하는 부분이다. Label Spreading은 가까운 node들은 같은 label을 가져야한다는 local consistency와 같은 cluster에 있는 node들 끼리는 같은 label을 가져야한다는 global consistency 모두 만족한다.


만약 node i가 그룹 B라는 label에 속하는데 node i의 주변에 위치한 node들이 그룹 A란 label을 가지고 있다면, node i의 그룹 정보를 B에서 A로 수정하는 것이 Label Spreading 모델이다.

Label Spreading은 네트워크 상에서 긴밀하게 연결된 노드들의 하위 집합을 도출하는 네트워크 분석 방식 중 하나인 커뮤니티 탐지(Community Detection) 방법론 중에서 별도 지표 계산이 필요하지 않아 속도가 빠르다는 장점도 있다.

Label Spreading 알고리즘은 다음의 프로세스를 통해 진행된다.

1. 모든 node는 각자의 label을 가진 상태에서 시작한다
2. node의 순서를 random하게 배치한 리스트를 작성하고
3. 리스트 순서대로 node를 선택한 후, 해당 node의 주변 node가 지닌 label 중에서 가장 빈도가 높은 label로 해당 node의 label을 변경함
4. 모든 node가 최대빈도 label을 유일하게 가질 때까지 2,3의 과정을 반복

node의 최대빈도 label이 여러개 존재한다는 것은 이 node가 여러 커뮤니티의 중간에 끼어있다는 것을 의미하고 최대빈도 label이 유일하다는 것은 근처 데이터로 퍼진 label이 안정적 경계 상태에 도달했다는 것을 의미한다고 볼 수 있다.

이 알고리즘을 반복할수록 node i의 근처 영역에서 입지가 약한 label이 소멸하고 node i 근처에서의 밀도가 높은 label 중심으로 label이 재설정 된다.

이를 조금 더 수학 느낌나게 표현하면 아래와 같다.

데이터셋 $$\mathbf{X} = \{ x_{1},x_{2},...,x_{l},x_{l+1},...,x_{n}\}$$이 있을 때(여기서 $$x_{l}$$ 까지는 label이 있는 데이터고 $$x_{l+1}$$부터 $$x_{n}$$까지는 label 없는 데이터이다) , 이를 c차원의 Label $$ L = \{1,...,c\} $$ 로 mapping 되는 함수 $$\mathbf{F}$$를 Label Spreading 모델이라 볼 수 있다.

$$x_{i}$$의 label $$y_{i}$$는 $$y_{i} \in L $$ 을 만족하며 $$ y_{i} = \text{argmax}_{j \leq c}F_{ij} $$ 이다.

분류모델 $$\mathbf{F}$$는 데이터 집합의 각각의 데이터간의 상대적 거리를 기준으로 그래프로 연결한다. 그래프의 구조를 나타내는 대칭 행렬 $$\mathbf{W}$$ 를 정의할 수 있게 되고 행렬의 각 원소인 $$w_{ij}$$ 값이 클수록 $$x_{i}$$와 $$x_{j}$$의 관계성이 높아서 동일한 Label에 할당 된다고 볼 수 있다.

Label Spreading은 이 그래프 행렬 $$\mathbf{W}$$를 Laplacian Normalize 처리한 $$ \mathbf{S} = \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2} $$ 를 활용한다. 여기서 $$D_{ii} = \sum_{j} w_{ij} $$인 대각행렬이다. (Label Propagation의 경우는 행렬 $$\mathbf{W}$$를 Random Walk Laplacian 처리하여 작업한다)

````
label_prop_model = LabelSpreading(kernel = 'knn'
                                  , n_neighbors = 5
                                  , alpha = 0.2
                                  , max_iter = 30
                                  , tol = 0.001)
````

Label Spreading 모델 Python 라이브러리의 hyperparameter에서 볼 수 있듯이 이 알고리즘은 Label 부여를 반복 수행한다. 이 과정은 아래와 같이 표현될 수 있다. 이 값이 수렴할 때까지 반복하며 $$\alpha$$ 값은 hyperparater 값이다.

이는 인접 Label의 값을 어느 정도의 강도로 활용하여 값을 업데이트 하는가?를 의미하는 hyperparameter이다. 이 값이 0이면 최초의 label을 계속 유지하며, 1이면 항상 값을 새롭게 할당되는 Label로 업데이트 한다고 볼 수 있다.

$$ F(t+1) = \alpha \mathbf{S} F(t) + (1-\alpha)\mathbf{Y}$$

함수 F의 수렴 결과를 $$F^{*}$$라고 할 때, $$y_{i}$$의 Label은 $$ y_{i} = \text{argmax}_{j\leq c} F^{*}_{ij} $$ 의 값을 갖는다고 할 수 있다.


Label Spreading에 대한 간략한 코드는 다음의 [링크](https://github.com/seolbluewings/python_study/blob/master/01.study/Label_Spreading.py)에서 확인 가능합니다.


#### 참조 문헌
1. [Label Propagation Algorithm](https://pizzathief.oopy.io/label-propagation-algorithm) <br>
2. [[네트워크이론] Label propagation algorithm for community detection](https://mons1220.tistory.com/168)

