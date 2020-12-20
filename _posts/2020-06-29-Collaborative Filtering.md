---
layout: post
title:  "Concept of Collaborative Filtering"
date: 2020-06-29
author: seolbluewings
categories: 추천시스템
---

[작성중]

우리는 일상 생활 속에서 추천 시스템(Recommendation System)을 자주 접하게 된다. 영화를 선택하는 과정에서 플랫폼(왓챠 또는 넷플릭스)은 나에게 잘 맞을 것 같은 영화를 제시해준다. 인터넷에서 물건을 구매하는 과정에서도 내가 관심있을 것으로 여겨지는 상품을 추천 받는다. 일상 생활 속에서 수많은 추천 시스템을 접하고 있기 때문에 우리는 이 추천 시스템이 어떤 논리와 이론 속에서 구현되고 있는지 알고 있을 필요가 있다.

추천 시스템은 크게 2가지로 구분된다. 1. 콘텐츠 기반(Content-Based) 방법과 2. 협업 필터링(Collaborative Filtering)이 추천 시스템의 2가지 대분류다. 이 글에서는 협업 필터링에 대해서 알아보고자 한다. 협업 필터링은 다른 추천 시스템들보다 일반적으로 더 좋은 성능을 가진 것으로 알려져 있다.

#### 콘텐츠 필터링(Content Filtering)

콘텐츠 필터링은 사용자 또는 영화에 대한 프로필을 만드는 것으로 시작된다. 예를 들면, 영화에 대한 프로필은 영화의 장르, 출연배우, 박스오피스 순위 등이 될 수 있다. 사용자 프로필은 가입자가 서비스 가입 과정에서 제출한 설문조사(연령, 선호 장르 등...) 같은 것들이 될 수 있다.

이렇게 수집된 프로필을 바탕으로 사용자와 영화를 연결지을 수 있다. 만약 내가 실화 기반의 영화를 좋아한다고 설문에 응답했다면, 알고리즘은 나에게 실화 기반의 영화 범주에 포함되는 스포트라이트, 빅쇼트 같은 영화들을 추천하게 된다.

#### 협업 필터링(Collaborative Filtering)

콘텐츠 필터링은 영화나 사용자에 대한 명시적 프로필(explicit profiles)을 만들어냈다. 그러나 오로지 과거의 평점(product ratings)에 의존하는 (상거래의 경우 거래 내역) 알고리즘도 존재한다. 이것이 바로 협업 필터링(Collaborative Filtering)이다.

협업 필터링은 사용자와 영화 간 상호 의존성을 분석하고 이를 바탕으로 새롭게 사용자와 영화의 연결성 정도를 측정한다. 대표적인 협업 필터링 방식은 바로 잠재 요인 모델(latent factor model)이다.

잠재 요인 모델은 평점 패턴을 통해 20~100가지 요인(factor)를 추론하고 이를 바탕으로 사용자와 영화의 특징지어 평점을 설명해보고자 하는 시도다. 여기서 이야기하는 요인이란 것은 액션의 양, 주제의 깊이 등과 같이 완벽하게 정의될 수 없는 차원을 의미한다.

![CF](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Latent_Factor.PNG?raw=true){:width="70%" height="70%"}{: .center}

그림과 같이 영화가 2차원(여성지향적vs남성지향적 & 생각을 요하는vs즐기는 용도)속에 존재한다고 생각해보자. 각 영화와 사용자는 이 4분면 내에서 어느 지점에 각자의 위치를 갖게 된다.

이 때, 사용자의 예상 평점은 영화의 평균 평점은 그래프 상에서 영화의 위치와 사용자 위치의 내적과 같다. 예를 들면, Gus는 덤앤 더머를 좋아할 것이고 컬러 퍼플을 싫어할 것이며, 브레이브 하트에 대해서는 평균을 살짝 상회하는 평점을 줄 것이라는걸 기대해볼 수 있다.

#### 행렬 분해(Matrix Factorization)

행렬 분해(Matrix Factorization)을 통해 우리는 잠재 요인 분석을 실행할 수 있다. 앞서 언급한 것처럼 평점 패턴에서 발견할 수 있는 잠재 요인으로 구성된 벡터(latent factor vector)를 사용하게 된다. 영화와 잠재요인, 사용자와 잠재요인 간의 일치성이 높을수록 추천으로 이어진다.

행렬 분해 과정에서 우리는 잠재 요인의 차원 $$f$$ 내에서 사용자와 잠재 요인, 영화와 잠재 요인을 매핑(mapping)시킨다. 그리고 이 둘의 내적을 통해 사용자와 영화 간의 상호작용 정도를 도출하게 된다.

각 영화는 벡터 $$q_{i} \in \mathbb{R}^{f}$$ 로 표현되며, 각 사용자는 벡터 $$p_{u} \in \mathbb{R}^{f}$$로 표현된다.

벡터 $$q_{i}$$는 영화 $$i$$가 각각의 잠재 요인이 긍정(positive)인지, 부정(negative)인지 표현할 수 있다.

벡터 $$p_{u}$$는 해당 잠재 요인과 상관 관계가 높은 영화에 대해서 사용자가 관심을 가질지 여부를 마찬가지로 긍정, 부정으로 표현하게 된다. 따라서 결국 이 둘의 내적 $$q_{i}^{T}p_{u}$$는 영화에 대한 사용자의 관심을 나타내는 값, 즉 예상 평점이라 볼 수 있다.

$$\hat{r}_{ui} = q_{i}^{T}p_{u}$$

이 모델은 특이값 분해(Singular Value Decomposition, SVD)와 관련 있다. SVD는 모든 $$m \times n$$ 크기 행렬에 적용 가능하며, 이 SVD를 행(row)이 $$m$$개, 열(column)이 $$n$$개인 행렬에 영화 평점 행렬 $$R$$에 적용할 수 있다. $$k < \text{min}\{m,n\}$$ 이라하고 k개의 고유값(eigen-value)가 존재한다고 하자.

$$
\begin{align}
R &= U \Sigma V^{T} \nonumber \\
&= U \Sigma^{0.5}\Sigma^{0.5} V^{T} \nonumber \\
&= Q P
\end{align}
$$

여기서 Q는 $$m \times k$$ 크기의 행렬, P는 $$ k \times n $$ 크기의 행렬이다. 각각의 행렬은 사용자와 잠재 요인 변수간의 행렬, 잠재 요인 변수와 영화간의 행렬인 것이다. 즉, 사용자 $$u$$가 영화 $$i$$의 점수를 주는 방식은 사용자 $$u$$의 영화에 대한 잠재 요인 $$p_{u}$$ 와 그에 대응되는 영화에 대한 잠재 요인 $$q_{i}$$에 의해 결정 된다.

정확한 예측을 위한 objective function은 아래와 같은 형태일 것이다. 여기서 $$ u,i \in \kappa $$란 표현은 관측된 $$u$$와 $$i$$의 쌍을 의미한다.

$$
\text{min}_{\hat{R}}\sum_{u,i \in \kappa} (r_{ui}-\hat{r}_{ui})^{2}\quad \text{s.t.}\;\; \text{rank}(\hat{R}) = k
$$

그런데 영화 평점과 관련된 행렬은 아래와 같이 수많은 null 값들이 존재한다. 이렇게 null 값이 존재하는 행렬의 SVD 결과는 과적합을 유발시키는 것으로 알려져 있다.

||기생충|조커|라라랜드|레옹|버드맨|머니볼|파운더|조디악|노트북|$$\cdot\cdot\cdot$$|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|A|5.0|3.5|null|5.0|4.0|null|3.5|null|2.0|$$\cdot\cdot\cdot$$|
|B|null|1.0|5.0|5.0|null|4.5|null|3.5|1.0|$$\cdot\cdot\cdot$$|
|C|4.0|null|4.5|5.0|null|4.5|5.0|4.0|null|$$\cdot\cdot\cdot$$|
|D|null|null|3.5|3.5|3.0|3.0|2.5|null|null|$$\cdot\cdot\cdot$$|
|E|3.5|4.0|null|null|null|null|3.5|null|null|$$\cdot\cdot\cdot$$|
|F|3.0|5.0|null|5.0|4.0|4.5|null|3.5|2.0|$$\cdot\cdot\cdot$$|
|G|null|null|3.5|5.0|null|4.5|3.5|null|null|$$\cdot\cdot\cdot$$|
|H|1.0|null|null|2.0|3.0|null|null|3.5|2.0|$$\cdot\cdot\cdot$$|
|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|$$\cdot\cdot\cdot$$|

과적합을 방지하기 위해 오차제곱합항 뒤에 벌점화항(penalty term)을 추가하는 것처럼 여기서도 똑같은 방식을 적용할 수 있어 다음과 같은 식을 최소화하는 값을 찾게 된다.

$$
\text{min}_{P,Q} \sum_{u,i \in \kappa} (r_{ui}-q_{i}^{T}p_{u})^{2} + \lambda(||q_{i}||^{2}+||p_{u}||^{2})
$$

이 식을 최소화시키는 벡터 $$p_{u}$$, $$q_{i}$$ 를 발견하는 과정은 보통 2가지 알고리즘 1. SGD(Stochastic Gradient Descent) 2. ALS(Alternating Least Square) 방법이 있다.

#### bias항 추가

평점을 짜게주는 사람이 있고, 후하게 주는 사람이 있다. 그리고 다들 좋다고 평가하는 작품이기 때문에 내가 선뜻 평점을 나쁘게 주기 애매한 영화들도 있다. 이런 것들 모두 평점을 매기는 과정에서 bias 항으로 작용할 수 있다.

사용자 $$u$$의 영화 $$i$$에 대한 bias term $$b_{ui}$$는 다음과 같이 분해될 수 있다. $$b_{u}$$는 사용자의 bias를, $$b_{i}$$는 item의 bias를 표현하는 것이라 볼 수 있다.

$$b_{ui} = \mu + b_{u} + b_{i} $$

이를 앞서 사용했던 식에 그대로 적용해보면 다음과 같은 식을 얻을 수 있고

$$
\hat{r}_{ui} = \mu + b_{u} + b_{i} + q_{i}^{T}p_{u}
$$

이를 다시 objective function에 대입하면 다음의 식을 구할 수 있다.

$$
\text{min}_{P,Q,B} \sum_{u,i \in \kappa} (r_{ui}-\mu-b_{i}-b_{u}-q_{i}^{T}p_{u})^{2} + \lambda(||q_{i}||^{2}+||p_{u}||^{2}+b_{u}^{2}+b_{i}^{2})
$$

이 objective function 역시 SGD 방식이나 ALS 방법을 통해 해결할 수 있다.





