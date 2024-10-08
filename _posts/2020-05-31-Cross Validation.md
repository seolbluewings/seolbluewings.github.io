---
layout: post
title:  "Cross-Validation"
date: 2020-05-31
author: seolbluewings
categories: Statistics
---

앞선 포스팅에서 우리는 전체 MSE를 낮게하는 것 중에서 분산과 편향의 적절한 균형을 찾아야 한다고 말했다. 그렇다면 우리는 어느 정도의 복잡도를 갖춘 모델을 결정해야 한다. 그 적정 수준은 어떻게 결정지을 수 있을까?

#### 학습의 일반화(generalization)

예측을 위한 모델의 경우, 훈련 데이터셋(training data set) 뿐만 아니라 아직 마주하지 못한 새로운 데이터(test data set)에 대해서도 예측이 잘 되어야 한다. 훈련 데이터셋을 모두 활용해서 모델을 만들면 새로운 데이터에 대해서도 예측 성능이 좋을 것이라고 생각할 수 있으나 실제로는 그렇지 않다.

생성한 모델이 훈련 데이터에 너무 과도하게 적합되면, 나머지 새로운 데이터셋(test data set)에 대해 분류 또는 예측이 맞지 않는 과적합(overfitting, 이하 오버피팅) 현상이 발생한다.

오버피팅이란 훈련 데이터에 지나치게 학습이 이루어진 모델이 훈련 데이터에 대해서 분류/예측을 잘 수행하지만 새로운 데이터에 대해서는 분류/예측을 잘하지 못하는 현상을 말한다. 우리는 모델을 만들어서 관측하지 못했던 데이터, 즉 새로운 데이터에 대한 예측을 진행하기 때문에 새로운 데이터에 대해 예측력이 좋지 못한 것은 결코 바람직하지 않다.

오버피팅이 있다면 그 반대 개념인 언더피팅(underfitting)이 있을 것이다. 언더피팅은 모델이 지나치게 단순한 결과, 모델이 데이터의 경향성을 충분히 따라가지 못하는 경우 발생한다.

특히 과적합은 우리가 모델링을 하는 과정에서 주의해야할 사항이다. 훈련 데이터를 가지고 과적합을 시킬 경우, 우리는 굉장히 좋은 모델을 만든 것으로 착각하지만 실제로는 그렇지 않기 때문이다.

우리가 지향하는 바는 적정수준의 피팅이 된 학습이다. 언더피팅과 오버피팅은 아래의 모델 복잡도 그림에서 각각 왼쪽, 오른쪽에 해당된다. 우리는 최적의 모델 복잡도를 가진 모델을 선택해야 한다.

![CV](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/model_complexity.PNG?raw=true){:width="70%" height="70%"}{: .center}

#### 검증 데이터(Validation Set)

오버피팅을 어떻게 피할 수 있을까? 이 질문은 적정수준의 적합을 어떻게 찾아낼 수 있는가란 질문과 동등하며 어느 선까지 모델을 학습시킬 것인가란 질문과도 동등하다. 이에 대한 대표적인 해결방안은 검증 데이터(Validation Set) 개념을 도입하는 것이다.

앞서 우리는 모든 훈련 데이터를 사용해서 모델을 학습시키는 경우, 훈련 데이터에 대한 오버피팅 현상이 발생하는 것을 언급하였다. 그렇다면, 우리는 모든 데이터를 훈련 데이터로 활용할 것이 아니라 어느 순간에 훈련을 중단시켜야 한다. 이 때, 활용하는 것이 검증 데이터셋(Validation Set)이다.

![CV](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/validationset.PNG?raw=true){:width="70%" height="70%"}{: .center}

위의 그림과 같이 훈련 데이터로 모델을 학습시키면 학습시킬수록 훈련 데이터에 대한 에러(Error)가 줄어든다. 그런데 훈련 데이터를 통해 생성된 모델을 검증 데이터셋에 적용하게 되면, 에러는 줄어들다가 어느 순간 증가하게 된다. 이렇게 검증 데이터셋의 에러가 증가하기 시작하는 변곡점이 과적합이 시작되는 곳으로 여겨질 수 있다.

훈련 데이터를 가지고 이러한 변곡점을 지나는 순간까지 학습을 진행하게 되면, 훈련 데이터에 지나치게 적응한 나머지 처음보는 데이터인 검증 데이터에 대해 자꾸 틀린 예측을 진행하게 된다.

우리는 검증 데이터셋을 통해 훈련 중인 모델이 과적합 또는 과소적합인지를 확인하여 최적의 적합을 발견해낸다. 이후, 테스트 데이터(test data set)를 활용하여 훈련 데이터 및 검증 데이터를 활용해 도출해낸 최적의 모델을 활용해 성적을 매기게 된다.

보통 우리에게 주어진 전체 데이터셋의 60%를 훈련 데이터셋(train data set)으로 설정하고 검증 데이터, 테스트 데이터 셋을 각 20% 로 설정한다. 이처럼 데이터를 6:2:2로 나누게 되는 경우, 모델 훈련에는 전체 데이터의 60%만 사용하게 된다. 검증 및 테스트 데이터를 활용하는 것은 과적합 방지를 위한 방법이지만, 전체 데이터의 60%만을 활용하여 모델을 만든다는 것은 문제발생의 소지가 있다. 전체 데이터 중에서 어떤 데이터가 60% 안에 들어가는지가 모형 생성에 영향을 미치게 되며 이러한 문제를 해결할 수 있는 방법이 교차 검증(Cross Validation)이다.

#### K-Fold 교차검증(K-Fold Cross Validation)

교차 검증은 모델의 일반화된 성능을 측정하기 위해 전체 데이터셋을 훈련/검증/테스트 셋으로 단 1번 나누는 것보다 더 안정적인 방법이다. 교차 검증 과정에서는 데이터를 여러번 반복하여 나누고 여러번 모델을 학습한다.

가장 흔하게 사용하는 교차 검증 방법은 K-Fold 교차 검증이다. 여기서 K는 상수인데 보통 K=5 또는 K=10으로 설정된다. Fold는 교집합이 없는 데이터의 부분 집합의 개념으로 받아들이면 된다.

K-Fold 교차검증을 진행하기 위해선 데이터셋을 우선 훈련 데이터와 테스트 데이터로 구분 짓는다. 그리고 훈련 데이터셋을 K등분한 후, (K-1)개의 Fold를 훈련 데이터셋으로 활용하고 나머지 1개의 Fold를 검증 데이터셋으로 활용하며 검증 데이터셋에 해당하는 Fold를 rotation 시키는 방법이다. 이를 그림으로 표현하면 아래와 같다.

![CV](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/CV.PNG?raw=true){:width="70%" height="70%"}{: .center}

구체적인 절차를 언급하자면 다음과 같이 진행된다.

1. 훈련 데이터를 K개의 부분집합 $$(\{ \mathcal{D}_{1}, \mathcal{D}_{2}, \cdot\cdot\cdot , \mathcal{D}_{k} \}  )  $$ 로 나눈다.
2. 이 중 $$(\{ \mathcal{D}_{1}, \mathcal{D}_{2}, \cdot\cdot\cdot , \mathcal{D}_{k-1} \})$$ 를 학습을 위한 훈련 데이터로 활용하여 모형을 만들고 남은 데이터셋 $$\mathcal{D}_{k}$$를 검증 데이터셋으로 활용한다.
3. 이렇게 1개의 Fold를 rotation시키면서 교차 검증을 진행한다.
4. 총 K개의 모형을 생성하고 K개의 교차검증 성능 결과를 가지고 최적의 모델을 결정짓기 위한 parameter를 결정짓는다.

#### 계층별 K-Fold 교차검증

만약 데이터가 2가지 클래스 ($$Y=0$$ 또는 $$Y=1$$) 로 구성되며 각 비율이 $$Y=0$$ 이 90%, $$Y=1$$ 이 10% 이라고 가정해보자. 만약 앞서 언급한 K-Fold 교차검증을 진행한다면, 어떤 Fold에는 $$Y=0$$인 데이터만 들어가는 현상이 발생해서 모델을 만드는 과정에 어려움이 발생할 수 있다.

전체 데이터가 각 클래스별 90%, 10% 비중을 가지고 있다면, 교차검증 과정에서의 각 Fold에 데이터가 클래스별 90%, 10%씩 포함되게 생성되는 것이 바람직할 것이다. 계층별 K-Fold 교차검증은 바로 이러한 처리를 해주는 것을 의미한다.

#### 간단한 파이썬 코드

먼저 필요한 라이브러리를 불러온다. load_iris를 통해 간단한 실습에 활용할 iris 데이터를 가져오고 데이터셋을 분류하기 위한 train_test_split, CV 성능을 판단하기 위한 cross_val_score, 간단한 모델을 만들기 위한 RandomForestClassifier 라이브러리를 불러온다.

~~~
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
~~~

sklearn을 통해 불러온 iris 데이터를 pandas의 데이터프레임 형태로 변형시킨다.

~~~
df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
~~~

데이터프레임에서 타겟 변수(y)와 예측을 위해 활용되는 변수(X)로 구분짓고 이를 8:2의 비율로 데이터셋을 훈련 데이터와 테스트 데이터로 구분 짓는다.

~~~
x = df.iloc[:,0:4]
y = df.iloc[:,4]
x_tr, x_ts, y_tr, y_ts = train_test_split(x,y, test_size = 0.2, random_state = 0)
~~~

간단한 랜덤 포레스트 모델을 생성하고 해당 모델에 대한 교차 검증의 점수를 출력해낸다.

~~~
rf_classifier = RandomForestClassifier(n_estimators = 50, max_depth = 3, max_leaf_nodes = 3)
score = cross_val_score(rf_classifier,x_tr,y_tr, cv= 5)
print("평균 정확도 :{:.2f}".format(score.mean()))
### 평균 정확도 : 0.94
~~~

먼저 소개한 계층별 K-Fold 교차검증을 진행한다면, train_test_split 함수를 입력하는 과정에서 stratify = y 옵션을 추가해주면 된다. 타겟 변수의 분포를 고려하여 Fold를 나누기 때문에 성능이 보다 좋아진 것을 확인할 수 있다.

~~~
n_x_tr, n_x_ts, n_y_tr, n_y_ts = train_test_split(x,y, test_size = 0.2, random_state = 0, stratify = y)
n_score = cross_val_score(rf_classifier,n_x_tr,n_y_tr, cv= 5)
print("평균 정확도 :{:.2f}".format(n_score.mean()))
### 평균 정확도 : 0.96
~~~










