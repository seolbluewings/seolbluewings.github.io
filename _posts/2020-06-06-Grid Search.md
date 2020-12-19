---
layout: post
title:  "Hyperparameter Tuning"
date: 2020-06-06
author: seolbluewings
categories: Tuning
---

교차검증은 모델의 일반화 성능을 측정하는 방법이며 이제는 모델의 hyperparameter를 튜닝함으로써 일반화 성능을 향상시키는 방법을 알아보고자 한다. 모델의 일반화 성능을 최대로 높여주는 hyperparameter 값을 찾는 일은 모든 데이터 분석에서 반드시 진행해야하는 작업이다. 이에 대하여 알아보자.

#### hyperparameter란?

hyperparameter는 모델링 과정에서 분석자가 직접 설정하는 값을 의미한다. 예시를 들어서 설명하자면, 랜덤 포레스트 모델에서 n_estimators 값이나 KNN모델의 K값 등을 hyperparameter라고 부르며 이런 값을 조정하는 과정을 hyperparameter tuning이라고 부른다.

hyperparameter tuning은 가장 높은 정확도를 확보하기 위해 적절한 hyperparameter set을 탐색하는 과정을 의미한다. 최적의 hyperparameter를 선택함으로써 우리는 더 높은 정확도를 갖춘 모델을 얻을 수 있다. 최적의 hyperparameter를 찾는 것은 어려운 일이지만 대표적으로 그리드 서치(Grid Search), 랜덤 서치(Random Search)를 활용하여 이 작업을 수행할 수 있다.

#### 그리드 서치 (Grid Search)

그리드 서치(이하 Grid Search)는 가장 빈번하게 활용되는 hyperparameter tuning 방식이다. 이 방식은 hyperparameter 후보로 설정한 값들의 가능한 모든 조합을 탐색한다. 가능한 모든 조합을 탐색한다는 점에서 장점이 있으나 이에 따른 계산량 증가가 단점인 방식이다.

![GSCV](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/gridsearch.PNG?raw=true){:width="70%" height="70%"}{: .center}

위 그림의 좌측과 같이 Grid Search 방법은 hyperparameter가 존재할 것으로 생각되는 범위 내에서 일정 간격으로 hyperparameter 값을 선택할 수 있고 이에 대한 여러가지 조합($$3\times 3 = 9$$개)을 탐색하여 가장 예측력이 높은 (또는 오차가 적은) 모델을 생성하도록 하는 hyperparameter 값을 선택하는 방식이다.

Grid Search 과정에서 각 hyperparameter를 꼭 등분할 필요는 없다. 앞서 말했듯이 hyperparameter는 분석자가 직접 설정하는 값이기 때문에 각 hyperpameter의 간격도 분석자가 설정할 수 있다. 각 hyperparameter간 간격을 좁게할 수도 있고 넓게 있는데 좁게 할수록 더 높은 예측력(오차가 작은)을 가진 모델을 탐색할 수 있으나 계산량이 많아진다는 단점이 있다.

파이썬에서는 다음과 같이 Grid Search를 통해 최적의 hyperparameter를 구할 수 있다. 우선 Grid Search를 할 수 있는 라이브러리를 불러오자.
~~~
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
~~~

데이터는 앞선 포스팅에서 사용했던 iris 데이터를 활용하는 것으로 가정하고 iris 데이터를 바탕으로 꽃의 종류를 예측하는 랜덤 포레스트 모델을 생성한다고 해보자. 그리고 이 때 1. 어떤 hyperparameter를 선택해야 가장 높은 예측력을 지닌 모델을 생성할 수 있는지와 2. 그 목적을 달성하기 위해 최적의 hyperparameter를 탐색하는 과정에 대해 살펴보자.

우선 가장 간단한(모든 hyperparameter를 default 값으로 설정한) 랜덤 포레스트 모델을 생성한다.

~~~
rf_classifier = RandomForestClassifier(random_state = 0)
rf_classifier.fit(x_tr,y_tr)
~~~

자 이제 hyperparameter 튜닝을 위한 scikit-learn의 GridSearchCV 함수를 사용해보자. 이 때, 우리가 사전에 결정짓는 hyperparameter 후보군을 key-value 쌍의 데이터로 입력하여 key값에 해당하는 hyperparameter 값을 튜닝한다. 랜덤 포레스트 함수에는 여러가지 hyperparameter가 있으나 대표적인 몇가지 hyperparameter만 Grid Search 해보기로 하자.

먼저 다음과 같이 hyperparameter를 key-value 쌍의 데이터로 입력하고 

~~~
rf_param_grid = {'n_estimators': [10,20,30,50,100],
              'max_depth': [2,3,4],
              'max_features' : [2,3],
              'min_samples_leaf': [1,2,3,4,5] }
~~~

GridSearchCV 함수를 활용하여 최적의 hyperparameter 값을 탐색해낸다. 함수 이름을 통해서 확인할 수 있듯이, 교차검증을 통해 최적의 hyperparameter를 도출해낸다. 이 과정에서 탐색할 조합의 숫자가 많을수록, CV횟수가 많을수록 시간이 오래 걸린다.

~~~
rf_classifier_grid = GridSearchCV(rf_classifier, param_grid = rf_param_grid, scoring ='accuracy',n_jobs= -1, cv= 5, verbose = 1)
rf_classifier_grid.fit(x_tr, y_tr)
~~~

이 2가지 함수를 활용한 뒤 <.best_params_> 를 뒤에 붙여 교차검증을 통해 도출해낸 최적의 hyperparameter를 출력해낼 수 있고 추후 분석 과정에서 여기서 출력된 hyperparameter 값을 활용한다.

~~~
print("가장 높은 정확도 : {0:.2f}".format(rf_classifier_grid.best_score_))
print("최적의 hyperparamter :",rf_classifier_grid.best_params_)
# 가장 높은 정확도 : 0.96
# 최적의 hyperparamter : {'max_depth': 4, 'max_features': 3, 'min_samples_leaf': 2, 'n_estimators': 100}
~~~

각각 max_depth = 4, max_features = 3, min_sample_leaf = 2, n_estimator = 100 으로 최적의 hyperparameter가 도출된 것을 확인할 수 있다.

#### 랜덤 서치 (Random Search)

랜덤 서치(이하 Random Search)는 이름을 통해서도 느낌을 받을 수 있듯이, hyperparameter의 임의 조합을 활용하여 최적의 hyperparameter 값을 결정짓는다. hyperparameter가 가질 수 있는 모든 값을 탐색하지 않고 우리가 시행횟수로 설정한 n_iter 수에 따라 hyperparameter 조합을 생성한다.

![GSCV](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/gridsearch.PNG?raw=true){:width="70%" height="70%"}{: .center}

위 그림의 우측과 같이 Random Search는 임의로 선정된 hyperparameter 조합을 활용하여 모델의 성능을 탐색한다.

파이썬에서 Random Search를 다음과 같이 시행할 수 있다. 우선 Random Search를 진행할 수 있도록 라이브러리를 불러오자.

~~~
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
~~~

앞선 Grid Search와 마찬가지로 간단한 랜덤 포레스트 모델을 생성하고

~~~
rf_classifier = RandomForestClassifier(random_state = 0)
rf_classifier.fit(x_tr,y_tr)
~~~

hyperparameter의 key-value 쌍을 입력한다.

~~~
rf_param_grid = {'n_estimators': [10,20,30,50,100],
              'max_depth': [2,3,4],
              'max_features' : [2,3],
              'min_samples_leaf': [1,2,3,4,5] }
~~~

이제는 hyperparameter 튜닝을 위한 scikit-learn의 RandomizedSearchCV 함수를 사용한다. 앞선 GridSearchCV와 달리 시행횟수에 해당하는 값을 입력하여 임의의 hyperparameter 조합을 탐색하는 횟수를 지정해준다.

~~~
rf_classifier_grid = RandomizedSearchCV(rf_classifier, param_distributions = rf_param_grid, n_iter = 10, scoring ='accuracy', n_jobs= -1, cv= 5, verbose = 1)
rf_classifier_grid.fit(x_tr, y_tr)
~~~

다음과 같이 Random Search를 통해 도출해낸 hyperparameter를 출력해낼 수 있고 이를 활용한 결과값도 확인할 수 있다.

~~~
print("가장 높은 정확도 : {0:.2f}".format(rf_classifier_grid.best_score_))
print("최적의 hyperparamter :",rf_classifier_grid.best_params_)
~~~

#### 결론

hyperparameter 튜닝은 머신러닝 모델을 생성하는 과정에서 가장 어려운 단계 중 하나지만, 더 좋은 예측력을 갖춘 모델을 생성하기 위해서는 반드시 거쳐야하는 단계이다. Grid Search와 Random Search는 최적의 hyperparameter를 찾아내는 효율적인 방식이며 우리는 이미 생성되어 있는 함수를 활용하여 최적의 hyperparameter 생성에 대한 목적을 달성해낼 수 있다. Grid Search는 hyperparameter의 조합의 개수가 많을수록 계산량이 많아져 Random Search를 활용하는 것이 낫다고 한다. 선택은 분석자의 자유다. 2가지 방법 말고 Bayesian Optimization과 같은 방법도 있다. 더 좋은 예측을 위해서는 다양한 시도를 해볼 필요가 있다.

##### 이 포스팅과 관련된 간단한 코드는 다음의 [주소](https://github.com/seolbluewings/code_example/blob/master/1.hyperparameter%20tuning.ipynb)에서 확인할 수 있습니다.
