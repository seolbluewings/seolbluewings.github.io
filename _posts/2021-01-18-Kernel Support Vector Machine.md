---
layout: post
title:  "Kernel Support Vector Machine"
date: 2021-01-18
author: seolbluewings
categories: 서포트벡터머신
---

[작성중...]

앞선 SVM [포스팅](https://seolbluewings.github.io/%EC%84%9C%ED%8F%AC%ED%8A%B8%EB%B2%A1%ED%84%B0%EB%A8%B8%EC%8B%A0/2020/11/29/Support-Vector-Machine.html)에서는 데이터가 선형 hyperplane을 통해 분리가 가능함을 가정하였다. 여기에 Slack Variable을 추가하여 약간의 오차는 눈감아주는 모델도 생성했었다. 그러나 현실적으로 데이터를 선형 hyperplane을 통해 분류해낼 수 있는 경우는 없다. 단순한 XOR 게이트 문제 역시 선형분리시키지 못하는 사례 중 하나다.

이러한 문제를 해결하기 위한 방법으로는 Kernel 기법이 있다. 데이터를 기존의 차원보다 더 높은 차원의 공간으로 mapping 시키고 한층 높아진 차원에서 hyperplane을 이용한 선형분리를 시킨다. 원 데이터셋이 유한한 차원에 존재한다면, 우리는 이를 더 고차원 공간에서 Class를 분류해낼 수 있다.

![SVM](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Kernel_SVM.png?raw=true){:width="100%" height="70%"}

이 그림을 통해서도 확인할 수 있다. 왼쪽 이미지는 원래 데이터 차원에서 그림을 그린 것이다. 두 Class를 선형 hyperplane을 통해서는 분리시킬 수 없다. 그러나 오른쪽 이미지와 같이 가상의 z축을 생성시켜 두 Class 데이터간의 높이 차이를 발생시킨다면, 우리는 이제 선형 hyperplane으로 이 둘을 분리시킬 수 있게 된다.

데이터셋 $$x$$를 고차원으로 mapping시킨 결과를 $$\phi(x)$$라고 한다면, 이 새로운 차원에서의 hyperplane 모델은 다음과 같이 표현 가능하다.

$$ f(x) = w^{T}\phi(x)+b$$



#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)
