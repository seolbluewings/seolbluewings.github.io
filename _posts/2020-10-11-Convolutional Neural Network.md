---
layout: post
title:  "합성곱 신경망(Convolutional Neural Network)"
date: 2020-10-11
author: seolbluewings
categories: Statistics
---

[작성중]

햡성곱 신경망(Convolutional Neural Network, 이하 CNN)은 주로 이미지, 음성인식에 쓰이는 딥러닝 기법으로 앞서 살펴본 NN의 심화형태라고 볼 수 있다.

기본적인 NN은 인접 계층의 모든 뉴런과 완전히 결합되어(fully-connected) 있다. 이처럼 완전하게 연결된 계층을 Affine 계층이라 하는데 기존의 NN은 다음의 그림과 같은 형태를 보일 것이다. 여기서 ReLU는 Sigmoid 함수처럼 결과 출력에 활용되는 활성화 함수이며 CNN에서는 Sigmoid보다 ReLU가 더 빈번하게 쓰인다.

$$
f(x) =
\begin{cases}
0 \quad \text{if} \quad x < a  \\
x \quad \text{otherwise}
\end{cases}
$$

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/NN_STRUCTURE.png?raw=true){:width="70%" height="70%"}{: .center}

그림과 같이 Affine 계층을 통해 ReLU값을 출력해내고 이를 다시 입력값으로 활용하여 계속해서 연산을 진행한 후, 최종적으로는 Softmax 함수를 사용하여 결과를 return한다.

반면, CNN은 그림과 같이 합성곱 계층(convolutional layer)과 풀링 계층(pooling layer)이 추가된다. 기존의 Affine-ReLU 절차가 CNN에서는 Conv-ReLU-(Pooling) 형태로 바뀌는 것이다.

또한 아래 그림을 통해 확인할 수 있는 것처럼 출력층과 가까운 곳에서는 Affine-ReLU 연산을 할 수 있으며 최종 출력층에서는 Affine-Softmax 를 활용한다.

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/CNN_STRUCTURE.png?raw=true){:width="70%" height="70%"}{: .center}

#### Conv층을 왜 사용하는가?






#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)

3. [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)