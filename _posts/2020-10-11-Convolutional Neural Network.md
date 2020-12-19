---
layout: post
title:  "Convolutional Neural Network"
date: 2020-10-11
author: seolbluewings
categories: NeuralNetwork
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

기존 NN에서 활용하는 Affine 계층은 인접하는 층에 존재하는 뉴런끼리 모두 연결되고 출력 뉴런의 개수를 사용자 임의로 설정할 수 있다. Affine 계층은 데이터의 차원을 무시한다는 단점을 갖고 있다.

이미지 데이터가 존재한다고 가정하자. 이미지 데이터는 일반적으로 3차원(가로,세로,색상) 데이터인데 이를 기존의 NN에 적용한다고 한다면, 이 데이터를 1차원 형태의 데이터로 변형시켜야 한다. 3차원 데이터를 완전히 활용하지 못하고 이를 1차원으로 차원을 낮춰야한다는 단점이 있는 것이다.

한편, Conv층은 데이터의 차원을 유지한다. Conv층의 입출력 데이터를 특징맵(feature map)이라 부르는데 Conv층은 입력 특징맵(input feature map)과 출력 특징맵(output feature map) 차원을 동등하게 유지시켜준다. 3차원 데이터로 입력받으면 다음 계층에 3차원 데이터로 이를 전달한다.

#### 용어 정리

##### 1. 합성곱 연산

합성곱(Conv) 계층에서는 합성곱 연산이 수행된다. 합성곱 연산을 필터 연산이라 부르기도 하는데, 그림과 같이 합성곱 연산은 입력 데이터에 필터(또는 커널이라 부른다)를 적용해서 출력을 한다.

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Conv.png?raw=true){:width="70%" height="70%"}{: .center}


데이터와 필터는 (높이 H, 너비 W)의 차원을 갖고 있고 그림에서 입력 데이터는 (4,4) Size, 필터는 (3,3) Size, 출력은 (2,2) Size라 할 수 있다.

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Conv1.png?raw=true){:width="100%" height="70%"}{: .center}

그림처럼 필터를 일정 간격으로 움직이면서 입력 데이터에 적용하여 값을 얻어낸다. 출력의 (1,1) 위치값은 다음과 같이 계산된다. 마찬가지 방법으로 다른 값을 구한다.

$$ 1\times2 + 2\times0 + 3\times1 + 0\times0 + 1\times1 + 2\times2 + 3\times1 + 0\times0 + 1\times2 = 15 $$

즉, CNN의 필터는 NN의 weight와 같은 역할을 한다고 볼 수 있다.

##### 2. 패딩(Padding)

그런데 합성곱 연산의 결과를 잘 살펴보면 이상한 점을 발견할 수 있다. 입력 데이터가  (4,4) Size인데 출력은 (2,2) Size로 나온 것이다. 합성곱 연산을 수행하니 데이터의 Size가 줄어든 셈인데 만약 합성곱 연산을 연속해서 수행한다면 최종적으로는 출력의 크기가 (1,1) Size로 나올 것이다. 출력 Size를 고정시키기 위한 목적으로 활용하는 기법이 바로 패딩(Padding)이다. 패딩을 사용하면 우리는 다음 계층으로 연결되는 출력 결과(다음 층에서의 입력 데이터)의 크기를 일관되게 고정시킬 수 있다.


![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Padding.png?raw=true){:width="100%" height="70%"}{: .center}

패딩은 아래의 그림처럼 입력 데이터 주변을 특정 값으로 채워서 처리하는 것이다. 그림처럼 (4,4) Size의 입력 데이터 주변에 특정 값을 추가하여 (6,6) Size의 입력 데이터를 만들었고 이 결과 출력이 (4,4) Size로 만들어지게 된다.

##### 3. 스트라이드(Stride)

입력 데이터에 필터를 이동시키면서 출력을 얻어내는데 이 때 필터를 이동시키는 간격을 스트라이드(Stride)라고 부른다. 앞선 예제에서는 간격을 1칸씩 옮겼으니 스트라이드는 1이다. 스트라이드를 키우게 되면 출력의 Size는 감소하게 된다. 반면에 앞서 패딩에서 언급했던 것처럼 패딩을 크게하면 출력의 Size가 커진다.

입력 데이터 Size(H,W) 와 필터 Size(FH,FW), 출력 Size(OH,OW) 패딩(P), 스트라이드(S) 사이의 관계는 다음의 수식과 같다.

$$
\begin{align}
OH &= \frac{H+2P-FH}{S}+1 \nonumber \\
OW &= \frac{W+2P-FW}{S}+1 \nonumber
\end{align}
$$

#### 3차원 데이터의 합성곱 연산

이미지 데이터는 3차원 형태의 데이터(채널, 높이, 너비)이다. 앞서서는 단일 채널의 데이터를 통한 합성곱 연산 예시를 살펴보았다. 이제는 채널까지 고려한 3차원 데이터 입력의 합성곱 연산을 생각해보자.

3차원 데이터의 합성곱 연산은 채널 방향으로 feature map의 개수가 늘어난다. 아래의 그림처럼 입력 특징맵의 합성곱 연산을 채널마다 수행하여 그 결과를 모두 더해서 출력 특징맵을 얻어낸다.

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Conv2.png?raw=true){:width="100%" height="70%"}{: .center}

당연하게도 입력 특징맵의 채널 수와 필터의 채널수는 동일해야 한다. 그리고 필터의 Size는 모두 동일해야 한다. 그런데 출력의 결과를 살펴보면 1개 채널의 출력 특징맵이다. 다수의 층을 활용해 반복해서 합성곱 연산을 수행하기 위해서는 출력특징맵이 n차원의 형태를 지니는 것이 좋다. 합성곱 연산의 결과 역시 다수의 채널을 갖추도록 만들기 위해선 필터를 다수 적용하여 합성곱 연산을 수행하는 것이다.

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Conv3.png?raw=true){:width="100%" height="70%"}{: .center}

입력 특징맵과 필터를 3차원 직육면체로 표현한 것은 (채널, 높이, 너비)의 크기를 갖기 때문이다. 그런데 여기에 필터의 차원을 4차원으로 (출력 채널 수, 입력 채널수, 높이, 너비) 만들어서 출력 특징맵 역시 다수의 출력 특징맵을 가지도록 만들 수 있다.

#### 풀링(Pooling) 계층

앞서 CNN은 기존 NN과 달리 Conv층 $$\to$$ ReLU $$\to$$ Pooling 층을 거친다고 하였다. 풀링은 선택적으로 수행할 수 있는 단계이며, Size를 줄이는 절차라고 할 수 있다.

![CNN](https://github.com/seolbluewings/seolbluewings.github.io/blob/master/assets/Pooling.png?raw=true){:width="100%" height="70%"}{: .center}

위의 그림은 스트라이드 2를 적용하여 $$2\times2$$ 크기의 데이터로 데이터의 Size를 줄인다. $$2\times2$$ Size 영역에서 가장 큰 원소를 하나 꺼내는 것으로 이를 최대 풀링(max pooling)이라 부른다. 이미지 처리에서는 주로 max pooling 처리를 진행하며, 이 외에는 평균 풀링(average pooling)처리가 있다.

Pooling은 단순히 평균값이나 최대값을 채택하므로 별도의 학습과정이라할 것이 없으며 채널마다 독립적으로 이루어지는 작업이기 때문에 입력 데이터의 채널 수를 그대로 보존하는 특징을 갖는다. Pooling층의 과정은 어떤 관점에서 살펴보면 Sampling을 진행하는 것이며 또한 가장 큰 값을 가져오기 때문에 데이터의 양을 줄이면서 동시에 유의미한 정보를 보존하는 형식이라 할 수 있다. 

... to be continued 

#### 참조 문헌
1. [PRML](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) <br>

2. [단단한 머신러닝](http://www.yes24.com/Product/Goods/88440860)

3. [밑바닥부터 시작하는 딥러닝](https://www.hanbit.co.kr/store/books/look.php?p_code=B8475831198)