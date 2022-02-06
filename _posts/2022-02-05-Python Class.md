---
layout: post
title:  "Python Class"
date: 2022-01-16
author: seolbluewings
categories: Data
---

[작성중... ]

클래스(Class)는 객체 지향형 언어에서 볼 수 있는 개념으로 이번 포스팅에서는 Python에서의 클래스에 대해 알아보고자 한다. 클래스는 인스턴스(instance)를 만드는데 활용되는 설계도로 인스턴스는 Class에 의해 생성된 객체라고 보면 된다.

객체 지향형 프로그래밍을 수행하는 목적은 코드 재사용, 중복방지, 유지보수 용이성 등으로 이는 개발 과정에서 항상 신경써야할 부분이 아닐 수가 없는 사항들이다.

Class를 써야하는 이유는 다음과 같다. 스타크래프트에서 유닛을 생성하는 과정을 Python 코드로 표현하면 다음과 같을 것이다. 마린을 생성한다고 가정한다면 다음과 같이 표현할 수 있을 것이다.

```
name = 'marine' # Unit 이름
hp = 40 # Unit HP
damage = 6 # Unit 공격력
print("{} 유닛이 생성되었습니다.".format(name))
print("체력 {0}, 공격력 {1}\n".format(hp,damage))
```

그런데 여기서 마린을 하나 더 생성한다고 한다면, name1, hp1, damage1을 또 정의해줘야 하는 상황이 발생한다. 이런 상황을 Unit이란 클래스 선언을 통해 해결할 수 있다.

```
class Unit :
    def __init__(self, name, hp, damage) :
        self.name = name
        self.hp = hp
        self.damage = damage
        print("{0} 유닛이 생성되었습니다.".format(self.name))
        print("체력 {0}, 공격력 {1}".format(self.hp, self.damage))

marine1 = Unit("marine",40,6)
marine2 = Unit("marine",40,6)
```

Unit 클래스로 만들어진 marine1, marine2 라는 별개의 인스턴스는 서로 독립적이다. 실제 게임이라면 마린을 수십개 생성할 것이기 때문에 Class 선언 방식으로 코드를 만드는 것이 굉장히 합리적이다.

#### self와 \_\_init\_\_

\_\_init\_\_은 인스턴스 생성 과정에서 항상 실행되는 항목이며 \_\_init\_\_ 함수를 정의할 때 사용된 변수(self 제외)는 인스턴스 생성 시 반드시 포함되어야 한다.

그리고 self의 경우는 클래스를 가지고 인스턴스를 생성했을 때, Class 내부에서 이를 지칭하기 위해 사용되는 변수이다. 상단의 Unit 클래스를 활용해 생성한 marine1,marine2 인스턴스에서 각각 marine1이 self가 되고 marine2가 self가 되는 것으로 볼 수 있다.

.....






포스팅 관련된 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Python%20Class.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [나도 코딩 유투브 강의](https://www.youtube.com/watch?v=kWiCuklohdY)
2. [점프 투 파이썬](https://wikidocs.net/book/1)