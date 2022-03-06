---
layout: post
title:  "Concept of Python's Class"
date: 2022-01-16
author: seolbluewings
categories: Data
---


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

#### Method

클래스 안에서 정의되는 함수를 Method(메소드)라고 부른다.

가장 빈번하게 활용되는 메소드는 인스턴스 메소드(instance Method)로 첫번째 parameter를 항상 객체 자신을 의미하는 self로 받는 메소드이다.

그 외 정적 메소드(static Method)와 클래스 메소드(Class Method)가 있는데 정적 메소드는 self가 쓰이지 않는 메소드로 객체와 독립적이며 클래스 로직 상 내부에서 써야하는 메소드일 때 활용하는 메소드이다.

클래스 메소드는 self 대신 cls라는 클래스를 의미하는 parameter를 전달한다. 따라서 cls parameter를 통해 클래스 변수를 접근할 수 있다. 

```
class AttackUnit:
    count = 0
        
    def __init__(self, name, hp, damage) :
        self.name = name
        self.hp = hp
        self.damage = damage
        AttackUnit.count += 1
    
    @classmethod
    def unit_count(cls) :
        print("유닛이 {}개 생성되었습니다".format(cls.count))
    
    def attack(self, location) :
        print("{0} : {1} 방향으로 적군을 공격 합니다. [공격력 {2}]".format(self.name, location, self.damage))
    
    def damaged(self, damage) :
        print("{0} : {1} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        print("{0} : 현재 체력은 {1} 입니다.".format(self.name, self.hp))
        if self.hp <= 0 :
            print("{0} : 유닛이 파괴되었습니다.".format(self.name))
```

#### 상속

새로운 Class를 생성하는 과정에서 앞서 생성한 Class를 활용하는 경우, 이를 Class를 상속받는다 라고 표현한다. 하나의 Class 선언 시, 2개 이상의 Class를 상속 받으면 이를 다중 상속이라 표현하며 콤마(,)를 사용해서 상속받는 Class를 구분해준다.

```
# 비행 기능에 대한 Class 선언
class Flyable :
    def __init__(self, flying_speed) :
        self.flying_speed = flying_speed
    
    def fly(self, name, location) :
         print("{0} : {1} 방향으로 날아갑니다합니다. [속도 {2}]"\
             .format(self.name, location, self.flying_speed))

# 공중 공격 유닛 Class 선언 (Wraith 등...)
# 다중상속 시, 콤마(,)로 구분
class Flyable_AttackUnit(AttackUnit, Flyable) :
    def __init__(self, name, hp, damage, flying_speed) :
        AttackUnit.__init__(self, name, hp, damage)
        Flyable.__init__(self,flying_speed)
```

이렇게 상속을 받아 새로운 Class를 생성했을 때, 부모 Class에서 정의한 Method를 자식 Class에서 동일 명칭이지만 기능은 다르게 만들고 싶다면 자식 Class 내에서 Method를 생성하면 된다. 이를 Method Overriding 이라 한다.




포스팅 관련된 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/Python%20Class.ipynb)에서 확인 가능합니다.


#### 참고문헌

1. [나도 코딩 유투브 강의](https://www.youtube.com/watch?v=kWiCuklohdY)
2. [점프 투 파이썬](https://wikidocs.net/book/1)