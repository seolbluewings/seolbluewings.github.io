---
layout: post
title:  "Pandas Cheating Sheet"
date: 2021-11-14
author: seolbluewings
categories: Data
---



#### csv 파일 불러오기

```python
df = pd.read_csv(r'C:\Users\seolbluewings\Desktop\sample\Hitters.csv')
```

- 역슬래쉬 사용할 경우, 가장 앞에 reverse를 뜻하는 r 표기 필요하고 아닌 경우에는 / 사용하면 됨

#### 데이터 살펴보기

```python
df.info()
df.describe()
df.isnull().sum()
df['HmRun'].unique()
df['HmRun'].nunique()
df.shape
df.columns
```

- df.info() : DataFrame을 구성하는 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 출력
- df.describe() : 컬럼별 요약 통계량 출력
- df.isnull().sum() : NULL 데이터 개수 확인
- df['컬럼명'].unique() : 컬럼에 포함된 유일한 값 확인
- df['컬럼명'].nunique() : 컬럼에 포함된 유일한 값 개수확인
- df.shape : DataFrame의 행과 열 출력, index하여 따로 출력 가능
- df.columns : DataFrame을 구성하는 컬럼명 확인


#### 데이터 인덱싱

```python
df['Hits]
df.Hits
df[['Hits','HmRun']].head(5)
```

- DataFrame에서 1개 열을 출력하면 Series 형태로 출력 됨
- 2개 이상의 Column을 가져오기 위해서는 DataFrame 뒤에 대괄호[] 하나를 입력해주고 그 안에 LIST 형식으로 컬럼을 입력해주면 된다.

```python
df.iloc[0:4,0:3]
df.iloc[[0,1,5],[0,7,10]]
df.loc[0:3,['HmRun','CHits','Hits']]
```

- df.iloc 함수는 Column 명이 아니라 index 숫자로 가져와야 함
- df.iloc 함수에서 Row, Column을 띄엄띄엄 가져오고 싶다면 LIST 형태로 입력해주면 된다
- df.loc 함수를 통해 인덱싱을 한다면, Row부분은 index를 입력해주고 Column 부분은 컬럼명을 입력해준다.

#### 조건에 맞는 데이터행 출력하기

```python
df.loc[df['Hits']>= 200,:]
df.loc[(df['Hits']>= 200) & (df['HmRun'] > 10),:]
```

- OR 조건이면 \| 으로 이어주면 되고 AND 조건이면 &로 이어주면 됨
- 다만 각 조건은 ()로 묶어서 표현해주어야 함

#### 결측값/중복 데이터 처리

```python
df.dropna(axis = 0, how = 'any')
df.dropna(subset = ['Hits','HmRun'])
df.fillna({'Salary' : 10})
df.fillna(df['Salary'].mean())
df.fillna(method = 'ffill'), df.fillna(method = 'bfill')
```

- df.dropna()의 default 옵션은 axis = 0, how = 'any' 이다. 중복 데이터가 존재하는 행(axis = 0)을 제거하고 하나의 Column 이라도 Null이 존재하면 제거
- subset은 특정 Column에서만 Null 데이터를 찾고자할 때 사용한다.
- df.fillna() 함수는 {} 형태로 Column : Value 형태로 입력해주면 된다.
- mean(), median() 등과 같은 값으로도 Null 값 대체 가능
- 바로 직전 데이터로 대체 ffill, 바로 다음 데이터로 대체 bfill


#### 데이터 순서정렬
