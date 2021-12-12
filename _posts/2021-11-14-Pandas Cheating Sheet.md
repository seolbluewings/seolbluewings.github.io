---
layout: post
title:  "Pandas Cheating Sheet"
date: 2021-11-14
author: seolbluewings
categories: Data
---

업무에 자주 사용하는 Python Pandas 함수를 정리하고자 한다. 이 포스팅을 업로드한 이유는 개인적인 Cheating Sheet로 활용하기 위함이며, 추후 계속 업데이트할 생각이다.

#### csv 파일 불러오기

```python
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv(r'C:\Users\seolbluewings\Desktop\sample\Hitters.csv')
```

- 역슬래쉬 사용할 경우, 가장 앞에 reverse를 뜻하는 r 표기 필요하고 아닌 경우에는 / 사용하면 됨
- 단순히 pandas로 테이블을 불러오면, Column 개수, Row 개수가 많으면 일부만 출력되는 현상을 확인할 수 있다. 전체 데이터를 다 보려고 한다면, pd.set_option() 함수를 사용
- 컬럼을 20개, 행을 60개만 출력하고자 한다면, pd.options.display.max_columns = 20 / pd.options.display.max_rows = 60 과 같은 내용을 입력해주면 됨

#### 데이터 살펴보기

```python
df.info()
df.describe()
df.isnull().sum()
df['HmRun'].unique()
df['HmRun'].nunique()
df.shape
df.columns
df['Division'].value_counts()
```

- df.info() : DataFrame을 구성하는 행과 열의 크기, 컬럼명, 컬럼을 구성하는 값의 자료형 출력
- df.describe() : 컬럼별 요약 통계량 출력
- df.isnull().sum() : NULL 데이터 개수 확인
- df['컬럼명'].unique() : 컬럼에 포함된 유일한 값 확인
- df['컬럼명'].nunique() : 컬럼에 포함된 유일한 값 개수확인
- df.shape : DataFrame의 행과 열 출력, index하여 따로 출력 가능
- df.columns : DataFrame을 구성하는 컬럼명 확인
- df['컬럼명'].value_counts() : 범주형 자료의 경우 각 범주에 대한 데이터개수(unique값이 아님)를 출력해준다

#### 컬럼명 변경

```python
df.rename(columns = {'Runs' : 'RUNS', 'HmRun' : 'HOMERUN'}, inplace =True)
```

- rename() 함수를 사용해서 기존의 데이터셋 컬럼명을 변경할 수 있고
- columns = \{\} 사용하여 컬럼명을 변경해주며, Before : After 형식으로 입력해주면 됨
- 컬럼명을 바꾼 다음에 바꾼 값으로 적용하려면 inplace option의 값을 True로 설정해주면 됨


#### 데이터 인덱싱

```python
df['Hits']
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

```python
df.sort_values(by = 'Hits', ascending = False, inplace = False)
df.sort_values(by = ['Hits','Walks'], ascending = [True. False])
df.sort_index(ascending = False)
```

- sort_values() 함수를 사용하여 데이터의 오름/내림차순 정렬 가능
- by option 뒤에 기준이 될 컬럼을 입력해주고 ascending option 으로 오름차순,내림차순 정렬을 진행한다.
- by 뒤에 입력할 컬럼의 개수와 ascending 에 대응하는 T/F 값은 동등해야 할 것
- default option은 axis = 0 인데 데이터 순서정렬은 기본적으로 행을 기준으로 처리하기 때문에 axis = 0을 건들 이유는 없어 보임
- default option으로 inplace = False인데 이는 정렬된 데이터가 기존의 값을 대체할지 여부를 표기해주는 것, 정렬된 데이터로 기존의 값을 대체하고자 한다면, inplace = True 옵션을 설정해주면 됨
- 데이터를 index 기준으로 정렬하고 싶다면, sort_index() 함수를 사용하면 됨

#### 중복 데이터 제거

```python
df.drop_duplicates(subset = ['Years','HmRun'], keep = 'first', ignore_index = True)
df.drop_duplicates(subset = ['Years','HmRun'], keep = 'last', ignore_index = True)
df.drop_duplicates(subset = ['Years','HmRun'], keep = False, ignore_index = True, inplace =True)
```

- 데이터프레임에서 중복 데이터를 제거하고자 할 때, drop_duplicates() 함수를 사용
- 중복 데이터를 제거하고자 할 컬럼을 subset option의 값으로 설정
- keep option의 값으로 first, last, False가 가능한데 이는 중복 데이터의 첫번째/마지막 값을 남길 것이냐?, 중복 데이터 전부를 제거할 것인가에 대한 지시를 내리는 것으로 보면 됨
- ignore_index = True/False는 삭제된 데이터셋의 index를 재설정하는데 영향을 줌
- inplace = True/False option은 중복이 제외된 데이터셋으로 기존의 데이터셋을 대체하는가 여부를 결정 지음

#### lambda 함수를 이용한 새로운 컬럼 생성

- 만약 HmRun 개수를 가지고 Excellent, Good, Average 타자를 구분하기로 결정하기로 했다면?
- HmRun > 30 이면 Excellent, HmRun > 10 이면 Good, 그 외에는 Average 인 것으로 하자

```python
df['Hitter_DSCD'] = df.HmRun.apply(lambda x : 'Excellent' if x>30 else('Good' if x>10 else 'Average'))
```

- apply() 함수는 내부에 포함된 함수를 데이터셋에 적용할 때, 사용하는데 lambda 함수를 사용하고자할 때 자주 활용
- 변경해서 입력될 값을 먼저 적어주고 그 뒤에 if else문으로 작성해주면 됨




포스팅 내용에 대한 코드는 다음의 [링크](https://github.com/seolbluewings/Python/blob/master/cheating%20sheet/pandas%20cheating%20sheet.ipynb)에서 확인 가능합니다.
