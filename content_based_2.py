# Linear Modeal로 유저 프로필 구성.
## 회귀 모델로 유저 프로필 만들기.
## 종속변수 y , 독립변수 x1(코미디), x2(스릴러), x3(액션)    =>    y = a + b1x1 + b2x2 + b3x3
## 선형회귀 모델 사용하기 위해서는 독립변수 x와 종속변수 y 간에 선형적인 관계를 가진다 || 독립변수들 간에 상관관계가 없이 독립적이다 등.

## MSE가 최소값이 되는 절편과 계수를 찾기
## mse = 시그마(조의 실제 평점 - 조의예상평점)^2 / 조가 평점을 준 영화의 계수 <-- 조의 예상 평점 == y = a(절편) + b1(계수)x1 + b2x2 + b3x3
## => 절편과 계수들이 x와 y간에 관계를 설명. 모델은 어떻게 절편과 계수를 적절하게 찾아줄 수 있을까.
## rmse => 선형회귀모델의 계수를 찾는데 사용이 됨.
## 일단 모델은 아무 절편이나 계수를 집어넣는다. -> rmse가 최소가 되는 절편과 계수를 찾아감...

# 선형 모델이란 ?
## 최소 자승법 => 각각의 점으로부터 직선까지의 거리를 재고 => 전반적인 거리가 작아질 때까찌 직선을 이동.

import pandas as pd
import numpy as np

## Read Data
ratings = pd.read_pickle('./data/ml-latest-sm/ratings_updated.p')
genres = pd.read_pickle('./data/ml-latest-sm/genrs.p')


# 샘플 작업 #

## User Profile  유저 프로파일링 => 유저가 어떤 특성을 갖고 있는지, 어떤 장르를 선호하는지에 대한 프로파일링, 그것을 기준으로 해서 새로운 영화가 들어왔을 때, 예측이 가능.
## 1. 샘플데이터로 유저 프로파일 만들어보고
## 2. 이 원리를 전체데이터에 적용해보기
user414 = ratings[ratings['userId'] == 414]
user414.sample()
user414 = user414.merge(genres, left_on='movieId', right_index=True)

## 훈련 데이터, 테스트 데이터 나누기
from sklearn.model_selection import train_test_split

# X => 해당 영화의 장르 정보. y => 해당 영화의 user 평점
# X == feature => y를 예측하기 위해 필요한 정보들. y == label => 정답
X_train, X_test, y_train, y_test = train_test_split(user414[genres.columns], user414['rating'], random_state=42, test_size=.1)

# print(X_train.shape)

# print(user414[genres.columns])
print(user414['rating'])

# Linear Regression (선형 회귀 모델)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
## 선형 모델 생성 완료 => 이 모델에 훈련 시키기...

reg.fit(X_train, y_train)

## coef(ficient) = 계수, intercept = 절편
## reg.coef_ == 각 장르의 계수
## print(reg.coef_)
## print(reg.intercept_)
list(zip(X_train.columns, reg.coef_))
## 액션+어드벤처 장르에 대한 user414의 예상평점 == reg.intercept_ + reg.coef[1]_ + reg.coef_[2]
# user414['rating'].hist()
# user414.loc[user414['Children'] == 1, 'rating'].hist()

predict = reg.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predict)
rmse = np.sqrt(mse)
rmse



# 전체 유저의 프로필 만들기
## 전체 데이터로 확장
## ratings 테이블에 장르 정보 붙이기
ratings = ratings.merge(genres, left_on ='movieId', right_index=True)
# print(ratings)
# ratings.sample()
train, test = train_test_split(ratings, test_size=0.1, random_state=42)

user_profile_list=[]
for userId in train['userId'].unique():
    user = train[train['userId'] == userId]
    X_train = user[genres.columns] # feature, X
    y_train = user['rating'] # label, y

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    user_profile_list.append([reg.intercept_, *reg.coef_]) # *는 리스트를 flatten 하게 만들어주는 파이썬의 기능

# print(user_profile_list)
user_profile = pd.DataFrame(user_profile_list, index=train['userId'].unique(), columns=['intercept', *genres.columns])
pd.set_option('float_format', '{:f}'.format )
# print(user_profile)











