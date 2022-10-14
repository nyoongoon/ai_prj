import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

## Read Data
movies = pd.read_csv('./data/ml-latest-sm/movies.csv')
ratings = pd.read_pickle('./data/ml-latest-sm/ratings_updated.p')
genres = pd.read_pickle('./data/ml-latest-sm/genrs.p')

## Preprocessing (데이터 전처리)
## ratings와 genres 붙이기
## how는 inner조인이 디폴트. 오른쪽 테이블에서 movieId칼럼이 따로없고, 인덱스로 들어가 있기있기 떄문에 right_index=True
ratings = ratings.merge(genres, how='inner', left_on='movieId', right_index=True)
ratings.sample()
## 0값 없애기 0 => nan
ratings = ratings.replace(0, np.nan)

## Train Test Split
from sklearn.model_selection import train_test_split
train, test = train_test_split(ratings, random_state=42, test_size=0.1)
# print(train.shape)
# print(test.shape)



# 유저 프로필 구성


## Item Profile 아이템을 설명.
# genres

## User Profile 유저를 설명 => train 데이터를 이용해서 연산.
genre_cols = genres.columns # 장르 이름 가져옴
## 유저프로필 => 각 장르에 대해서 그 유저가 평균평점을 몇점으로 남겼는지 .. ! => 유저 | 무비 | 평점 | 장르 목록(1점) => 평점 x 장르 목록
for cols in genre_cols:
    train[cols] = train[cols] * train['rating']

## 한 유저가 장르에 대한 평점을 평균 어떻게 주었나 ?
train.groupby('userId')['Action'].mean() # 각 유저별로 Action 아이템에 준 평점의 평균.
user_profile = train.groupby('userId')[genre_cols].mean()# 각 유저별로 각 장르에 준 평점의 평균.

user_profile.sample()

#print(test)

# Predict 예측 해보기
sample = test.loc[99981]
sample_user = sample['userId']
sample_user_profile = user_profile.loc[sample_user] # 샘플 유저에 대한 유저 프로파일

#print(sample['movieId'])
#print(sample[genre_cols])
## 유저의 장르별 평균 => 해당 영화가 가진 장르를 해당 유저의 장르 점수 더하고 개수 나누기 -> 그 영화의 예상 평점.
# print(sample_user_profile * sample[genre_cols]) # 각 장르에 몇점을 주는가
# print(sample_user_profile * sample[genre_cols].mean()) # 각 장르의 점수 평균 => 예상점수



# Predict 예측 해보기 (전체 데이터)

predict = []
for idx, row in test.iterrows():        # 인덱스 정보와 row정보를 for문 돌림
    user = row['userId']
    # user profile * item profile
    predict.append((user_profile.loc[user] * row[genre_cols]).mean())

test['predict'] = predict
print(test['predict'])
test['predict'].isnull() # 널값이 존재하는 데이터 있음 => 장르 이전에 본 적이 없는 경우 => globalmean 넣어주기
test.loc[test['predict'].isnull(), 'predict'] = train['rating'].mean()

# 모델 평가
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test['rating'], test['predict'])
rmse = np.sqrt(mse)
print(rmse)





















