import matplotlib
import seaborn as seaborn
from pasta.augment import inline
# import sys
#
# def sum(v1,v2):
#     result = int(v1) + int(v2)
#     print(result)
#
#
# def main(argv):
#     sum(argv[1], argv[2])
#
# if __name__ == "__main__":
#     main(sys.argv

import pandas as pd

print(pd.__version__)

movies = pd.read_csv('./data/ml-latest-sm/movies.csv', index_col='movieId')
print(movies.shape) # 데이터 갯수, 컬럼 갯수 출력

movies.head() # 상위 다섯개
# movies.tail()
# movies.sample() 랜덤추출
movies.columns

movies.to_csv('./data/ml-latest-sm/save_test.csv')


# 개봉연도 데이터 정제하기(데이터 전처리)
movies['year'] = movies['title'].str.extract('(\(\d\d\d\d\))')
movies['year'] = movies['year'].str.extract('(\d\d\d\d)')
movies['year'].unique()


# 결측값 핸들링 하기 (nan)
# NaN(Not a Number), 결측치
movies[movies['year'].isnull()]
movies['year'] = movies['year'].fillna('2050')
movies['year'].unique


# 데이터에서 가장 많이 출현한 개봉연도 찾기.
print(movies['year'].value_counts())


# 데이터 시각화 라이브러리 seaborn -> matplotlib을 간편하게 래핑
# %matplotlib inline

import seaborn as sns
import matplotlib.pyplot as plt #seaborn figure 크기 조절을 위함.

# plt.figure(figsize=(50, 10))
# sns.countplot(data=movies, x='year')
# plt.show()


# genres 분석
print(movies['genres'])
sample_genre = movies['genres'][1]

sample_genre.split("|")

genres_list = list(movies['genres'].apply(lambda x: x.split("|"))) # 리스트로 저장
genres_list[:3]

# Flatten list of list
flat_list = []
for sublist in genres_list:
    for item in sublist:
        flat_list.append(item)

# print(set(flat_list)) # 중복 제거
genres_unique = list(set(flat_list)) # 중복 제거
len(genres_unique) # 장르 갯수 세기

# 장르형 데이터 숫자형으로 변환하기.
'Adventure' in sample_genre # Adventure 장르 있는지 확인
# print(movies['genres'].apply(lambda x: 'Adventure' in x))
movies['Adventure'] = movies['genres'].apply(lambda x: 'Adventure' in x)
movies['Comedy'] = movies['genres'].apply(lambda x: 'Adventure' in x) # -> 복잡함 -> 판다스 이용하면 편함

genres_dummies = movies['genres'].str.get_dummies(sep='|')
genres_dummies.to_pickle('./data/ml-latest-sm/genrs.p') # 피클 자료구조로 저장함.

print(genres_dummies)

# 데이터 상관관계 분석 -> 같이 묶이는 정도 파악.
# 두 장르의 관계가 1에 가깝다는 것은 : 두 장르가 자주 같이 출현.
# print(genres_dummies.corr())
# plt.figure(figsize=(30, 15))
# sns.heatmap(genres_dummies.corr(), annot=True) # 시각화 함수
# plt.show()

ratings = pd.read_csv('./data/ml-latest-sm/ratings.csv')
ratings.sample()
print(ratings.shape)
print(len(ratings['userId'].unique())) # 610명의 유저 데이터
print(len(ratings['movieId'].unique())) # 9724개의 영화 데이터
print(ratings['rating'].describe())

# ratings['rating'].hist()

# 사람들은 평균적으로 몇 개의 영화에 대해서 rating을 남겼는가?
# print(ratings.groupby('userId')['movieId'].count())
users = ratings.groupby('userId')['movieId'].count()
# values 평가한 영화 수
sns.displot(users.values)
plt.show() # 0이 높고 갈수록 줄어드는 분포 => power law distribution, 멱함수 분포 => 평균값이 전체 데이터를 잘 설명하지 못함.

# 유저별 평점 패턴 분포
### 사람들이 많이 보는 영화는?
films = ratings.groupby('movieId')['userId'].count()
films.sort_values() # 많이 본 영화 오름차순
### 개별 평점보기
frozen = ratings[ratings['movieId'] == 106696]
# frozen['rating'].hist()
# plt.show()

# print(ratings[ratings['userId'] == 567])
#ratings.loc[ratings['userId'] == 567, 'rating'].hist()
#plt.show()

# 나의 평점 데이터 기록
### timestamp 컬럼처리
from datetime import datetime

ratings['timestamp'] = ratings['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
# print(ratings.sample())

### 내 데이터 붙이기
myratings = pd.read_csv('./data/ml-latest-sm/my-ratings.csv')
# print(myratings.sample())
# print(myratings['timestamp']) // 타입이 object -> datetime으로 변환해야함
myratings['timestamp'] = pd.to_datetime(myratings['timestamp'])
ratings_concat = pd.concat([ratings, myratings]) # 데이터 이어붙이기 !
# 데이터 저장
ratings_concat.to_pickle('./data/ml-latest-sm/ratings_updated.p')

# 평가지표 RMSE

print(ratings)








