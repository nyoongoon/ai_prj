import pandas as pd

ratings = pd.read_pickle('./data/ml-latest-sm/ratings_updated.p')
print(ratings)

# RMSE
rating_example = [[4, 3.5], [5, 5], [0.5, 1], [3, 5]]
rating_example = pd.DataFrame(rating_example, columns=['Actual', 'Predict'])
# error = Actual - Predict
rating_example['error'] = rating_example['Actual'] - rating_example['Predict']
# squared error=> rating_example['error'].sum()으로 처리 시, 양수의 차이와 음수의 차이가 상쇄되면서 차이가 늘어나는게 아니라 줄어드므로. 제곱(mse)해주고, 제곱근까(rmse)지
rating_example['squared error'] = rating_example['error'] ** 2;
# 모델의 전체적인 퍼포먼스를 알 수 있게 해주는 mean_squared_error => mse // 제곱근 처리 시 => root_mean_squared_error => rmse
mse = rating_example['squared error'].mean()
import numpy as np

# root mean squared error : rmse
rmse = np.sqrt(mse)
# ================= #
# RMSE with sklearn => 패키지가 매우 크므로 특정 모듈만 불러와서 사용
from sklearn.metrics import mean_squared_error

mean_squared_error(rating_example['Actual'], rating_example['Predict'])
rmse = np.sqrt(mse)

# Train Test Split
# 성과를 측정하기 위해 데이터 셋을 두 개로 나눔 => 모델을 훈련하는 데이터와 성능을 테스트하는 데이터는 분리되야함. => 오버피팅 방지
from sklearn.model_selection import train_test_split

# randon_state => 섞는 방식 인덱스 => 설정해주지 않으면 실행할 때마다 다른 방식으로 섞음 => 그렇게 되면 만들어놓은 rmse등의 값 복원이 안 (42는 임의값)
# test_size => 전체 데이터 중에서 테스트를 위한 값 퍼센트 지정.
train, test = train_test_split(ratings, random_state=42, test_size=0.1)

# 가장 간단한 예측하기
predictions = [0.5] * len(test)  # 모든 결과를 0.5로 평가하는 모델
# mean_squared_error(actual, predict)
mse = mean_squared_error(test['rating'], predictions)
rmse = np.sqrt(mse)  # 모든 결과를 0.5를 평가하는 모델과 테스트 데이터셋을 비교하여 평가했을 경우 3.xx의 차이를 보인다.
# print(rmse)

# 데이터의 평균으로 예측하기
# 주의점 : train 데이터의 평균으로 **test 데이터의 평균을 예측**해야함.
rating_avg = train['rating'].mean()
predictions = [rating_avg] * len(test)
mse = mean_squared_error(test['rating'], predictions)
rmse = np.sqrt(mse)
rmse

# 사용자 평점 기반 예측하기
# train에 해당 사용자에 대한 평점기록이 전혀 없다면 ?
# 각 유저별 평균평점 구하기 => 근거 : 한 유저는 일관된 점수로 평가할 것이다.
users = train.groupby('userId')['rating'].mean().reset_index()  # reset_index() => 표형식으로 바꾸기
## rating에 대한 명칭이 중복되서 다시 명명
users = users.rename(columns={'rating': 'predict'})
users[:1]
## 생성된 users테이블과 test 테이블 합치기.
## left를 기준(현재test) => users테이블을 train 데이터를 기준으로 했기 때문에 users에 있는 데이터가 test에 없을 수도 있음.
## test테이블에 있는 것이 users에 없을 수도 그 경우에도 test테이블 기준으로.
predict_by_users = test.merge(users, how='left', on='userId')  ## how=>어떤 테이블을 기준으로. #on => 어떤 칼럼으로 합칠것인지
predict_by_users.isnull().sum()  # nan, null값 확인 => true이면 1이므로 0이 아니면 문제가 있는 것.
mse = mean_squared_error(predict_by_users['rating'], predict_by_users['predict'])
rmse = np.sqrt(mse)
rmse  ## rmse 개선됨

## 예측의 근거 => train['rating'].std() 표준편차 계산. 데이터들이 얼마나 분포되어 있는지
train.groupby('userId')['rating'].std()  ## => 값이 작으면 작을 수록 특정 유저가 모든 영화에 대해 모든 영화에 대해 비슷한 평점을 주는 것.
## => 각각의 유저 중 표준편차가 큰 편이 있으므로 => 전체 유저 평균보다, 각 유저별 평균을 가지고 계산하는 게 더 예측이 퀄리티가 좋움


# 영화 평점 기반 예측하기
# => 하나의 영화에 대한 유저들의 취향이 비슷할 것이다.
# train에 해당 영화에 대한 평점기록이 전혀 없다면 ?
movies = train.groupby('movieId')['rating'].mean().reset_index()  ## => 값이 작으면 작을 수록 특정 유저가 모든 영화에 대해 모든 영화에 대해 비슷한 평점을 주는 것.
movies = movies.rename(columns={'rating': 'predict'})

predict_by_movies = test.merge(movies, how='left', on='movieId')
predict_by_movies.sample()
predict_by_movies.isnull().sum() # => 비어있는 값이 많이 발견됨 !!train에 해당 영화에 대한 평점기록이 전혀 없는 경우 ==> 전체 평균으로 채워넣어주기
# NaN (Not a Number)
# Pandas location =>
# df.loc[index(filtering), column] => 특정 인덱스, 특정 칼럼에 값 넣어주기

predict_by_movies.loc[predict_by_movies['predict'].isnull(), 'predict'] = train['rating'].mean()
print(predict_by_movies)
predict_by_movies.isnull().sum() # => 널값 확인 ok
## 모델평가
mse = mean_squared_error(predict_by_movies['rating'], predict_by_movies['predict'])
rmse = np.sqrt(mse)

# 장르별 평균으로 유저 프로필 만들기
## 조의 유저 프로필 구하는 방법 -> 조의 평점과, 조가 평점을 준 영화들의 아이템 프로필을 사용해서 계산.
### 코미디 : (3.5) / 1 = 3.5
### 스릴러 : (5.0+1.0) / 2 = 3.0
### 액션 : (4.5 + 4.5 + 1.0) / 3 = 3.17

## 조의 기생충 예상 평점(안본경우) => 조의 코미디 점수 * 기생충의 코미디 더비 변수 값 + 조의 스릴러 점수 * 기생충의 스릴러 더미변수값 + 조의 액션점수 * 기생충의 액션 더미변수 값(0) / 기생충에 포함된 장르 개수 (장르더미변수 값의 값)

# cf) => 장르를 과목으로 선정해야할까?

## Cold-Start 문제 : 마미는 코미디 장르가 포함되는 영화에 평점을 준 적이 한 번도 없기 때문에, 마미의 유저 프로필 중 코미디 장르에 대한 값은 NaN. 따라서 오직 코미디 장르만 가진 '정직한 후보'에 대한 마미의 예상평점은 구할 수가 없다.
## +) 장르가 하나도 포함되지 않는 영화를 추천해야할 경우...
## Cold-Start의 경우 => global mean으로 해결하기. vs 마미의 데이터 평균으로 넣기


