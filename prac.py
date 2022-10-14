import pandas as pd
ratings = pd.read_pickle('./data/ml-latest-sm/ratings_updated.p')
genres = pd.read_pickle('./data/ml-latest-sm/genrs.p')


print(ratings) # 데이터프레임

user414 = ratings[ratings['userId'] == 414]
#user414.sample()

user414 = user414.merge(genres, left_on='movieId', right_index=True)

print(user414)

print(user414[genres.columns])