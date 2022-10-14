import pandas as pd
import openpyxl
pd.set_option('display.max_row', 10000)
pd.set_option('display.max_columns', 10000)

items = pd.read_csv("../data_ai/test.csv")

print(items.shape)

import seaborn as sns
import matplotlib.pyplot as plt #seaborn figure 크기 조절을 위함.

print(items['gidx'])

# gidx 리스트로 저장. 평탄화

gidx_list = list(items['gidx'].apply(lambda x: x.split("|"))) # 리스트로 저장
# print(gidx_list[:3]) 테스트

# Flatten list of list
flat_list = []
for sublist in gidx_list:
    for item in sublist:
        flat_list.append(item)

gidx_unique = list(set(flat_list)) # 중복 제거
# print(gidx_unique)
# gidx_unique.head()

items_dummies = items['gidx'].str.get_dummies(sep='|')
items_dummies.to_pickle('./data/outputs/test.p')

print(items_dummies)

# items_dummies.head()
#
# plt.figure(figsize=(30, 15))
# sns.heatmap(items_dummies.corr(), annot=True)
# plt.savefig('test-result.png')
# plt.show()

# print(items_dummies.corr())
correl = items_dummies.corr()
#correlation=[]
#correlation.append(correl)
# correl.to_csv("./data/outputs/corr.csv")
correl.to_excel("./data/outputs/corr.xlsx")