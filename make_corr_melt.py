import pandas as pd

corr = pd.read_csv("data/outputs/corr.csv")
#corr = corr.drop(['Unnamed: 0'], axis=1)
print(corr)

corr_melt = pd.melt(corr, id_vars='Unnamed: 0', var_name = 'IDX', value_name='CORR_VAL')




col = corr_melt.columns.to_numpy()
corr_melt = corr_melt[col[[1,0,2]]]
corr_melt.rename(columns = {'Unnamed: 0':'ITEM'},inplace=True)
print(corr_melt)

corr_melt.to_csv("./data/outputs/corr_melt.csv")