#%%
"""
    Data importing and simpl observations about data
"""
import pandas as pd 
import numpy as np

train =  pd.read_csv("titanic/train.csv")

train.head()
train.describe()

#%%

# nan data elimination detection
cabin_data = tarin.Cabin
not_na = pd.Series(cabin_data.array,index=cabin_data.isna())[False]

not_na.head()


# %%
"""
    Printing histogram for all columns
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

less = []
more = []

for ele in train.columns:
    if ele not in ["PassengerId","Name"]:
        temp = train[ele]
        dist = temp.value_counts()
        print(ele,end="=>\t")
        if len(dist.keys())<10:
            print("Less Cateogry (Smaller than 10)")
            less.append(ele)
            plt.figure(figsize=(10,6))
        else:
            print("Many Category (More than 60)")
            more.append(ele)
            plt.figure(figsize=(100,6))
        sns.barplot(x=dist.keys(),y=dist.values)
        plt.show()

# %%
"""
 Unused observation of data wich varies too much
"""
print(train.loc[:,more].describe())

# %%
"""
    calculating pmf 
"""

n = len(train)
pmf = {}

for ele in train.columns:
    if ele not in ["PassengerId","Name"]:
        temp = train[ele]
        dist = temp.value_counts()
        # pmf[ele] = map(lambda x: {x.:} )
        pmf[ele] = {}
        for key in dist.keys():
            pmf[ele][key] = {key:round(dist[key]/n,2)}
            # print(key)
            # pmf[ele][key] = {key:dist[key]/n}

print(pmf)

# %%

for ele in pmf:
    for key in ele:
        sns.barplot(x=)
