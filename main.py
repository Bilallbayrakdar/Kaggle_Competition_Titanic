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
cabin_data = train.Cabin
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

print(pmf.keys())

# %%

train.Fare.quantile(0.9)
# %%
"""
 random exponential function 
"""

for i in [1,10,100,1000,10000000000]:
    x = sorted(np.random.exponential(scale=2,size=100))
    y = [ele for ele in range(len(y))]
    sns.lineplot(x=x,y=y)
    print(i)
    plt.show()
# %%
"""
normal distribution function
"""

from scipy.stats import norm
test_cdf = pd.DataFrame(
    {
        'test1' : [norm.cdf(x=ele,loc=1,scale=0.5) for ele in np.linspace(-1,4,num=100)],
        'test2' : [norm.cdf(x=ele,loc=2,scale=0.4) for ele in np.linspace(-1,4,num=100)],
        'test3' : [norm.cdf(x=ele,loc=3,scale=0.3) for ele in np.linspace(-1,4,num=100)]
    }
)
test_pdf = pd.DataFrame(
    {
        'test1' : [norm.pdf(x=ele,loc=0,scale=1) for ele in np.linspace(-1,4,num=100)],
        'test2' : [norm.pdf(x=ele,loc=2,scale=0.4) for ele in np.linspace(-1,4,num=100)],
        'test3' : [norm.pdf(x=ele,loc=3,scale=0.3) for ele in np.linspace(-1,4,num=100)]
    }
)
print("CDF")
sns.lineplot(data=test_cdf)
plt.show()
print("PDF")
sns.lineplot(data=test_pdf)
plt.show()

# %%
#%%
import math
import sys
import pandas
import numpy as np

import thinkstats2
import thinkplot
import brfss

import matplotlib.pyplot as plt
import seaborn as sns
# %%
"""
EX 1 of  the thinkstats 5. Chapter

Trying to find percentage of the men who are
capable of being member of Blue Man Group
"""

ex_1_data = brfss.ReadBrfss()

heigth_men = ex_1_data.htm3[(ex_1_data["htm3"]<200) & (ex_1_data["htm3"]>160) & (ex_1_data["sex"]==1)]
heigth_men.hist(bins=12)
sort_heigth = sorted(heigth_men)

m_mu = round(heigth_men.mean(),3)
m_sig = round(heigth_men.std(),3)

min_height = round(5.11 *30.5,3)
max_height = round(6.1 * 30.5,3)

# %%
"""
Plotting CDF of the Men's heigth
"""
heigth_men.describe()

x_val = np.linspace(-1,1,num=len(heigth_men))
hist = heigth_men.value_counts()

sns.lineplot(x=x_val,y=sorted(heigth_men))
plt.show()
# %%
blue = [ele for ele in heigth_men if (ele>=min_height) and (ele<=max_height)]
result = len(blue)/len(heigth_men)*100
print("Percentage of the man who are cabaple of get accepted Blue Man Group: %{}".format(result))