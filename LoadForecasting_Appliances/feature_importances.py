# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:19:26 2017

@author: ljc
"""

from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'


# load data
#dataframe = read_csv('day_index.csv', header=0)
#dataframe = read_csv('hour_index.csv', header=0)
dataframe = read_csv('min_index.csv', header=0)
array = dataframe.values
# split into input and output
X = array[:,0:-1]
y = array[:,-1]
# fit random forest model
model = RandomForestRegressor(n_estimators=50, random_state=1)
model.fit(X, y)
# show importance scores
print(model.feature_importances_)
# plot importance scores
names = dataframe.columns.values[0:-1]
ticks = [i for i in range(len(names))]
plt.figure(figsize=(10,4))
plt.bar(ticks, model.feature_importances_)
plt.xticks(ticks, names)
plt.xlabel('Time index')
plt.ylabel('Important index')
plt.savefig('图5日期指数.png',dpi=200,bbox_inches='tight')
plt.show()
