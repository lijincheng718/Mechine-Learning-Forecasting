# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 10:21:06 2017

@author: ljc
"""

'''
import pandas as pd
dataset = pd.read_csv('data_selection.csv')
data1 = dataset.iloc[:-3,1:2].values
data2 = dataset.iloc[1:-2,1:2].values
data3 = dataset.iloc[2:-1,1:2].values
data4 = dataset.iloc[3:,:].values
data1 = pd.DataFrame(data1)
data2 = pd.DataFrame(data2)
data3 = pd.DataFrame(data3)
data4 = pd.DataFrame(data4)
data = pd.concat([data4,data3,data2,data1],axis=1)

#data.to_csv('data_input.csv',index=False,header=['date','t','lights','T2','Windspeed','RH1','T3','RH6','RH8','RHO','t-1','t-2','t-3'])
'''


import pandas as pd
data = pd.read_csv('data_input.csv')
trainlen = int(0.8*(len(data.t)))
testlen = len(data.t)-trainlen
print(trainlen)
print(testlen)