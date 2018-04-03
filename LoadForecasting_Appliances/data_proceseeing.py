# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:10:41 2017

@author: ljc
"""

import pandas as pd

'''
dataset = pd.read_csv('AppliancesLoadForecasting.csv')
data = dataset.Appliances
data1 = data[42:-144*14].values
data2 = data[42+144*1:-144*13].values
data3 = data[42+144*2:-144*12].values
data4 = data[42+144*3:-144*11].values
data5 = data[42+144*4:-144*10].values
data6 = data[42+144*5:-144*9].values
data7 = data[42+144*6:-144*8].values
data8 = data[42+144*7:-144*7].values
data9 = data[42+144*8:-144*6].values
data10 = data[42+144*9:-144*5].values
data11= data[42+144*10:-144*4].values
data12 = data[42+144*11:-144*3].values
data13 = data[42+144*12:-144*2].values 
data14 = data[42+144*13:-144*1].values
data15 = data[42+144*14:].values
             
data_1 = pd.DataFrame(data1)
data_2 = pd.DataFrame(data2)
data_3 = pd.DataFrame(data3)
data_4 = pd.DataFrame(data4)
data_5 = pd.DataFrame(data5)
data_6 = pd.DataFrame(data6)
data_7 = pd.DataFrame(data7)
data_8 = pd.DataFrame(data8)
data_9 = pd.DataFrame(data9)
data_10 = pd.DataFrame(data10)
data_11 = pd.DataFrame(data11)
data_12 = pd.DataFrame(data12)
data_13 = pd.DataFrame(data13)
data_14 = pd.DataFrame(data14)
data_15 = pd.DataFrame(data15)

time_data = pd.concat([data_1,data_2,data_3,data_4,data_5,data_6,data_7,data_8,data_9,data_10,data_11,data_12,data_13,data_14,data_15],axis=1)
print(time_data)
time_data.to_csv('time_data.csv',index=False)
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 24
#plt.rcParams['axes.grid'] =False
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #指定默认字体
import warnings
warnings.filterwarnings('ignore')

DataSet = pd.read_csv('time_data.csv')
dataset = DataSet.iloc[:,:]

plt.figure(figsize=(10,8))
corrmat = dataset.corr()
cor = pd.DataFrame(corrmat)
cor.iloc[:,-1:].to_csv('cor_daytime.csv')
'''
#print(corrmat.head())
#saleprice correlation matrix
#k = 10 #number of variables for heatmap
k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 't')['t'].index
cm = np.corrcoef(dataset[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
#plt.savefig('图5时刻相关性.png',dpi=200,bbox_inches='tight')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
'''