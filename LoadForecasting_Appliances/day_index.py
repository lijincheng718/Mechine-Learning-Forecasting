# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:14:46 2017

@author: ljc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.size'] = 24
#plt.rcParams['axes.grid'] =False
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #指定默认字体
import warnings
warnings.filterwarnings('ignore')


dataset = pd.read_csv('AppliancesLoadForecasting.csv')
minload = dataset.Appliances
minload = minload[180::144]
df = pd.DataFrame()
for i in range(14,0,-1):
    df['t-'+str(i)] = minload.shift(i)
df['t'] = minload.values
df = df[15:]
df.to_csv('min_index.csv',index=False)
DataSet = pd.read_csv('min_index.csv')
dataset = DataSet

plt.figure(figsize=(10,8))
corrmat = dataset.corr()
cor = pd.DataFrame(corrmat)
cor_t = cor.iloc[:,14:15]
cor_t.to_csv('E:\\AppliancesLoadForecasting\\dayindex\\23.csv',header=None,index=False)
cor_t.plot()
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