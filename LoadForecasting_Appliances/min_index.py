# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:11:28 2017

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

DataSet = pd.read_csv('time_data.csv')
dataset = DataSet.iloc[:,:]

plt.figure(figsize=(10,8))
corrmat = dataset.corr()
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