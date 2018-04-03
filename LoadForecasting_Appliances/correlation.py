# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:54:47 2017

@author: ljc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['font.size'] = 15
#plt.rcParams['axes.grid'] =False
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #指定默认字体
 
import warnings
warnings.filterwarnings('ignore')

# 查看数据集前5行
DataSet = pd.read_csv('AppliancesLoadForecasting.csv')
dataset = DataSet.iloc[:,1:]
data = dataset.Appliances
# 计算变量相关性
plt.figure(figsize=(10,8))
corrmat = dataset.corr()
#print(corrmat.head())
#saleprice correlation matrix
#k = 10 #number of variables for heatmap
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Appliances')['Appliances'].index
cm = np.corrcoef(dataset[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cmap='rainbow', annot=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
#plt.savefig('图6单变量相关性10.svg',bbox_inches='tight')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
plt.tight_layout()
plt.show()