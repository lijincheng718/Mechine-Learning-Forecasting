# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:18:01 2017

@author: ljc
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#plt.rcParams['font.size'] = 15
#plt.rcParams['axes.grid'] =False
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #指定默认字体
 
from matplotlib.font_manager import FontProperties as FP
import warnings
warnings.filterwarnings('ignore')

# 查看数据集前5行
DataSet = pd.read_csv('AppliancesLoad.csv')
dataset = DataSet.iloc[:,1:]
#print(dataset.head())
#print(dataset.columns)
data = dataset.Appliances
#print(data.head())
#print(data.describe())

# 统计各列缺失值总数
#missing = dataset.isnull().sum()
#print(missing)

# 绘图查看
font1 = {'family': 'Times New Roman',  
        'color':  'k',  
        'weight': 'normal',  
        'size': 15,  
        } 
font2 = FP('Times New Roman', size=15)

'''
plt.figure(figsize=(10,4))
plt.hist(data,bins=50)
plt.xlabel('Wh',fontdict=font1)
plt.ylabel('Frequency',fontdict=font1)
plt.xticks([0,200,400,600,800,1000],['0','200','400','600','800','1000'],fontproperties=font2)
plt.yticks([0,1000,2000,3000,4000,5000,6000,],['0','1000','2000','3000','4000','5000','6000'],fontproperties=font2)
plt.savefig('图3直方图.svg',bbox_inches='tight')
'''


'''
plt.figure(figsize=(2,10))
sns.boxplot(data=dataset.iloc[:,1:2])
#plt.xlabel('load',fontdict=font1)
plt.xticks([])
plt.xlabel('Appliances',fontdict=font1)
plt.ylabel('Wh',fontdict=font1)
plt.yticks([0,200,400,600,800,1000],['0','200','400','600','800','1000'],fontproperties=font2)
#plt.savefig('图4箱线图.svg',bbox_inches='tight')
'''

# 概率图
sns.distplot(dataset['Appliances'], fit=norm)
fig = plt.figure()
res = stats.probplot(dataset['Appliances'], plot=plt)
fig = plt.figure()
dataset['Appliances'] = np.log(dataset['Appliances'])
#histogram and normal probability plot
sns.distplot(dataset['Appliances'], fit=norm)
fig = plt.figure()
res = stats.probplot(dataset['Appliances'], plot=plt)

plt.figure()
f = pd.melt(dataset)
g = sns.FacetGrid(f, col="variable",  col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

plt.tight_layout()
plt.show()
