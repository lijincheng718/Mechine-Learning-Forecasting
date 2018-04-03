# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:59:02 2017

@author: ljc
"""

# autocorrelation
#计算负荷序列的自相关系数

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = (10,4)          
DataSet = pd.read_csv('AppliancesLoadForecasting.csv')
dataset = DataSet.Appliances[42:186]


'''
DataSet = pd.read_csv('hourload.csv')
dataset = DataSet.hourload[22:720:24]
'''

'''
DataSet = pd.read_csv('dayload.csv')
dataset = DataSet.dayload[8:]
'''

plot_acf(dataset)
#plot_pacf(dataset)
plt.title('')
#plt.xticks([0,200,400,600,800,1000],['t','t-200','t-400','t-600','t-800','t-1000'])
plt.yticks([-0.2,0,0.5,1.0],[-0.2,0,0.5,1.0])
#plt.savefig('图11自相关.png',dpi=200,bbox_inches='tight')




