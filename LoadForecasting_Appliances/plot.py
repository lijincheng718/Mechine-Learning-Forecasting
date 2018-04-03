# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 10:39:31 2017

@author: ljc
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 20
#plt.rcParams['axes.grid'] =False
mpl.rcParams['font.sans-serif'] = ['Times New Roman'] #指定默认字体


dataset = pd.read_csv('cor_daytime.csv')
data = dataset.iloc[:-1,-1:].values
data = data[::-1]
plt.figure(figsize=(10,4))
plt.xlabel('Day index')
plt.ylabel('r')
plt.xticks([i for i in range(14)],['d-1','d-2','d-3','d-4','d-5','d-6','d-7','d-8','d-9','d-10','d-11','d-12','d-13','d-14'])
plt.plot(data,'k')
plt.plot(data,'ro')
#plt.savefig('图5日指数.png',dpi=200,bbox_inches='tight')
plt.show()


'''
dataset = pd.read_csv('cor_time.csv',header=None)
data = dataset.values

plt.figure(figsize=(10,4))
plt.xlabel('Time index')
plt.ylabel('r')
plt.xticks([i for i in range(14)],['t-1','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12','t-13','t-14'])
plt.plot(data,'k')
plt.plot(data,'ro')
plt.savefig('图5时刻指数.png',dpi=200,bbox_inches='tight')
plt.show()
'''

