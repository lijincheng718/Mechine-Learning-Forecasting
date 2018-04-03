# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 19:18:10 2017

@author: ljc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'

DataSet = pd.read_csv('data_input.csv')
dataset = DataSet.t
dataset = dataset[39:39+144*136]
data = dataset.values.reshape(-1,6)
data = data.sum(axis=1)

date = DataSet.date[39:39+144*136:6]
hourload = pd.DataFrame(data,index=date,columns=list(['hourload']))
hourload.to_csv('hourload.csv')

hourload = pd.read_csv('hourload.csv')
plt.figure(figsize=(10,4))
plt.plot(hourload.hourload,color='k')
plt.xticks([0,720,1440,2160,2880],['2016-01-12','2016-02-12','2016-03-12','2016-04-12','2016-05-12'],rotation=45)
plt.yticks(rotation=0)
#plt.xlabel('Time')
plt.ylabel('Wh')
#plt.legend(['hourload'])
#plt.savefig('图9完整时负荷.svg',bbox_inches='tight')

plt.figure(figsize=(10,4))
plt.plot(hourload.hourload[:168],color='k')
plt.xticks([0,24,48,72,96,120,144],['2016-01-12','2016-01-13','2016-01-14','2016-01-15','2016-01-16','2016-01-17','2016-01-18'],rotation=45)
#plt.savefig('图10一周时负荷1.svg',bbox_inches='tight')
plt.show()
