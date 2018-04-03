# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:08:17 2017

@author: ljc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Times New Roman'

DataSet = pd.read_csv('AppliancesLoadForecasting.csv')
dataset = DataSet.Appliances
dataset = dataset[42:42+144*136]
#print(dataset.head(10))
daydata = dataset.values.reshape(-1,144)
dayload = daydata.sum(axis=1)
#print(dayload)
date = pd.date_range('20160112',periods=136)
dayload = pd.DataFrame(dayload,index=date,columns=list(['dayload']))
#dayload.to_csv('dayload.csv')

dayload = pd.read_csv('dayload.csv')
plt.figure(figsize=(10,4))
plt.plot(dayload.dayload,color='k')
plt.xticks([0,30,60,90,120],['2016-01-12','2016-02-12','2016-03-12','2016-04-12','2016-05-12'],rotation=45)
#plt.yticks([0,10000,20000],['0','10000','20000'])
#plt.xlabel('Time')
plt.ylabel('Wh')
#plt.legend(['dayload'])
#plt.savefig('图8日负荷.svg',bbox_inches='tight')
plt.show()