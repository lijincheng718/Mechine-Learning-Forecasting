# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:50:23 2017

@author: ljc
"""

# Appliances energy prediction --read dataset

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
from matplotlib.font_manager import FontProperties as FP

dataset = pd.read_csv('AppliancesLoadForecasting.csv')
#print(dataset.head())
#data = dataset.values
#print(data[:5])
ad1 = dataset.Appliances
ad2 = ad1[43:1051]

font1 = {'family': 'Times New Roman',  
        'color':  'k',  
        'weight': 'normal',  
        'size': 24,  
        } 
font2 = FP('Times New Roman', size=24)

plt.figure(figsize=(10,4))
plt.xlabel('Time(2016)',fontdict=font1)
plt.ylabel('Wh',fontdict=font1)
plt.xticks([0,4363,8539,13003,17350],['1/11','2/11','3/11','4/11','5/11'],fontproperties=font2)
plt.yticks([0,200,400,600,800,1000],['0','200','400','600','800','1000'],fontproperties=font2)
#plt.axis('off') #去掉坐标系
plt.plot(ad1,color='k',lw=1)
#plt.savefig('图1完整时序图.svg',bbox_inches='tight')

plt.figure(figsize=(10,4))
plt.xlabel('Week',fontdict=font1)
plt.ylabel('Wh',fontdict=font1)
plt.xticks([43,187,331,475,619,763,907],['1','2','3','4','5','6','7'],fontproperties=font2)
plt.yticks([0,200,400,600,800,1000],['0','200','400','600','800','1000'],fontproperties=font2)
#plt.axis('off') #去掉坐标系
plt.plot(ad2,color='k',lw=1)
#plt.savefig('图2一周时序图.svg',bbox_inches='tight')
plt.show()