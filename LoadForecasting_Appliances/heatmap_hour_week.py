# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:12:33 2017

@author: ljc
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.sans-serif'] = ['Times New Roman'] #指定默认字体 

dataset = pd.read_csv('hourload.csv')
data = dataset.hourload.values

data1 = data[144:816]
#print(data1)

data2 = data1.reshape(4,168)
data3 = data2[:1,:].reshape(7,24)
data4 = data2[1:2,:].reshape(7,24)
data5 = data2[2:3,:].reshape(7,24)
data6 = data2[3:4,:].reshape(7,24)

plt.figure(figsize=(10,10))
plt.subplot(4,1,1)
df = pd.DataFrame(data3)
xticks = [i for i in range(24)]
yticks = [i for i in range(1,8)]
#yticks = [7,6,5,4,3,2,1]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=xticks)
#sns.set(font_scale=1.5,font='Times New Roman')    #加上字体型号会变
#plt.xticks([0,4,8,12,16,20,23],[0,4,8,12,16,20,23])    #刻度不居中
plt.ylabel('Week1')

plt.subplot(4,1,2)
df = pd.DataFrame(data4)
xticks = [i for i in range(24)]
yticks = [i for i in range(1,8)]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=xticks)
plt.ylabel('Week2')

plt.subplot(4,1,3)
df = pd.DataFrame(data5)
xticks = [i for i in range(24)]
yticks = [i for i in range(1,8)]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=xticks)
plt.ylabel('Week3')

plt.subplot(4,1,4)
df = pd.DataFrame(data6)
xticks = [i for i in range(24)]
yticks = [i for i in range(1,8)]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=xticks)
plt.xlabel('Day(h)')
plt.ylabel('Week4')
#plt.savefig('图7heatmap.svg',bbox_inches='tight')
plt.tight_layout()
plt.show()





