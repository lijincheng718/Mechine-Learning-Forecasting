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

dataset = pd.read_csv('data_input.csv')
data = dataset.t.values     #目标负荷值
start = 903     #星期一开始index
end = 903+144*28

data1 = data[start:end]
data2 = data1.reshape(4,144*7)
data3 = data2[:1,:].reshape(7,144)
data4 = data2[1:2,:].reshape(7,144)
data5 = data2[2:3,:].reshape(7,144)
data6 = data2[3:4,:].reshape(7,144)

plt.figure(figsize=(16,9))
plt.subplot(4,1,1)
df = pd.DataFrame(data3)
yticks = [i for i in range(1,8)]
#yticks = [7,6,5,4,3,2,1]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=[])
#sns.set(font_scale=1.5,font='Times New Roman')    #加上字体型号会变
#plt.xticks()    #刻度不居中
plt.ylabel('Week1')

plt.subplot(4,1,2)
df = pd.DataFrame(data4)
yticks = [i for i in range(1,8)]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=[])
plt.ylabel('Week2')

plt.subplot(4,1,3)
df = pd.DataFrame(data5)
yticks = [i for i in range(1,8)]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks,xticklabels=[])
plt.ylabel('Week3')

plt.subplot(4,1,4)
df = pd.DataFrame(data6)
#xticks = [i for i in range(144)]
yticks = [i for i in range(1,8)]
sns.heatmap(df,cmap='rainbow',annot=False,fmt='.2f', annot_kws={'size': 0},yticklabels=yticks)
plt.xlabel('Time/10min')
plt.ylabel('Week4')
#plt.savefig('heatmap_min_week.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
print('end')
plt.show()






