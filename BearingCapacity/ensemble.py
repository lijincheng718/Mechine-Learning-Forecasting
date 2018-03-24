# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:47:53 2017

@author: ljc
"""


import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('data.csv')       #读取数据
data_input = dataset.iloc[:,:-1].values     #输入数据
data_output = dataset.iloc[:,-1:].values    #输出数据
x_train, y_train,x_test, y_test = train_test_split(data_input, data_output, test_size=0.1, random_state=0)  #划分训练集和测试集
train_output = x_test
test_output = y_test

#决策树模型
t1 = time.time()    #起始时间
model_dtr = tree.DecisionTreeRegressor()    
model_dtr.fit(x_train,x_test)
t2 = time.time()    #结束时间
print('time:',t2-t1)

train_dtr=model_dtr.predict(x_train)
test_dtr=model_dtr.predict(y_train)
train_dtr = train_dtr.reshape(-1,1)
test_dtr = test_dtr.reshape(-1,1)
print('MAPE_train_dtr:', sum(abs(train_output-train_dtr)/train_output)/len(train_output))
print('MAPE_test_dtr:', (sum(abs(test_output-test_dtr)/test_output))/len(test_output))

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_dtr)
plt.xlabel('Number')
plt.ylabel('Values')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_dtr)
plt.xlabel('Number')
plt.ylabel('Values')
plt.legend(['real test value','predict test value'],loc='upper right')
plt.savefig('DT预测.png',dpi=200,bbox_inches='tight')
plt.tight_layout()

'''

#集成模型
t1 = time.time()    #起始时间
#model_e = ensemble.AdaBoostRegressor(n_estimators=10)
#model_e = ensemble.GradientBoostingRegressor(n_estimators=10)
#model_e = ensemble.BaggingRegressor(n_estimators=300)
model_e = ensemble.RandomForestRegressor(n_estimators=500)  #这里使用100个决策树
model_e.fit(x_train,x_test)
t2 = time.time()    #结束时间
print('time:',t2-t1)

train_e = model_e.predict(x_train)
test_e = model_e.predict(y_train)
train_e = train_e.reshape(-1,1)
test_e = test_e.reshape(-1,1) 
print('MAPE_train_e:', sum(abs(train_output-train_e)/train_output)/len(train_output))
print('MAPE_test_e:', (sum(abs(test_output-test_e)/test_output))/len(test_output))

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_e)
plt.xlabel('Number')
plt.ylabel('Values')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_e)
plt.xlabel('Number')
plt.ylabel('Values')
plt.legend(['real test value','predict test value'],loc='upper right')
plt.savefig('RandomForest.png',dpi=200,bbox_inches='tight')
plt.tight_layout()



#K近邻模型
t1 = time.time()    #起始时间
model_knr = neighbors.KNeighborsRegressor()
model_knr.fit(x_train,x_test)
t2 = time.time()    #结束时间
print('time:',t2-t1)

train_knr = model_knr.predict(x_train)
test_knr = model_knr.predict(y_train)
train_knr = train_knr.reshape(-1,1)
test_knr = test_knr.reshape(-1,1)
print('MAPE_train_knr:', sum(abs(train_output-train_knr)/train_output)/len(train_output))
print('MAPE_test_knr:', (sum(abs(test_output-test_knr)/test_output))/len(test_output))

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_knr)
plt.xlabel('Number')
plt.ylabel('Values')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_knr)
plt.xlabel('Number')
plt.ylabel('Values')
plt.legend(['real test value','predict test value'],loc='upper right')
plt.savefig('kNN.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
'''
plt.show()
