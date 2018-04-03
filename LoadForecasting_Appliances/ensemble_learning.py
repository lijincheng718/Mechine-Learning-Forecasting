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
from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn import preprocessing
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1337)  # for reproducibility

dataset = pd.read_csv('data_input.csv')      #读取数据
data_input = dataset.iloc[:,2:13].values     #输入数据
data_output = dataset.iloc[:,1:2].values    #输出目标数据

trainlen = int(len(data_input)*0.8)     #训练样本数量
testlen = int(len(data_input)-trainlen)     #测试样本数量

train_output = data_output[:trainlen]    #训练输出数据
test_output = data_output[trainlen:]    #测试输出数据

x_train = data_input[:trainlen].reshape(trainlen,-1)   #训练输入
x_test = data_output[:trainlen].reshape(trainlen,1)    #训练输出
y_train = data_input[trainlen:].reshape(testlen,-1)   #测试输入
y_test = data_output[trainlen:].reshape(testlen,1)    #测试输出


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
print('RMSE_train_dtr:', sqrt(mean_squared_error(train_output, train_dtr)))
print('RMSE_test_dtr:', sqrt(mean_squared_error(test_output, test_dtr)))
'''
plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_dtr)
plt.xlabel('Wh')
plt.ylabel('Values')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_dtr)
plt.xlabel('Wh')
plt.ylabel('Values')
plt.legend(['real test value','predict test value'],loc='upper right')
#plt.savefig('DT预测.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
'''


#集成学习模型
t1 = time.time()    #起始时间
#model_e = ensemble.AdaBoostRegressor(n_estimators=10)
model_e = ensemble.GradientBoostingRegressor(n_estimators=50)
#model_e = ensemble.BaggingRegressor(n_estimators=300)
#model_e = ensemble.RandomForestRegressor(n_estimators=50)  #这里使用100个决策树
model_e.fit(x_train,x_test)
t2 = time.time()    #结束时间
print('time:',t2-t1)

train_e = model_e.predict(x_train)
test_e = model_e.predict(y_train)
train_e = train_e.reshape(-1,1)
test_e = test_e.reshape(-1,1) 
print('MAPE_train_e:', sum(abs(train_output-train_e)/train_output)/len(train_output))
print('MAPE_test_e:', (sum(abs(test_output-test_e)/test_output))/len(test_output))
print('RMSE_train_e:', sqrt(mean_squared_error(train_output, train_e)))
print('RMSE_test_e:', sqrt(mean_squared_error(test_output, test_e)))

#保存数据
tp = pd.DataFrame(test_e)
to = pd.DataFrame(test_output)
e_data = pd.concat([to,tp],axis=1)
e_data.to_csv('e_data.csv',header=['to','tp'],index=False)

'''
plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_e)
plt.xlabel('Wh')
plt.ylabel('Values')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_e)
plt.xlabel('Wh')
plt.ylabel('Values')
plt.legend(['real test value','predict test value'],loc='upper right')
#plt.savefig('RandomForest.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
'''


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
print('RMSE_train_knr:', sqrt(mean_squared_error(train_output, train_knr)))
print('RMSE_test_knr:', sqrt(mean_squared_error(test_output, test_knr)))
'''
plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_knr)
plt.xlabel('Wh')
plt.ylabel('Values')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_knr)
plt.xlabel('Wh')
plt.ylabel('Values')
plt.legend(['real test value','predict test value'],loc='upper right')
#plt.savefig('kNN.png',dpi=200,bbox_inches='tight')
plt.tight_layout()

plt.show()
'''


'''
time: 0.22865915298461914
MAPE_train_dtr: [0.]
MAPE_test_dtr: [0.36293949]
RMSE_train_dtr: 0.0
RMSE_test_dtr: 93.10146331899296
time: 8.044211626052856
MAPE_train_e: [0.11202995]
MAPE_test_e: [0.31033356]
RMSE_train_e: 24.89765721579048
RMSE_test_e: 65.19020118310588
time: 0.058539628982543945
MAPE_train_knr: [0.24038264]
MAPE_test_knr: [0.26225409]
RMSE_train_knr: 55.80290884584036
RMSE_test_knr: 65.56525323378948
'''