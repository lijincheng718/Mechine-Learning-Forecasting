# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 22:25:26 2017

@author: ljc
"""


import numpy as np
import pandas as pd
from math import sqrt
import time
from sklearn.svm import SVR
#from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
#from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('data_input.csv')      #读取数据

min_max_scaler_input = preprocessing.MinMaxScaler()     #输入标准化函数
min_max_scaler_output = preprocessing.MinMaxScaler()    #输出标准化函数


data_input = dataset.iloc[:,2:13].values     #输入数据
data_output = dataset.iloc[:,1:2].values    #输出数据

trainlen = int(len(data_input)*0.8)     #输入样本数
testlen = int(len(data_input)-trainlen)     #测试样本数

train_output = data_output[:trainlen]    #训练输出数据
test_output = data_output[trainlen:]    #测试输出数据

data_input = min_max_scaler_input.fit_transform(data_input)     #输入标准化
data_output = min_max_scaler_output.fit_transform(data_output)      #输出标准化

x_train = data_input[:trainlen].reshape(trainlen,-1)   #训练输入
x_test = data_output[:trainlen].reshape(trainlen,1)    #训练输出
y_train = data_input[trainlen:].reshape(testlen,-1)   #测试输入
y_test = data_output[trainlen:].reshape(testlen,1)    #测试输出


t1 = time.time()    #起始时间
model_svr = SVR(kernel='rbf',C=1000,gamma=0.1)       #svr_rbf模型,C越大，时间越长
#model_svr = SVR(kernel='linear',C=100)       #svr_linear模型
model_svr.fit(x_train,x_test)       #训练拟合数据
t2 = time.time()    #结束时间
print('time:',t2-t1)

train_svr = model_svr.predict(x_train)      #预测数据
train_svr = min_max_scaler_output.inverse_transform(train_svr.reshape(-1,1))
train_output = min_max_scaler_output.inverse_transform(x_test.reshape(-1,1))
test_svr = model_svr.predict(y_train)
test_svr = min_max_scaler_output.inverse_transform(test_svr.reshape(-1,1))
test_output = min_max_scaler_output.inverse_transform(y_test.reshape(-1,1))

print('MAPE_train:', sum(abs(train_output-train_svr)/train_output)/len(train_output))
print('MAPE_test:', sum(abs(test_output-test_svr)/test_output)/len(test_output))
print('RMSE_train:', sqrt(mean_squared_error(train_output, train_svr)))
print('RMSE_test:', sqrt(mean_squared_error(test_output, test_svr)))

'''
plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_svr)
plt.xlabel('Number')
plt.ylabel('Wh')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_svr)
plt.xlabel('Number')
plt.ylabel('Wh')
plt.legend(['real test value','predict test value'],loc='upper right')
#plt.savefig('SVR预测2.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
plt.show()
'''