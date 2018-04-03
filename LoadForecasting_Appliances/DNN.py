# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:12:33 2017

@author: ljc
"""

from math import sqrt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1337)  # for reproducibility
dataset = pd.read_csv('data_input.csv')      #读取数据

min_max_scaler_input = preprocessing.MinMaxScaler()     #输入标准化函数
min_max_scaler_output = preprocessing.MinMaxScaler()    #输出标准化函数


#data_input = dataset.iloc[:,2:].values     #输入数据
data_input = dataset.iloc[:,-3:].values     #无环境变量输入数据
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
#DNN搭建模型
model = Sequential()
model.add(Dense(32, input_dim=data_input.shape[1], activation='tanh'))
#model.add(Dense(64, activation='tanh'))
#model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
#model.compile(loss='mse',optimizer='sgd')
model.compile(loss='mse',optimizer='adam')
#model.compile(loss='mse',optimizer='RMSprop')
loss = model.fit(x_train, x_test, epochs=20, batch_size=64,validation_data=(y_train, y_test), verbose=2, shuffle=False)
#loss = model.fit(x_train, x_test,epochs=10,batch_size=64)
t2 = time.time()    #结束时间
print('time:',t2-t1)
#训练损失函数图像
plt.figure(figsize=(10,5))
plt.plot(loss.history['loss'], label='Multistep')
plt.plot(loss.history['val_loss'], label='Single step')
plt.legend(loc='upper right')

#评估模型
#loss, accuracy = model.evaluate(x_test, y_test)
#score = model.evaluate(y_train, y_test, batch_size=64)
#print(score)
train_dnn = model.predict(x_train)
train_dnn = min_max_scaler_output.inverse_transform(train_dnn.reshape(-1,1))
train_output = min_max_scaler_output.inverse_transform(x_test.reshape(-1,1))
test_dnn = model.predict(y_train)
test_dnn = min_max_scaler_output.inverse_transform(test_dnn.reshape(-1,1))
test_output = min_max_scaler_output.inverse_transform(y_test.reshape(-1,1))
print('MAPE_train:', sum(abs(train_output-train_dnn)/train_output)/len(train_output))
print('MAPE_test:', sum(abs(test_output-test_dnn)/test_output)/len(test_output))
print('RMSE_train:', sqrt(mean_squared_error(train_output, train_dnn)))
print('RMSE_test:', sqrt(mean_squared_error(test_output, test_dnn)))

#保存数据
tp = pd.DataFrame(test_dnn)
to = pd.DataFrame(test_output)
dnn_data = pd.concat([to,tp],axis=1)
dnn_data.to_csv('ann_data.csv',header=['to','tp'],index=False)

#绘图
plt.figure(figsize=(6,6))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_dnn)
plt.xlabel('Number')
plt.ylabel('Wh')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_dnn)
plt.xlabel('Number')
plt.ylabel('Wh')
plt.legend(['real test value','predict test value'],loc='upper right')
#plt.savefig('DNN预测.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
plt.show()

'''

time: 64.79880738258362
MAPE_train: [0.2512379]
MAPE_test: [0.21192322]
RMSE_train: 67.81315395112006
RMSE_test: 60.389993696317774
'''