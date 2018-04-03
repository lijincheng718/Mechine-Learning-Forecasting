# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:53:29 2017

@author: ljc
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import pandas as pd
import numpy as np 
import time
from sklearn import preprocessing
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1337)  # for reproducibility
data_dim = 1
#timesteps = 12
#dataset = pd.read_csv('timestep_12.csv')
timesteps = 3
dataset = pd.read_csv('timestep_3.csv')


min_max_scaler_input = preprocessing.MinMaxScaler()     #输入标准化函数
min_max_scaler_output = preprocessing.MinMaxScaler()    #输出标准化函数


data_input = dataset.iloc[:,:-1].values     #输入数据
data_output = dataset.iloc[:,-1:].values    #输出数据

trainlen = int(len(data_input)*0.8)     #输入样本数
testlen = int(len(data_input)-trainlen)     #测试样本数

train_output = data_output[:trainlen]    #训练输出数据
test_output = data_output[trainlen:]    #测试输出数据

data_input = min_max_scaler_input.fit_transform(data_input)     #输入标准化
data_output = min_max_scaler_output.fit_transform(data_output)      #输出标准化

x_train = data_input[:trainlen].reshape(trainlen,timesteps,data_dim)   #训练输入
x_test = data_output[:trainlen].reshape(trainlen,1)    #训练输出
y_train = data_input[trainlen:].reshape(testlen,timesteps,data_dim)   #测试输入
y_test = data_output[trainlen:].reshape(testlen,1)    #测试输出

t1 = time.time()

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
model.add(LSTM(64, return_sequences=True))  # 返回维度为 32 的向量序列
model.add(LSTM(64))  # 返回维度为 32 的单个向量
model.add(Dense(1, activation='tanh'))
model.compile(loss='mse', optimizer='RMSprop')

#loss = model.fit(x_train, x_test, epochs=10, batch_size=64,validation_data=(y_train, y_test), verbose=2, shuffle=False)
loss = model.fit(x_train, x_test,epochs=10,batch_size=64)
t2 = time.time()
print('train time:',t2-t1)

plt.figure(figsize=(10,5))
plt.plot(loss.history['loss'], label='train')
#plt.plot(loss.history['val_loss'], label='test')
plt.legend(loc='upper right')


train_lstm = model.predict(x_train)
train_lstm = min_max_scaler_output.inverse_transform(train_lstm.reshape(-1,1))
train_output = min_max_scaler_output.inverse_transform(x_test.reshape(-1,1))
test_lstm = model.predict(y_train)
test_lstm = min_max_scaler_output.inverse_transform(test_lstm.reshape(-1,1))
test_output = min_max_scaler_output.inverse_transform(y_test.reshape(-1,1))
print('MAPE_train:', sum(abs(train_output-train_lstm)/train_output)/len(train_output))
print('MAPE_test:', sum(abs(test_output-test_lstm)/test_output)/len(test_output))
print('RMSE_train:', sqrt(mean_squared_error(train_output, train_lstm)))
print('RMSE_test:', sqrt(mean_squared_error(test_output, test_lstm)))

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_lstm)
plt.xlabel('Number')
plt.ylabel('Wh')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_lstm)
plt.xlabel('Number')
plt.ylabel('Wh')
plt.legend(['real test value','predict test value'],loc='upper right')
#plt.savefig('lstm_stack.png',dpi=200,bbox_inches='tight')
plt.tight_layout()

plt.show()
'''
model = Sequential()
model.add(Embedding(1, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
'''


'''
trainlen = int(len(data_input)*0.9)
train time: 428.4639513492584
MAPE_train: [0.29651168]
MAPE_test: [0.26554261]
'''

'''
trainlen = int(len(data_input)*0.8)
train time: 82.67277979850769
MAPE_train: [0.38506943]
MAPE_test: [0.32719795]
'''