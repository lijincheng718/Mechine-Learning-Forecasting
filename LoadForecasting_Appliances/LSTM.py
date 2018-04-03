# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:53:29 2017

@author: ljc
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import GRU
import pandas as pd
import numpy as np 
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import warnings
warnings.filterwarnings('ignore')

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
#LSTM搭建模型
model = Sequential()
model.add(Embedding(3, output_dim=256))
model.add(GRU(128))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mse',optimizer='adam')
loss = model.fit(x_train, x_test, batch_size=128, epochs=10)
t2 = time.time()    #结束时间
print('time:',t2-t1)

#训练损失函数图像
plt.figure(figsize=(10,5))
plt.plot(loss.history['loss'], label='train')
#plt.plot(loss.history['val_loss'], label='test')
plt.legend(loc='upper right')

#预测数据
train_lstm = model.predict(x_train)
train_lstm = min_max_scaler_output.inverse_transform(train_lstm.reshape(-1,1))
train_output = min_max_scaler_output.inverse_transform(x_test.reshape(-1,1))
test_lstm = model.predict(y_train)
test_lstm = min_max_scaler_output.inverse_transform(test_lstm.reshape(-1,1))
test_output = min_max_scaler_output.inverse_transform(y_test.reshape(-1,1))
print('MAPE_train:', sum(abs(train_output-train_lstm)/train_output)/len(train_output))
print('MAPE_test:', sum(abs(test_output-test_lstm)/test_output)/len(test_output))

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
#plt.savefig('DNN预测.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
plt.show()



