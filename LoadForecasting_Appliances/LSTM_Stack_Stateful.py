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
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1337)  # for reproducibility
data_dim = 1
#timesteps = 12
#dataset = pd.read_csv('timestep_12.csv')
timesteps = 6
batch_size = 32

dataset = pd.read_csv('timestep_6.csv')
data_len = int(len(dataset)/batch_size)*batch_size  #转成整数倍的batch_size的长度
data_input = dataset.iloc[:data_len,:-1].values     #输入数据
data_output = dataset.iloc[:data_len,-1:].values    #输出数据(19725, 1)
print(data_output.shape,data_input.shape)

trainlen = int(data_len/batch_size*0.8)*batch_size   #输入样本长度
testlen = int(data_len-trainlen)     #测试样本数
train_output = data_output[:trainlen]    #训练输出数据
test_output = data_output[trainlen:]    #测试输出数据

min_max_scaler_input = preprocessing.MinMaxScaler()     #输入标准化函数
min_max_scaler_output = preprocessing.MinMaxScaler()    #输出标准化函数
data_input = min_max_scaler_input.fit_transform(data_input)     #输入标准化
data_output = min_max_scaler_output.fit_transform(data_output)      #输出标准化

x_train = data_input[:trainlen].reshape(trainlen,timesteps,data_dim)   #训练输入
x_test = data_output[:trainlen].reshape(trainlen,1)    #训练输出
y_train = data_input[trainlen:].reshape(testlen,timesteps,data_dim)   #测试输入
y_test = data_output[trainlen:].reshape(testlen,1)    #测试输出

t1 = time.time()

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
# 请注意，我们必须提供完整的 batch_input_shape，因为网络是有状态的。
# 第 k 批数据的第 i 个样本是第 k-1 批数据的第 i 个样本的后续。
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(64, return_sequences=True, stateful=True))
model.add(LSTM(64, stateful=True))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mse', optimizer='rmsprop')

#loss = model.fit(x_train, x_test, epochs=10, batch_size=batch_size, shuffle=False, validation_data=(y_train, y_test))
loss = model.fit(x_train, x_test,epochs=10,batch_size=batch_size)
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

#保存数据
tp = pd.DataFrame(test_lstm)
to = pd.DataFrame(test_output)
lstm_data = pd.concat([to,tp],axis=1)
lstm_data.to_csv('lstm_data.csv',header=['to','tp'],index=False)

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
#plt.savefig('lstm_stack_state.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
plt.show()


'''
train time: 86.75051593780518
MAPE_train: [0.26779229]
MAPE_test: [0.22412197]
'''
