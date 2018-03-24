import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'Times New Roman'
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('data.csv')       #读取数据

min_max_scaler_input = preprocessing.MinMaxScaler()     #输入标准化函数
min_max_scaler_output = preprocessing.MinMaxScaler()    #输出标准化函数

'''
data_input = dataset.iloc[:,:-1].values     #输入数据
data_output = dataset.iloc[:,-1:].values    #输出数据
train_output = data_output[:210]    #训练输出数据
test_output = data_output[210:]    #测试输出数据
data_input = min_max_scaler_input.fit_transform(data_input)     #输入标准化
data_output = min_max_scaler_output.fit_transform(data_output)      #输出标准化
x_train = data_input[:210].reshape(210,9)   #训练输入
x_test = data_output[:210].reshape(210,1)    #训练输出
y_train = data_input[210:].reshape(17,9)   #测试输入
y_test = data_output[210:].reshape(17,1)    #测试输出
'''

data_input = dataset.iloc[:,:-1].values     #输入数据
data_output = dataset.iloc[:,-1:].values    #输出数据
data_input = min_max_scaler_input.fit_transform(data_input)     #输入标准化
data_output = min_max_scaler_output.fit_transform(data_output)      #输出标准化
x_train, y_train,x_test, y_test = train_test_split(data_input, data_output, test_size=0.1, random_state=0)

t1 = time.time()    #起始时间
model = Sequential()
model.add(Dense(32, input_dim=9, activation='tanh'))
#model.add(Dropout(0.8))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(64, activation='tanh'))
#model.add(Dropout(0.8))
model.add(Dense(1, activation='linear'))
#model.compile(loss='mse',optimizer='sgd',metrics=['accuracy'])
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
#model.compile(loss='mse',optimizer='RMSprop',metrics=['accuracy'])
model.fit(x_train, x_test,epochs=3000,batch_size=105)
t2 = time.time()    #结束时间
print('time:',t2-t1)

score = model.evaluate(y_train, y_test, batch_size=17)
train_ann = model.predict(x_train)
train_ann = min_max_scaler_output.inverse_transform(train_ann.reshape(-1,1))
train_output = min_max_scaler_output.inverse_transform(x_test.reshape(-1,1))
test_ann = model.predict(y_train)
test_ann = min_max_scaler_output.inverse_transform(test_ann.reshape(-1,1))
test_output = min_max_scaler_output.inverse_transform(y_test.reshape(-1,1))
print('MAPE_train:', sum(abs(train_output-train_ann)/train_output)/len(train_output))
print('MAPE_test:', (sum(abs(test_output-test_ann)/test_output))/len(test_output))
#print(score)

plt.figure(figsize=(10,10))
plt.subplot(211)
plt.plot(train_output)
plt.plot(train_ann)
plt.xlabel('Number')
plt.ylabel('Value')
plt.legend(['real train value','predict train value'],loc='upper right')
plt.subplot(212)
plt.plot(test_output)
plt.plot(test_ann)
plt.xlabel('Number')
plt.ylabel('Value')
plt.legend(['real test value','predict test value'],loc='upper right')
plt.savefig('DNN预测.png',dpi=200,bbox_inches='tight')
plt.tight_layout()
plt.show()