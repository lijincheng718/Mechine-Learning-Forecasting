# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:56:16 2017

@author: ljc
"""

import pandas as pd
import numpy as np
dataset = pd.read_csv('data_input.csv')
load = dataset.t.values[:-4]    #选取整除6的样本数

d = 144
h = 6
w = 7

load_week1 = load[:-d*w]     #一周前T
load_day1_t1 = load[d*(w-1)-h:-d-h]    #一天前T的前一小时
load_day1 = load[d*(w-1):-d]       #一天前T
load_current_t1 = load[d*w-h:-h]   #当前T的前一个小时
load_current = load[d*w:]   #当前T

load_week1 = load_week1.reshape(-1,h)
load_day1_t1 = load_day1_t1.reshape(-1,h)
load_day1 = load_day1.reshape(-1,h)
load_current_t1 = load_current_t1.reshape(-1,h)
load_output = load_current.reshape(-1,h)

load_week1 = pd.DataFrame(load_week1)
load_day1_t1 = pd.DataFrame(load_day1_t1)
load_day1 = pd.DataFrame(load_day1)
load_current_t1 = pd.DataFrame(load_current_t1)
load_output = pd.DataFrame(load_output)

load_multistep_feature = pd.concat([load_week1,load_day1_t1,load_day1,load_current_t1,load_output],axis=1)
load_multistep_feature.to_csv('multistep_feature.csv',header=None,index=False)
