# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:17:14 2017

@author: ljc
"""

import pandas as pd
dataset = pd.read_csv('data_input.csv')
minload = dataset.t
#minload = minload[42:]
df = pd.DataFrame()
for i in range(14,0,-1):
    df['t-'+str(i)] = minload.shift(i)
df['t'] = minload.values
df = df[15:]
df.to_csv('min_index.csv',index=False)


'''
import pandas as pd
dataset = pd.read_csv('hourload.csv')
hourload = dataset.hourload
hourload = hourload[::24]
df = pd.DataFrame()
for i in range(14,0,-1):
    df['t-'+str(i)] = hourload.shift(i)
df['t'] = hourload.values
df = df[15:]
df.to_csv('hour_index.csv',index=False)
'''


'''
import pandas as pd
dataset = pd.read_csv('dayload.csv')
dayload = dataset.dayload
dayload = dayload[::24]
df = pd.DataFrame()
for i in range(14,0,-1):
    df['t-'+str(i)] = dayload.shift(i)
df['t'] = dayload.values
df = df[15:]
df.to_csv('dayload_index.csv',index=False)
'''







