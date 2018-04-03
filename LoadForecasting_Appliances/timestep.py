# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 17:53:29 2017

@author: ljc
"""


import pandas as pd

dataset = pd.read_csv('data_input.csv')
data = dataset.t
step = 3
df = pd.DataFrame()
for i in range(step,0,-1):
    df['t-'+str(i)] = data.shift(i)
df['t'] = data.values
df = df[step+1:]
df.to_csv('timestep_3.csv',index=False)