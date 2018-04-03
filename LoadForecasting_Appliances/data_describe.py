# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:37:56 2017

@author: ljc
"""

import pandas as pd

dataset = pd.read_csv('data_input.csv')
data = dataset.t
print(data.describe())
print(data.value_counts())
value_count = data.value_counts()
value_count = pd.DataFrame(value_count)
value_count.to_csv('value_count.csv')