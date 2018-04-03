# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:08:33 2017

@author: ljc
"""

import pandas as pd

DataSet = pd.read_csv('AppliancesLoadForecasting.csv')
dataset = DataSet.loc[:,['date','Appliances','lights','T2','Windspeed','RH1','T3','RH6','RH8','RHO']]
dataset.to_csv('data_selection.csv',index=False)


