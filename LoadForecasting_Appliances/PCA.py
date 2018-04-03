# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:05:19 2017

@author: ljc
"""

import pandas as pd
from sklearn.decomposition import PCA

DataSet = pd.read_csv('AppliancesLoadForecasting.csv')
datafeature = DataSet.iloc[:,2:]


pca = PCA()
pca.fit(datafeature)
print(pca.components_)
print(pca.explained_variance_ratio_)
