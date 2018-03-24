import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15


DataSet = pd.read_csv('data.csv')
dataset = DataSet.iloc[:,:]
plt.figure(figsize=(10,8))
corrmat = dataset.corr()    #计算相关系数矩阵
k = 9 #number of variables for heatmap
cols = corrmat.nlargest(k, 'sjczl')['sjczl'].index  
cm = np.corrcoef(dataset[cols].values.T)
#sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15},cmap='rainbow', yticklabels=cols.values, xticklabels=cols.values)
plt.savefig('相关性.png',dpi=200,bbox_inches='tight')
#plt.savefig('图时刻相关性.svg',bbox_inches='tight')
plt.title('Correlation matrix')
plt.tight_layout()
plt.show()