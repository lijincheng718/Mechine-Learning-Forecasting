# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:10:10 2018

@author: ljc
"""

from PyEMD import EMD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16           
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据
dataset = pd.read_csv('data.csv',header=None)
data = dataset.iloc[:,-1:].values
s = data.ravel()
print(s.shape)
#s = (s-s.min())/(s.max()-s.min())

# Execute EMD on signal
IMF = EMD().emd(s)
N = IMF.shape[0]+1

# Plot results
plt.figure(figsize=(10,20))
plt.subplot(N,1,1)
plt.plot(s, 'r')
plt.xlabel("Number")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(imf, 'g')
    plt.legend(["IMF "+str(n+1)],loc='upper right')
    plt.xlabel("Number")

plt.tight_layout()
plt.savefig('emd.png')
plt.show()

'''
plt.figure(figsize=(12,8))
t = np.linspace(0, 1, 200)
sin = lambda x,p: np.sin(2*np.pi*x*t+p)
s = 3*sin(18,0.2)*(t-0.2)**2
s += 5*sin(11,2.7)
s += 3*sin(14,1.6)
s += 1*np.sin(4*2*np.pi*(t-0.8)**2)
s += t**2.1 -t
# Execute EMD on signal
IMF = EMD().emd(s,t)
N = IMF.shape[0]+1

# Plot results
plt.subplot(N,1,1)
plt.plot(t, s, 'r')
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.subplot(N,1,n+2)
    plt.plot(t, imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

plt.tight_layout()
#plt.savefig('simple_example')
plt.show()
'''


#---------------------------------------------------------
'''
#Simplest case of using Esnembld EMD (EEMD) is by importing EEMD and passing your signal to eemd method.
from PyEMD import EEMD

# Define signal
t = np.linspace(0, 1, 200)
sin = lambda x,p: np.sin(2*np.pi*x*t+p)
S = 3*sin(18,0.2)*(t-0.2)**2
S += 5*sin(11,2.7)
S += 3*sin(14,1.6)
S += 1*np.sin(4*2*np.pi*(t-0.8)**2)
S += t**2.1 -t


if __name__ == '__main__':
# Assign EEMD to `eemd` variable
    eemd = EEMD()
    eIMFs = eemd.eemd(S, t)
    emd = eemd.EMD
    emd.extrema_detection="parabol"
    
    # Execute EEMD on S
    eIMFs = eemd.eemd(S, t)
    nIMFs = eIMFs.shape[0]
    
    # Plot results
    plt.figure(figsize=(12,9))
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(t, S, 'r')
    
    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)
    
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()
'''

