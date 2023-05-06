#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:20:28 2023

@author: damonchang
"""

import pandas as pd
import numpy as np
#%%
#path = r'/Users/damonchang/Desktop/研究生/课题/SVM/数据集/heart/heart.txt'

def libsvm_to_metrix(path):
    f = open(path, encoding='utf-8')
    #txt = pd.read_csv(path,header=None,sep=' ')
    X = []
    Y = []

    lines = f.readlines()
    n_feature = max([int(i.split(' ')[-2].split(':')[0]) for i in lines])

    f = open(path, encoding='utf-8')
    line = f.readline()
        
    while line:
        line = line.split(' ')
        if line[0] == '+1' or line[0] == '1':
            Y.append(1)
        else:
            Y.append(-1)
            
        _X = []
        _dic_refer = dict(zip(list(map(int,np.linspace(1,n_feature,n_feature))),list(np.zeros(n_feature))))
        for key in _dic_refer.keys():
            _dic_refer[key] = 3.1415926
        for i in range(len(line) - 2):
            _dic_refer[int(line[i+1].split(':')[0])] = float(line[i+1].split(':')[1])
            #_X.append(float(line[i+1].split(':')[1]))
        _X = [value for value in _dic_refer.values()]
        X.append(np.array(_X))
        line = f.readline()
    X = np.array(X)
    
    '''
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j] == 3.1415926:
                X[i][j] = np.mean(np.array([x for x in X[:,j] if x != 3.1415926]))
    '''            
    for j in range(X.shape[1]):
        mean_j = np.mean(np.array([x for x in X[:,j] if x != 3.1415926]))
        X[:,j][X[:,j]==3.1415926] = mean_j

    #np.savez(path.replace('txt','npz'), X = np.array(X,dtype='float64'), Y = np.array(Y,dtype='int'))
    #d = np.load(path.replace('txt','npz'),allow_pickle=True)
    #xout,yout = np.array(X,dtype='float64'), np.array(Y,dtype='int')
    return np.array(X,dtype='float64'), np.array(Y,dtype='int')
#%%
def libsvm_to_metrix_multi_attribute(path):
    f = open(path, encoding='utf-8')
    #txt = pd.read_csv(path,header=None,sep=' ')
    X = []
    Y = []

    lines = f.readlines()
    n_feature = max([int(i.split(' ')[-2].split(':')[0]) for i in lines])

    f = open(path, encoding='utf-8')
    line = f.readline()
        
    while line:
        line = line.split(' ')
        Y.append(float(line[0]))
            
        _X = []
        _dic_refer = dict(zip(list(map(int,np.linspace(1,n_feature,n_feature))),list(np.zeros(n_feature))))
        for key in _dic_refer.keys():
            _dic_refer[key] = 3.1415926
        for i in range(len(line) - 2):
            _dic_refer[int(line[i+1].split(':')[0])] = float(line[i+1].split(':')[1])
            #_X.append(float(line[i+1].split(':')[1]))
        _X = [value for value in _dic_refer.values()]
        X.append(np.array(_X))
        line = f.readline()
    X = np.array(X)
    
    #将缺失值补齐为列的均值
    '''
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j] == 3.1415926:
                X[i][j] = np.mean(np.array([x for x in X[:,j] if x != 3.1415926]))
    '''
    for j in range(X.shape[1]):
        mean_j = np.mean(np.array([x for x in X[:,j] if x != 3.1415926]))
        X[:,j][X[:,j]==3.1415926] = mean_j


    #np.savez(path.replace('txt','npz'), X = np.array(X,dtype='float64'), Y = np.array(Y,dtype='int'))
    #d = np.load(path.replace('txt','npz'),allow_pickle=True)
    #xout,yout = np.array(X,dtype='float64'), np.array(Y,dtype='int')
    return np.array(X,dtype='float64'), np.array(Y,dtype='int')



