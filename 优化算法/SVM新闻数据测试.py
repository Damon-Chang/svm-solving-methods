#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 13:32:39 2023

@author: damonchang
"""
import numpy as np
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/优化算法')
import vector2array
from sklearn import preprocessing


min_max_scaler = preprocessing.MinMaxScaler()
#x_minmax = min_max_scaler.fit_transform(x)

#%%
path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/SVM新闻数据/vector_js.txt'
path2 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/SVM新闻数据/vector_njs.txt'
x_js,y_js = vector2array.vector2array(path1)
x_njs,y_njs = vector2array.vector2array(path2)
X_all = np.append(x_js,x_njs,axis=0)
Y_all = np.append(y_js,y_njs,axis=0)
#%%
#数据标准化
X_all -= np.mean(X_all,axis=0)
X_all = np.std(X_all,axis=0)
#%%
Y_all[Y_all==0] = -1
X = X_all[-2*np.round(len(y_njs)):]
Y = Y_all[-2*np.round(len(y_njs)):]
cut = np.round(len(Y)*0.8)
#x,y,x_t,y_t = X[:cut],Y[:cut],X[cut:],Y[cut:]
#X = np.random.shuffle(X) #打乱顺序


state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(Y)

def X_1(X):
    one = np.ones(X.shape[0])
    X_ = np.column_stack((X,one))
    return X_
#%%
#数据标准化

X_m =X - np.mean(X,axis=0)
X_s = X_m/np.std(X_m,axis=0)

#%%

X = X_1(X)
#%%
X_m =X - np.mean(X,axis=0)
X_s = X_m/np.std(X_m,axis=0)

import BB_SSGD_SVM
model = BB_SSGD_SVM.SSGD(use_BB_step=True)
model.fit(X_s, Y)
#print('test accuracy:', model.get_accuracy(x_t, y_t))
#%%
import TR
model_tr = TR.TR()
model_tr.fit(X_s,Y)

#%%
import O_RDRSVMbyADMM
model_admm = O_RDRSVMbyADMM.RDSVMbyADMM()
model_admm.fit(X, Y)

#%%
import TRLMF
model_lmf = TRLMF.TRLMF()
model_lmf.fit(X,Y)
#%%
import DC_SVM
model_dc = DC_SVM.DC()
#model_dc.fit(X,Y)
model_dc.fit(X_all,Y_all)