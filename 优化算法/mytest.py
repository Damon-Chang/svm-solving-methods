#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 08:26:28 2023

@author: damonchang
"""

import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
import numpy as np
import TRLMF
#%%

def X_1(X):
    one = np.ones(X.shape[0])
    X_ = np.column_stack((X,one))
    return X_

if __name__ == '__main__':

    print('-------choose dataset--------\n')
    print('---    1.heart(240,13)    ---\n')
    print('---    2.ijcnn1(49990,21) ---\n')
    print('-------choose dataset--------\n')
    
    path_choose = eval(input('your choise: '))
    if path_choose == 1:
        path = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/heart/heart.txt'
    elif path_choose == 2:
        path = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/ijcnn/ijcnn1train.txt'
        #'''
    x,y = libsvm_to_metrix.libsvm_to_metrix(path)
    x = X_1(x)
    X_train,y_train,X_test,y_test = x[:200],y[:200],x[200:],y[200:]

    model = TRLMF.TRLMF()
    model.fit(X_train,y_train)
    print('test accuracy:',model.get_accuracy(X_test,y_test))
    
    