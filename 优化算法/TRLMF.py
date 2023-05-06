#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 20:37:50 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
import libsvm2array
import time
import pandas as pd

#%%
class TRLMF(object):
    def __init__(self,gtol=1e-5,eta1=0.25,eta2=0.75,gamma1=0.5,gamma2=2,iteration=64):
        self.gtol = gtol
        self.eta_1 = eta1
        self.eta_2 = eta2
        self.gamma_1 = gamma1
        self.gamma_2 = gamma2
        self.n_iter = iteration
        self.C = 1.
        self.eta = 0.1
        
    
    
    def fit(self,X,Y):
        self.n_class = len(list(set(Y)))
        self.n_samples = X.shape[0] #n
        self.n_features = X.shape[1] #p+1
        self._lambda = 100*np.sqrt(self.n_features)
        self.w = np.zeros(self.n_features) + 0.01
        I = np.eye(self.n_features)
        
        self.f_s = [np.linalg.norm(self.residual_fun(X, Y, self.w),2)+1]
        self.f_s.append(np.linalg.norm(self.residual_fun(X, Y, self.w),2))
        self.w_s = [self.w]
        k = 0
        t1 = time.time()
        while self.f_s[-1] >= self.gtol and k <= self.n_iter and \
            abs(self.f_s[-1] - self.f_s[-2]) >= self.gtol:
        #while self.f_s[-1] >= self.gtol and k <= self.n_iter:            
            J = self.jacobian(X,Y)
            H = J.T@J
            A = H + self._lambda*I
            #A = H + self._lambda*np.diag(H)
            f = self.residual_fun(X,Y,self.w)
            b = -J.T@f
            #QR分解求解子问题
            p = self.QR_solver(A,b)
            
            w_ = self.w + p
            #print(w_)
            self.w_s.append(w_)
            self.f_s.append(np.linalg.norm(self.residual_fun(X,Y,w_),2))
            
            
            #rho_k = abs((-b@w_ + w_@H@w_ + b@self.w - self.w@H@self.w + 0.001)\
            #            / ((self.residual_fun(X, Y,self.w) - self.residual_fun(X, Y, w_)+0.001)))
            #rho_k = abs((b@p + 0.5*p@A@p + 0.00001)\
            #            / (abs(self.f_s[-1] - self.f_s[-2]) +0.00001))
            rho_k = abs(0.5*(abs(np.linalg.norm(self.residual_fun(X,Y,self.w),2)**2 \
                                 - np.linalg.norm(self.residual_fun(X,Y,w_),2)**2)) \
                        / (b@p + 0.5*p@H@p + 0.00001))
            #print('rho_k',rho_k)
            #rho_k = abs((self.obj_loss(self.w) - self.obj_loss(w_)) / (g@d + 0.5 * d@B@d))
            #确定是否更新信赖域半径
            #print('比值：',rho_k)
            if abs(1 - abs(rho_k - 1)) >= self.eta and abs(1 - abs(rho_k - 1)) <= 1:
                print('rho_k={},此次更新值得信任。'.format(rho_k))
                #接受此次更新，并记录上一步的函数值和梯度
                #self.f_s.append(np.linalg.norm(f,2))
                self.w = w_
            #调整信赖域半径
            if abs(1 - abs(rho_k - 1)) <= self.eta_1 or abs(1 - abs(rho_k - 1)) > 1 or \
                np.isnan(rho_k):
                print('rho_k={},此次更新不值得信任，缩减信赖域半径重新更新.'.format(rho_k))
                self.w_s.pop()
                self.f_s.pop()
                self._lambda = self.gamma_2 * self._lambda
                
            if abs(1 - abs(rho_k - 1) >= self.eta_2 and abs(1 - abs(rho_k - 1)) <= 1):
                print('rho_k={},近似效果很好，增大信赖域半径.'.format(rho_k))
                self._lambda = self.gamma_1 * self._lambda
                #self.w_s.append(self.w)
                
            k += 1
            print('{}th iteration,loss = 「{}」'.format(k,self.f_s[-1]))
            
        t2 = time.time()
        t_all = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(t_all,k))
        print('训练准确率：',self.get_accuracy(X,Y))
        self.show_loss()
        return 'Done!'
      
    def get_accuracy(self,X,Y):
        return (sum(self.predict(X) == Y)) / len(Y)
        
    def predict(self,x):
        if self.n_class == 2:
            return np.where(x@self.w >= 0.0, 1, -1)
        else:
            pre = []
            for sample in list(x):
                dist = [abs(x@self.w - i) for i in list(set(self.Y))]
                pre.append(list(set(self.Y))[np.argmin(dist)])
            return pre
    
    def show_loss(self):
        plt.scatter(list(range(len(self.f_s))), self.f_s,s=5)
        plt.xlabel('Iteration')
        plt.ylabel(r'$ || t^k-t^{k-1}||_2 $')
        plt.title('LOSS')
        plt.show()
    

    def jacobian(self,X,Y):
        result = np.zeros((self.n_samples + 1,self.n_features))
        w_norm = np.linalg.norm(self.w)
        for j in range(self.n_features):
            result[0][j] = self.w[j] / w_norm
        for i in range(self.n_samples):
            for j in range(self.n_features):
                if 1 - self.C*Y[i]*X[i][j] >= 0:
                    result[i+1][j] = -self.C*Y[i]*X[i][j]
                else:
                    result[i+1][j] = 0.
        
        return result
    
    def residual_fun(self,X,Y,w):
        result = [np.linalg.norm(w,2)]
        for i in range(self.n_samples):
            result.append(self.C*max(0,1-Y[i]*w@X[i]))
        return np.array(result)
    
    def QR_solver(self,A,b):
        #solve Ax=b by QR分解
        
        Q = np.zeros_like(A) #Q.shape=A.shape
        w = 0
        for a in A.T:
            u = np.copy(a)
            for i in range(0, w):
                u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
            ex = u / np.linalg.norm(u)
            Q[:, w] = ex
            w += 1
        R = np.dot(Q.T, A) # A=Q*R
        N = np.dot(Q.T,b)
        x2 = np.linalg.solve(R,N)  # Rx=Q.T*b
        self.x2 = x2
        #print(x2)   #solve x
        
        print("done QR!")
        return x2
    
    def householder(self,A):
        (r, c) = np.shape(A)
        Q = np.identity(r)
        R = np.copy(A)
        for i in range(r - 1):
            x = R[i:,i]
            e = np.zeros_like(x)
            e[0] = np.linalg.norm(x)
            u = x - e
            v = u / np.linalg.norm(u)
            Q_i = np.identity(r)
            Q_i[i:, i:] -= 2.0 * np.outer(v,v)
            R = np.dot(Q_i, R)  
            Q = np.dot(Q, Q_i)  
        return (Q,R,c)
 


#%%
if __name__ == '__main__':
    #使用heart数据
    def X_1(X):
        one = np.ones(X.shape[0])
        X_ = np.column_stack((X,one))
        return X_
    
    print('-------choose dataset-----------\n')
    print('---    1.heart(240,13)       ---\n')
    print('---    2.ijcnn1(49990,21)    ---\n')
    print('---    3.svmguide(3089,4)    ---\n')
    print('---    4.w3a(4912,300)       ---\n')
    print('---    5.svmguide3(1243,21)  ---\n')
    print('---    6.covtype(581012,54)  ---\n')
    print('-------choose dataset-----------\n')
    
    path_choose = eval(input('your choise: '))
    if path_choose == 1:
        path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/heart/heart.txt'
        path2 = path1
    elif path_choose == 2:
        path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/ijcnn/ijcnn1train.txt'
        path2 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/ijcnn/ijcnn1test.txt'
    elif path_choose == 3:
        path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/svmguide1/svmguide1_train.txt'
        path2 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/svmguide1/svmguide1_test.txt'
    elif path_choose == 4:
        path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/w3a/w3a_train.txt'
        path2 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/w3a/w3a_test.txt'
    elif path_choose == 5:
        path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/svmguide3/svmguide3.txt'
        path2 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/svmguide3/svmguide3_test.txt'
    elif path_choose == 6:
        path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/covtype/covtype.txt'
        path2 = path1

    #'''
    #path = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/heart/heart.txt'
    #x,y = libsvm_to_metrix.libsvm_to_metrix(path1)
    print('----  data processing  ----')
    x,y = libsvm2array.svm_read_problem(path1)
    x = X_1(x)
    #x_t,y_t = libsvm_to_metrix.libsvm_to_metrix(path2)
    x_t,y_t = libsvm2array.svm_read_problem(path2)
    x_t = X_1(x_t)
    print('----  done  ----')
    
    X_train,y_train = x,y
    X_test,y_test = x_t,y_t 
    
    model = TRLMF()
    model.fit(X_train,y_train)
    print('test accuracy:',model.get_accuracy(X_test,y_test))
    #'''
   
    '''
    #path = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/ijcnn/ijcnn1train.txt'
    x,y=libsvm_to_metrix.libsvm_to_metrix(path)
    x = X_1(x)
    cut = round(0.8*x.shape[0])
    X_train,y_train,X_test,y_test = x[:cut],y[:cut],x[cut:],y[cut:]
    
    model = TRLMF()
    model.fit(X_train,y_train)
    print('test accuracy:',model.get_accuracy(X_test,y_test))
    '''

    