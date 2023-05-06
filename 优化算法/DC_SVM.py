#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:40:22 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')

import libsvm2array
import time
import random
random.seed(3047)
#%%
class DC():
    def __init__(self,iteration=32,loss='l1'):
        self.n_iter = iteration
        self.loss = loss
        self.epsilon = 1e-2
        
        
    def fit(self,X,Y):
        self.X = X
        self.Y = Y
        self.n_s = X.shape[0]
        self.n_f = X.shape[1]
        self.w = np.zeros(self.n_f)
        self.b = 0.
        self.alpha = np.zeros(self.n_s) - 1
        self.C = np.ones(self.n_s)
        self.L = np.zeros(self.n_s)
        self.U = np.ones(self.n_s)
        self.loss_s = [100]
        
        u = sum([Y[i]*self.alpha[i]*X[i] for i in range(self.n_s)])
        t = 0
        f_diff = 100
        t1 = time.time()
        while (self.is_optimal_alpha(self.alpha) or \
            abs(Y.T@self.alpha) >= self.epsilon) and \
                t <= self.n_iter and \
                    f_diff >= self.epsilon:
            per_id = np.random.permutation(list(range(self.n_s)))
            for j in range(self.n_s):
                i = per_id[j]
                G = 0
                Q_ii = 0
                if self.loss == 'l1':
                    G = Y[i]*u.T@X[i] - 1
                    Q_ii = Y[i]*Y[i]*X[i].T@X[i]
                    
                elif self.loss == 'l2':
                    G = Y[i]*u.T@X[i] - 1 + self.alpha[i]/(2*self.C[i])
                    Q_ii = Y[i]*Y[i]*X[i].T@X[i] + 1/(2*self.C[i])
                #print(Q_ii)
                d = max(-self.alpha[i], min(self.C[i] - self.alpha[i],-G/Q_ii))
                self.alpha[i] = self.alpha[i] + d
                u = u + d*Y[i]*X[i]
            self.loss_s.append(self.loss_fun(u, 0))
            t += 1
            print('{}th iteration, loss = {}'.format(t, self.loss_s[-1]))
            f_diff = abs((self.loss_s[-1] - self.loss_s[-2])/self.loss_s[-1])
            print('diff',f_diff)
            
        self.w = u
        t2 = time.time()
        t_all = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(t_all,t))
        print('训练准确率：',self.get_accuracy(X,Y))
        
        
        self.show_loss()
        
        
        return 'Done!'

    def fit2(self,X,Y):
        self.X = X
        self.Y = Y
        self.n_s = X.shape[0]
        self.n_f = X.shape[1]
        self.w = np.zeros(self.n_f)
        self.b = 0.
        self.alpha = np.zeros(self.n_s) - 1
        self.C = np.ones(self.n_s)
        self.L = np.zeros(self.n_s)
        self.U = np.ones(self.n_s)
        self.loss_s = [100]
        
        u = sum([Y[i]*self.alpha[i]*X[i] for i in range(self.n_s)])
        t = 0
        f_diff = 100
        t1 = time.time()
        print('---- start training ----')
        while (self.is_optimal_alpha(self.alpha) or \
            abs(Y.T@self.alpha) >= self.epsilon) and \
                t <= self.n_iter and \
                    f_diff >= self.epsilon:
            per_id = np.random.permutation(list(range(self.n_s)))
            id_2 = [(per_id[i],per_id[i+1]) for i in range(len(per_id)-1)]
            
            alpha_i_bar,alpha_j_bar = 0.,0.
            
            for _id in id_2:
                i,j = _id[0],_id[1]
                #print(i,j)
                p_i = self.grad_f_i(i)
                p_j = self.grad_f_i(j)
                delta = self.Q(i,i)*self.Q(j,j) - self.Q(i,j)**2
                use_j = False
                Q_ij = self.Q(i,j)
                Q_ii = self.Q(i,i)
                Q_jj = self.Q(j,j)
                #print(alpha_i_bar)
                alpha_i_bar = min(self.C[i],\
                                  max(0,self.alpha[i]+(-Q_jj*p_i+Q_ij*p_j)/delta))
                alpha_j_bar = min(self.C[j],\
                                  max(0,self.alpha[j]+(-Q_ii*p_j+Q_ij*p_i)/delta))
                if alpha_i_bar >= self.C[i]:
                    if Q_ii*(alpha_i_bar-self.alpha[i])+Q_ij*(alpha_j_bar-self.alpha[j])+p_i <= 0:
                        alpha_j_bar = min(self.C[j],\
                                          max(0,self.alpha[j]-(Q_ij*(alpha_i_bar-self.alpha[i])+p_j)/Q_jj))
                    else:
                        use_j = True
                elif alpha_i_bar <= 0:
                    if Q_ii*(alpha_i_bar-self.alpha[i])+Q_ij*(alpha_j_bar-self.alpha[j])+p_i >= 0:
                        alpha_j_bar = min(self.C[j],\
                                          max(0,self.alpha[j]-(Q_ij*(alpha_i_bar-self.alpha[i])+p_j)/Q_jj))
                    else:
                        use_j = True
                else:
                    use_j = True
                if use_j == True:
                    alpha_i_bar = min(self.C[i],\
                                      max(0,self.alpha[i]-(Q_ij*(alpha_j_bar-self.alpha[j])+p_i)/Q_ii))
                    
                u = u + (alpha_i_bar-self.alpha[i])*Y[i]*X[i] + (alpha_j_bar-self.alpha[j])*Y[j]*X[j]
                self.alpha[i] = alpha_i_bar
                self.alpha[j] = alpha_j_bar
            self.loss_s.append(self.loss_fun(u, 0))
            t += 1
            print('{}th iteration, loss = {}'.format(t, self.loss_s[-1]))
            f_diff = abs((self.loss_s[-1] - self.loss_s[-2])/self.loss_s[-1])
            print('diff',f_diff)
        
        self.w = u
        t2 = time.time()
        t_all = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(t_all,t))
        print('训练准确率：',self.get_accuracy(X,Y))
        
        self.show_loss()
        
        
        return 'Done!'
            
            
 
    def Q(self,i,j):
        if self.loss == 'l1':
            Q_ij = self.Y[i]*self.Y[j]*self.X[i].T@self.X[j]  
        elif self.loss == 'l2':
            if i == j:
                Q_ij = self.Y[i]*self.Y[j]*self.X[i].T@self.X[j] + 1/(2*self.C[i])
            else:
                Q_ij = self.Y[i]*self.Y[j]*self.X[i].T@self.X[j]
        return Q_ij
    
    def Q_i(self,i):
        return self.Y[i]*self.X[i].T@(self.X.T)*self.Y
    
    def grad_f_i(self,i):
        return self.Q_i(i)@self.alpha - 1.
    
    def loss_fun(self,w,b):
        if self.loss == 'l1':
            return 1/2 * np.linalg.norm(w,2) + \
                sum([max(0,1-self.Y[i]*(self.X[i]@w + b)) for i in range(self.n_s)])
        elif self.loss == 'l2':
            return 1/2 * np.linalg.norm(w,2) + \
                sum([(max(0,1-self.Y[i]*(self.X[i]@w + b)))**2 for i in range(self.n_s)])
                
    def is_optimal_alpha(self,alpha):
        judge = 1.
        for i in range(len(alpha)):
            if self.L[i] > alpha[i] or alpha[i] > self.U[i]:
                judge  = judge*0
                
            if judge == 1.:
                return 0
            else:
                return 1
    
    def get_accuracy(self,X,Y):
        return (np.sum(self.predict(X) == Y)) / len(Y)
        
    def predict(self,x):
        return np.where(x@self.w >= 0.0, 1, -1)
    
    def show_loss(self):
        plt.scatter(list(range(len(self.loss_s))), self.loss_s,s=5)
        plt.xlabel('Iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n} max({0,y_i f(x_i)}) $')
        plt.title('LOSS')
        plt.show()
        
            
    
    
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
    
    model = DC()
    model.fit2(X_train,y_train)
    print('测试准确率:',model.get_accuracy(X_test,y_test))
    #'''


    
    
    