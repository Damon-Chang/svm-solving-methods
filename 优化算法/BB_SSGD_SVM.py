#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:14:24 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
import libsvm2array
import time
import random
random.seed(3047)
#%%
class SSGD(object):
    def __init__(self,iteration=1024,epsilon=1e-5,eta0=0.1,use_BB_step = True,selection = 1):
        
        self.n_iter = iteration
        self.epsilon = epsilon
        self.eta0 = eta0 #initialize learning rate
        #self._lambda = 5
        self.use_BB_step = use_BB_step
        self.selection = selection
        
        
        
        
    def fit(self,X,Y):
        t1 = time.time()
        print('------ Start training... ------')
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.w = np.zeros(self.n_features)
        #self._lambda = self.n_samples / 10
        #self._lambda = np.sqrt(self.n_samples)
        self._lambda = 100
        
        self.tau = 1 #[2**6,2**4,1,2**(-2),2**(-4),2**(-6),0]
        
        self.w_s = []
        self.i = []
        self.loss = [self.hinge_loss(X,Y,self.w)]
        self.loss_s = []
        diff = 100
        t = 0
        while diff >= self.epsilon and t <= self.n_iter:
            t += 1
            eta = self.eta0 / (1 + self._lambda*self.eta0*t) #消失步长
            if t >= 4 and self.use_BB_step:
                '''
                if t % 2 == 0:
                    eta = self.BB(selection=2) / t
                else:
                    eta = self.BB(selection=1) / t
                '''
                #取BB步长
                #eta = BB_family(X,Y,self.i,self.w_s,selection=1).BB_step
                eta = self.BB(selection=1) / t
                print('bbbbb')
                #eta = self.BB(selection=1)
                #print('BB_step:', eta)
            i_t = np.random.randint(self.n_samples)
            self.i.append(i_t)
            
            self.w_s.append(self.w)
            self.w = self.w - eta*self.grad_l_i(i_t,self.w)
            #print('up:',self.w)
            #self.w_s.append(self.w)
            #print('w_s:',len(self.w_s))
            del self.i[0:-3]
            del self.w_s[0:-3]
            
            self.loss.append(self.hinge_loss(X,Y,self.w))
            diff = abs(self.loss[-1] - self.loss[-2]) / self.loss[-1]
            self.loss_s.append(diff)
            print('{}th iteration,loss = 「{}」'.format(t,self.loss_s[-1]))
        
        self.T = t
        
        t2 = time.time()
        t_all = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(t_all,t))
        print('训练准确率：',self.get_accuracy(X,Y))
        
        
        self.show_loss()
        
        #return self.w[0:-1],self.w[-1]
        return 'Done!'
    
    def sub_grid(self,):#get all gradient
        result = np.zeros(self.n_features)
        for i in range(self.n_samples):
            if self.Y[i]*self.activation(self.X[i],self.w) < 1.:
                result += -self.Y[i]*self.X[i]
        
        return result + self._lambda*self.w
        
    def grad_l_i(self,i,w):
        if self.Y[i]*self.activation(self.X[i],w) < 1.:
            return -self.Y[i]*self.X[i] + self._lambda*w
        else:
            return self._lambda*w
        
    def hinge_loss(self,X,y,w):
        return sum([max(0,1-y[i]*self.activation(X[i],w)) for i in range(len(y))])\
            + self._lambda* np.linalg.norm(w,2) / 2
        
    def activation(self,x,w):
        #f(x)=wx+b
        return x@w
    
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
        
    def BB(self,selection):
        self.s_t_1 = self.w_s[-1] - self.w_s[-2]
        
        if self.selection == 1:
            self.BB_step = self.BB1()
        elif self.selection == 2:
            self.BB_step == self.BB2()
        elif self.selection == 3:
            self.BB_step == self.BB3()
        elif self.selection == 4:
            self.BB_step == self.BBR1()
        elif self.selection == 5:
            self.BB_step == self.BBR2()
        else:
            print('not avaliable!')
        return self.BB_step
        
        
    def BB1(self):
        #print(self.i)
        v_t_1 = self.grad_l_i(self.i[-1],self.w_s[-1]) - \
            self.grad_l_i(self.i[-2],self.w_s[-2])
        #print(v_t_1.shape)
        #print('s',self.s_t_1,'v',v_t_1)
        return self.s_t_1@self.s_t_1 / (self.s_t_1@v_t_1)
    def BB2(self):
        v_t_1 = self.grad_l_i(self.i[-2], self.w_s[-1]) - \
            self.grad_l_i(self.i[-2], self.w_s[-2])
        return self.s_t_1@self.s_t_1 / (self.s_t_1@v_t_1 + 0.01)
    def BB3(self):
        v_t_1 = self.grad_l_i(self.i[-1], self.w_s[-1]) - \
            self.grad_l_i(self.i[-1], self.w_s[-1])
        return self.s_t_1@self.s_t_1 / self.s_t_1@v_t_1
    def BBR1(self):
        v_t_1 = self.grad_l_i(self.i[-2], self.w_s[-2]) - \
            self.grad_l_i(self.i[-3], self.w_s[-3])
        return self.s_t_1@self.s_t_1 / self._t_1@v_t_1
    def BBR2(self):
        v_t_1 = self.grad_l_i(self.i[-1], self.w_s[-2]) - \
            self.grad_l_i(self.i[-2], self.w_s[-3])
        return self.s_t_1@self.s_t_1 / self._t_1@v_t_1
    
    
    
        
        
#%%
class BB_family(SSGD):
    def __init__(self,X,Y,i,w_s,selection):
        super().__init__()
        
        self.X = X
        self.Y = Y
        self.i = i
        self.w_s = w_s
        self.selection = selection
        #self.x_s = self.X[self.i]
        self.s_t_1 = self.w_s[-1] - self.w_s[-2]

        
        if self.selection == 1:
            self.BB_step = self.BB1()
        elif self.selection == 2:
            self.BB_step == self.BB2()
        elif self.selection == 3:
            self.BB_step == self.BB3()
        elif self.selection == 4:
            self.BB_step == self.BBR1()
        elif self.selection == 5:
            self.BB_step == self.BBR2()
        else:
            print('not avaliable!')
        
        
    def BB1(self):
        print(self.i)
        v_t_1 = self.grad_l_i(self.i[-1],self.w_s[-1]) - \
            self.grad_l_i(self.i[-2],self.w_s[-2])
        #print(v_t_1.shape)
        print('s',self.s_t_1,'v',v_t_1)
        return self.s_t_1@self.s_t_1 / (self.s_t_1@v_t_1)
    def BB2(self):
        v_t_1 = self.grad_l_i(self.i[-2], self.w_s[-1]) - \
            self.grad_l_i(self.i[-2], self.w_s[-2])
        return self.s_t_1@self.s_t_1 / (self.s_t_1@v_t_1 + 0.01)
    def BB3(self):
        v_t_1 = self.grad_l_i(self.i[-1], self.w_s[-1]) - \
            self.grad_l_i(self.i[-1], self.w_s[-1])
        return self.s_t_1@self.s_t_1 / self.s_t_1@v_t_1
    def BBR1(self):
        v_t_1 = self.grad_l_i(self.i[-2], self.w_s[-2]) - \
            self.grad_l_i(self.i[-3], self.w_s[-3])
        return self.s_t_1@self.s_t_1 / self._t_1@v_t_1
    def BBR2(self):
        v_t_1 = self.grad_l_i(self.i[-1], self.w_s[-2]) - \
            self.grad_l_i(self.i[-2], self.w_s[-3])
        return self.s_t_1@self.s_t_1 / self._t_1@v_t_1
    
    
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
    
    model1 = SSGD(use_BB_step=True)
    model1.fit(X_train,y_train)
    print('test accuracy:',model1.get_accuracy(X_test,y_test))
    #'''

