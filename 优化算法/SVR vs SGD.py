#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 12:41:35 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
from sklearn.linear_model import SGDClassifier
import time
from numpy import linalg as la
import random


#%%
#使用heart数据
path = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/heart/heart.txt'
x,y = libsvm_to_metrix.libsvm_to_metrix(path)
X_train,y_train,X_test,y_test = x[:200],y[:200],x[200:],y[200:]



#%%
#使用sklearn机器学习算法库中的SGD来实现
sgd_clf = SGDClassifier(random_state=42)
t1 = time.time()
sgd_clf.fit(X_train, y_train)
t2 = time.time()
pred = sgd_clf.predict(X_test)
acc = sum(pred == y_test) / len(y_test)
print('模型预测准确率：', acc)

#%%
#定定义一个新的类来实现SGD算法，输入x,y，目标函数sum(wx+b-y)，[w,b]是需要训练的参数
class SGD(object):
    def __init__(self, n_iter=300, epsilon=1e-5, m=10, learning_rate = 0.01,loss='hinge',gamma=1):
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.lr = learning_rate
        self.w = 0
        self.loss = loss
        self.gamma = gamma
        
        
        
      
    def object_fun(self,X,Y):
        #hinge损失，添加正则项
        _sum = 0
        if self.loss == 'hinge':
            for i in range(X.shape[0]):
                #sum=sum+math.log(1+math.exp(-b[i,:][0]*A[i,:]@x))
                #_sum = _sum + np.linalg.norm(self.w@X[i] + self.b - Y[i], ord=2)**2
                #_sum = _sum + np.linalg.norm(self.w@X[i] - Y[i], ord=2)**2
                #_sum = _sum + np.power(self.w@X[i] - Y[i], 2)
                _sum = _sum + max(0,1 - Y[i]*(self.w@X[i]))
                #f=(1/X.shape[0])*sum+lam*np.linalg.norm(x,ord=2)**2
            f = (1/X.shape[0]) * _sum
            return f
        
        elif self.loss == 'l2':
            for i in range(X.shape[0]):
                _sum = _sum + np.power(self.w@X[i] - Y[i], 2)
            f = (1/X.shape[0]) * _sum
            return f
    
    def obj_fun_grad(self,X,Y):
        grad = 0
        if self.loss == 'hinge':
            for i in range(X.shape[0]):
                #grad=grad+(-math.exp(-b[i,:][0]*A[i,:]@x)*b[i,:][0]*A[i,:].T)/(1+math.exp(-b[i,:][0]*A[i,:]@x))+2*lam*x
                #grad = grad + 2*X[i]*(self.w@X[i] + self.b - Y[i])
                if Y[i]*self.w@X[i] < 1:
                    #分类错误时
                    grad += np.linalg.norm(-Y[i]*X[i] + 2*(self.gamma / (i+1))*self.w,2)
                else:
                    grad += np.linalg.norm(2*(self.gamma / (i+1))*self.w,2)
                #grad = grad + 2*X[i]*(self.w@X[i] - Y[i])
            gradF=grad/X.shape[0]
            return gradF
        elif self.loss == 'l2':
            for i in range(X.shape[0]):
                grad += np.linalg.norm(2*X[i]*(self.w@X[i] - Y[i]),2)
            gradF = grad / X.shape[0]
            return gradF
    
    def fit(self,X,Y):
        self.w = np.zeros((X.shape[1])) #初始化0向量，长度是len(w)
        #self.b = np.zeros((X.shape[1],1))
        F=[]
        self.F_grad = []
        f = self.object_fun(X,Y)
        F.append(f) #存储每一次迭代的函数值
        deltaF = 1 #目标函数前后差值
        #gradF = self.obj_fun_grad(X, Y)
        
        T=0
        self.ws=[self.w]
        #while abs(deltaF/f)>0.00001: #相对差值
        while self.obj_fun_grad(X,Y) >= self.epsilon and T < self.n_iter:
            k=np.random.randint(0,X.shape[0])
            #gradkx=(-math.exp(-b[k,:][0]*A[k,:]@x)*b[k,:][0]*A[k,:].T)/(1+math.exp(-b[k,:][0]*A[k,:]@x))+2*lam*x
            #gradky=(-math.exp(-b[k,:][0]*A[k,:]@y)*b[k,:][0]*A[k,:].T)/(1+math.exp(-b[k,:][0]*A[k,:]@y))+2*lam*y     
            if self.loss == 'hinge':
                if Y[k]*self.w@X[k] < 1:
                    #分类错误时
                    gradk_w =  -Y[k]*X[k] + 2*(self.gamma / (T+1))*self.w
                else:
                    gradk_w = 2*(self.gamma / (T+1))*self.w
                    
            if self.loss == 'l2':
                gradk_w =  2*X[k]*(self.w@X[k] - Y[k]) 
            
            self.w = self.w - self.lr * gradk_w
            f = self.object_fun(X, Y)
            F.append(f)
            deltaF = F[-1] - F[-2]
            #print(deltaF/f,np.power(sum(self.obj_fun_grad(X,Y)), 2),T)
            self.F_grad.append(self.obj_fun_grad(X, Y))
            self.ws.append(self.w)
            T = T + 1
            
        return 'Done!'
        #plt.scatter(list(range(len(F))), F,s=5)
        
        #print(f)  
    def show_loss(self,):
        plt.scatter(list(range(len(self.F_grad))), self.F_grad,s=5)
        plt.xlabel('iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n}\nabla f_i (x)^2 $')
        plt.title('SVRG')
        plt.legend('gradient of '+ self.loss + ' loss')
        plt.show()
        
    def judge_fun(self,X):
        #return self.w@X
        return X@self.w
    def predict(self, X):
        return np.where(self.judge_fun(X)>=0.0, 1, -1) #输出值大于0预测值为1，否则为0
#%%
class SGD(object):
    def __init__(self, eta=0.01, n_iter=100, shuffle=True, random_state=None):
        self.eta = eta #学习率
        self.n_iter = n_iter #迭代次数，默认10
        self.w_initialized = False #参数初始化
        self.shuffle = shuffle #打乱数据顺序
        self.random_state = random_state #模型随机数
        
    def fit(self, X, y):
        self._initialize_weights(X.shape[1]) #X(n_sample,n_feature) w(n_feature)
        self.cost_ = []
        '''
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y) #打乱后的数据
            cost = []
        '''
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target)) #随机梯度更新
                
                #self.cost_.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y) #计算平均损失
            self.cost_.append(avg_cost)
        plt.scatter(list(range(len(self.cost_))), self.cost_,s=5)
        plt.xlabel('iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n}f_i (x)^2 $')
        plt.title('SGD')
        plt.show()
        #print(self.cost_)
        return self
        
    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else: 
            self._update_weights(X, y)
        return self
            
        
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state) #随机数生成器
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m ) #正态分布随机分布随机化参数
        self.w_initialized = True
     
    # Shuffle the data
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y)) # use this method to get a randomly array
        return X[r], y[r] # return the randomly array, this kind of indexing exsiting only in np
        
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi)) #输出wx+b
        error = target - output #误差
        self.w_[0] += self.eta * error #更新b
        self.w_[1:] += self.eta * xi.dot(error) #更新w
        cost = 0.5 * error**2 #平方损失
        return cost
        
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1) #输出值大于0预测值为1，否则为0
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0] #这里返回网络的输出y=wx+b
    
    
#%%


#%%
SGDSolver = SGD(loss = 'l2')
SGDSolver.fit(X_train, y_train)
print('acc sgd= ',sum(SGDSolver.predict(X_test) == y_test) / len(y_test))
SGDSolver.show_loss()
#%%
#||wx+b-y||^2
class SVRG(object):
    def __init__(self, n_iter=300, epsilon=1e-5, m=10, learning_rate = 0.01,loss='hinge',gamma=1):
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.lr = learning_rate
        self.m = m
        self.w = 0
        self.w_ = 0
        self.loss = loss
        self.gamma = gamma #learning rate
        
        
        
      
    def object_fun(self,X,Y):
        #hinge损失，添加正则项
        _sum = 0
        if self.loss == 'hinge':
            for i in range(X.shape[0]):
                #sum=sum+math.log(1+math.exp(-b[i,:][0]*A[i,:]@x))
                #_sum = _sum + np.linalg.norm(self.w@X[i] + self.b - Y[i], ord=2)**2
                #_sum = _sum + np.linalg.norm(self.w@X[i] - Y[i], ord=2)**2
                #_sum = _sum + np.power(self.w@X[i] - Y[i], 2)
                _sum = _sum + max(0,1 - Y[i]*(self.w@X[i]))
                #f=(1/X.shape[0])*sum+lam*np.linalg.norm(x,ord=2)**2
            f = (1/X.shape[0]) * _sum
            return f
        
        elif self.loss == 'l2':
            for i in range(X.shape[0]):
                _sum = _sum + np.power(self.w@X[i] - Y[i], 2)
            f = (1/X.shape[0]) * _sum
            return f
    
    def obj_fun_grad(self,X,Y):
        grad = 0
        if self.loss == 'hinge':
            for i in range(X.shape[0]):
                #grad=grad+(-math.exp(-b[i,:][0]*A[i,:]@x)*b[i,:][0]*A[i,:].T)/(1+math.exp(-b[i,:][0]*A[i,:]@x))+2*lam*x
                #grad = grad + 2*X[i]*(self.w@X[i] + self.b - Y[i])
                if Y[i]*self.w@X[i] < 1:
                    #分类错误时
                    grad += np.linalg.norm(-Y[i]*X[i] + 2*(self.gamma / (i+1))*self.w,2)
                else:
                    grad += np.linalg.norm(2*(self.gamma / (i+1))*self.w,2)
                #grad = grad + 2*X[i]*(self.w@X[i] - Y[i])
            gradF=grad/X.shape[0]
            return gradF
        elif self.loss == 'l2':
            for i in range(X.shape[0]):
                grad += np.linalg.norm(2*X[i]*(self.w@X[i] - Y[i]),2)
            gradF = grad / X.shape[0]
            return gradF
    
    def fit(self,X,Y):
        t=0
        #x=np.zeros((X.shape[1],1)) #初始化0向量，长度是len(w)
        #y=x
        self.w = np.zeros((X.shape[1])) #初始化0向量，长度是len(w)
        #self.b = np.zeros((X.shape[1],1))
        
        
        F=[]
        self.F_grad = []
        f = self.object_fun(X,Y)
        F.append(f) #存储每一次迭代的函数值
        deltaF = 1 #目标函数前后差值
        #gradF = self.obj_fun_grad(X, Y)
        
        T=0
        self.ws=[self.w]
        #while abs(deltaF/f)>0.00001: #相对差值
        while self.obj_fun_grad(X,Y) >= self.epsilon and T<self.n_iter:
            print(self.obj_fun_grad(X, Y))
            #while np.power(sum(self.obj_fun_grad(X,Y)), 2) >= self.epsilon:
        #     while T<10000:
            self.w_ = self.ws[-1]
             #存储迭代点列
            t=0
            grad=0
            for i in range(X.shape[0]):
                #grad=grad+(-math.exp(-b[i,:][0]*A[i,:]@y)*b[i,:][0]*A[i,:].T)/(1+math.exp(-b[i,:][0]*A[i,:]@y))+2*lam*y
                if self.loss == 'hinge':
                    if Y[i]*self.w_@X[i] < 1:
                        #分类错误时
                        grad += -Y[i]*X[i] + 2*(self.gamma / (T+1))*self.w_
                    else:
                        grad += 2*(self.gamma / (T+1))*self.w_
                elif self.loss == 'l2':
                    grad = grad + (2*X[i]*(self.w_@X[i] - Y[i]))
            mu = grad / X.shape[0]
            self.w = self.w_
            while t < self.m:
                t=t+1
                #T=T+1
                k=np.random.randint(0,X.shape[0])
                #gradkx=(-math.exp(-b[k,:][0]*A[k,:]@x)*b[k,:][0]*A[k,:].T)/(1+math.exp(-b[k,:][0]*A[k,:]@x))+2*lam*x
                #gradky=(-math.exp(-b[k,:][0]*A[k,:]@y)*b[k,:][0]*A[k,:].T)/(1+math.exp(-b[k,:][0]*A[k,:]@y))+2*lam*y
                
                if self.loss == 'hinge':
                    if Y[k]*self.w@X[k] < 1:
                        #分类错误时
                        gradk_w =  -Y[k]*X[k] + 2*(self.gamma / (T+1))*self.w
                        gradk_w_ =  -Y[k]*X[k] + 2*(self.gamma / (T+1))*self.w_
                    else:
                        gradk_w = 2*(self.gamma / (T+1))*self.w
                        gradk_w_ = 2*(self.gamma / (T+1))*self.w_
                        
                if self.loss == 'l2':
                    gradk_w =  2*X[k]*(self.w@X[k] - Y[k]) 
                    gradk_w_ =  2*X[k]*(self.w_@X[k] - Y[k])
                v = gradk_w - (gradk_w_ - mu)
                #x=x-0.01*v
                self.w = self.w - self.lr * v
                #self.ws.append(self.w)
                '''
                _sum = 0
                for i in range(X.shape[0]):
                    #sum=sum+math.log(1+math.exp(-b[i,:][0]*A[i,:]@x))
                    #_sum = _sum + np.linalg.norm(self.w@X[i] - Y[i], ord=2)**2
                    _sum = _sum + np.power(self.w@X[i] - Y[i], 2)
                f=(1/X.shape[0]) * _sum
                '''
                f = self.object_fun(X, Y)
                F.append(f)
                deltaF = F[T] - F[T-1]
                #print(deltaF/f,np.power(sum(self.obj_fun_grad(X,Y)), 2),T)
            self.F_grad.append(self.obj_fun_grad(X, Y))
            self.ws.append(self.w)
            T = T + 1
            
        return 'Done!'
        #plt.scatter(list(range(len(F))), F,s=5)
        
        #print(f)  
    def show_loss(self,):
        plt.scatter(list(range(len(self.F_grad))), self.F_grad,s=5)
        plt.xlabel('iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n}f_i (x)^2 $')
        plt.title('SVRG')
        plt.legend('gradient of hinge loss')
        plt.show()
        
    def judge_fun(self,X):
        #return self.w@X
        return X@self.w
    def predict(self, X):
        return np.where(self.judge_fun(X)>=0.0, 1, -1) #输出值大于0预测值为1，否则为0
    
#%%
#SVRG train data prepare
def x_svrg(X):
    one = np.ones(X.shape[0])
    X_ = np.column_stack((X,one))
    return X_
#%%
svrg = SVRG(loss='l2')
XX = x_svrg(X_train)
XX_test = x_svrg(X_test)
svrg.fit(XX,y_train)
#%%
svrg.show_loss()
#%%
print('acc svgr = ',sum(svrg.predict(XX_test) == y_test) / len(y_test))




