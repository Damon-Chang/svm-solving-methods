#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 09:27:35 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm2array
from sklearn.model_selection import KFold



#%%
class RDSVMbyADMM(object):
    def __init__(self,iteration = 64):
        
        self.lambda_1 = 0.1 #L1正则项系数
        self.lambda_2 = 0.5 #L2正则项系数
        self.gamma = 1. #增广拉格朗日函数罚因子
        self.alpha = 1.8 #松弛因子
        self.delta = 1e-5
        self.n_iter = iteration

        
        #self.lambda_candidate = [0.001,0.01,0.1,0.2,0.5,1,2,5,10,100,1000]
        
    def fit(self,X,Y):
        self.X,self.Y = X,Y
        self.n_class = len(list(set(Y)))
        n_samples = X.shape[0] #n
        n_features = X.shape[1] #p
        self.n = n_samples
        self.u_1 = 100/n_samples
        self.u_2 = 50
        #self.lambda_1,self.lambda_2 = self._kfold_validation(X, Y, 5)
        #print('五折交叉验证完成，最优乘子为lambda_1=「{}」，lambda_2=「{}」'.format(self.lambda_1,self.lambda_2))
        self.lambda_1,self.lambda_2 = 0.01,100
        #初始化变量
        self.w = np.zeros(n_features) + 0.01
        self.b = 0.
        self.a = np.zeros(n_samples)
        #self.z = np.zeros(n_features - 1)
        self.c = np.zeros(n_features)
        self.u = np.zeros(n_samples) #n
        self.v = np.zeros(n_features) #p+1
        #self.u = np.array(list(self.u_1) + list(self.u_2))
        #self.v = -self.gamma * self.u
        
        self.E = []
        
        k = 0
        t1 = time.time()
        stop_cri1 = 1
        stop_cri2 = 2
        print('---------  start training  -------')
        while stop_cri1 > self.delta or stop_cri2 > self.delta:  
            #计算w^k
            
            Y_d = np.diag(Y)
            
            H = np.row_stack((np.column_stack((((self.lambda_2 + self.u_2)*np.eye(n_features)\
                                             + self.u_1*X.T@X).reshape(n_features,n_features), (self.u_1*X.T@np.ones(n_samples)).reshape(n_features,1))),\
                             np.column_stack(((self.u_1*np.ones(n_samples).T@X).reshape(1,n_features), np.array(self.u_1*n_samples).reshape(1,1)))))
            self.H = H
            #compute w^k
            w = (np.row_stack((((X.T@Y_d@self.u - self.u_1*X.T@Y_d@(self.a - 1) \
                                - self.v + self.u_2*self.c)).reshape(X.shape[1],1),\
                              np.ones(n_samples).T@Y_d@(self.u - self.u_1*self.a + self.u_1*1)))).flatten()
            #compute b^k
            
            result, stop = self.cg(H,w)
            #result = np.linalg.inv(H)@w
            #print(stop)
            self.w,self.b = result[0:-1],result[-1]
            _a = 1 + self.u/self.u_1 - Y_d@(X@self.w + self.b*np.ones(n_samples))
            self.a = self._project(_a,1/(n_samples*self.u_1))
            _c = self.v/self.u_2 + self.w
            self.c = self._fun_soft(_c,self.lambda_1/self.u_2)
            self.u = self.u + self.u_1*(1 - Y@(X@self.w + self.b*np.ones(n_samples)) - self.a)
            self.v = self.v + self.u_2*(self.w - self.c)
            
            stop_cri1 = 1/np.sqrt(n_samples)*np.linalg.norm(1-Y_d@(X@self.w+self.b*np.ones(n_samples))-self.a,2)
            stop_cri2 = 1/np.sqrt(n_features)*np.linalg.norm(self.w - self.c,2)
            self.E.append(stop_cri1)
            
            k += 1
            print('{}th iteration,loss = 「{}」'.format(k,self.E[-1]))
            
            
        self.show_loss(self.E)
        
        t2 = time.time()
        _time = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(_time,k))
        print('训练准确率：',self.get_accuracy(X,Y))
        return self.get_accuracy(X, Y)
    
    def Phi(self,w,b):
        return 1/self.n * sum([max(0,1-self.Y[i]*(self.X[i].T@self.w+self.b)) for i in range(self.n)]) +\
            self.lambda_1*np.linalg.norm(self.w,1) + self.lambda_2*np.linalg.norm(self.w,2)**2 / 2
    
    def cg(self,A, b, eta=1e-3, i_max=50):
        """共轭梯度法求解方程Ax=b，这里A为对称正定矩阵
        Args:
            A: 方程系数矩阵
            b:
            x: 初始迭代点
            eta: 收敛条件
            i_max: 最大迭代次数
        Returns: 方程的解x
        """
        x = np.zeros(A.shape[1])
        i = 0 # 迭代次数
        r_0 = np.dot(A, x)-b
        p = -r_0
        r_norm = np.linalg.norm(r_0)
        while r_norm > eta and i < i_max:
            alpha = np.dot(r_0,r_0) / np.dot(p, np.dot(A, p))
            x = x + alpha * p
            r_1 = r_0 + alpha * np.dot(A, p)
            
            beta = np.dot(r_1, r_1) / np.dot(r_0, r_0)
            p = -r_1 + beta * p
            r_0 = r_1
            r_norm = np.linalg.norm(r_0)
            i = i + 1
        
        return x,i  
    
    def CG(self,g,B,kappa=1e-3,theta=1,max_iter=50):
        #kappa,theta是截断共轭梯度法中用于判断收敛的参数，牛顿方程求解精度的参数
        #返回迭代方向d，stop_tCG表示截断共轭梯度法退出的原因
        #初始化
        
        
        d = np.zeros(len(g)) #目标方向
        r0 = g
        r = g #初始化梯度
        p = -r #初始化共轭向量
        #_p = []
        #_r = []
        #_d = []
        if np.linalg.norm(r) <= kappa:
            d_ = d
            stop = '第一步取负梯度方向就收敛。'
            return d_,stop
        
        for i in range(max_iter):
            #_p.append(p)
            #_r.append(r)
            Hp = B@p.reshape(-1,1)
            #计算曲率和共轭法的步长
            pHp = p@Hp
            alpha_i = r@r / pHp #更新步长
            #print('alpha_k',alpha_i,'pbp',pHp,'rr',r@r)
            d_ = d + alpha_i * p #更新迭代向量
            r_ = r + alpha_i*B@p #更新梯度
            if np.linalg.norm(r_,2) <= kappa * np.linalg.norm(r0,2):
                stop = 'superlinear convergence 超线性收敛'
                return d_,stop
            beta_i = np.linalg.norm(r_,2) / np.linalg.norm(r,2) #更新组合系数
            p_ = -r_ + beta_i * p #更新共轭向量
            #print('beta',beta_i)
            d = d_
            p = p_
            r = r_
            #print('信赖域内更新')
            stop = 'maximal iteration number reached 达到最大迭代次数'
        return d,stop
    
    def _project(self,x,kappa):
        result = []
        for x_i in x:
            p_kappa = np.sign(x_i)*max(0,abs(x_i - kappa/2) - kappa/2)
            result.append(p_kappa)
        return np.array(result)
    
    def _fun_soft(self,x,kappa):
        result = []
        for x_i in x:
            s_kappa = np.sign(x_i)*max(0,abs(x_i) - kappa)
            result.append(s_kappa)
        return np.array(result)
    
    def get_accuracy(self,X,Y):
        return (sum(self.predict(X) == Y)) / len(Y)
        
    def predict(self,x):
        if self.n_class == 2:
            return np.where(x@self.w + self.b >= 0.0, 1, -1)
        else:
            pre = []
            for sample in list(x):
                dist = [abs(x@self.w - i) for i in list(set(self.Y))]
                pre.append(list(set(self.Y))[np.argmin(dist)])
            return pre
    
    def show_loss(self,l):
        plt.scatter(list(range(len(l))), l,s=5)
        plt.xlabel('Iteration')
        plt.ylabel(r'$ || t^k-t^{k-1}||_2 $')
        plt.title('LOSS')
        plt.show()
        
    def show_grad(self):
        plt.scatter(list(range(len(self.grad_s))), self.grad_s,s=5)
        plt.xlabel('Iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n}\nabla f_i (x)^2 $')
        plt.title('Grad value')
        plt.show()

    
#%%    
if __name__ == '__main__':
    '''
    def X_1(X):
        one = np.ones(X.shape[0])
        X_ = np.column_stack((X,one))
        return X_
    '''
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
    print('----  data processing  ----')
    x,y = libsvm2array.svm_read_problem(path1)
    #x = X_1(x)
    #x_t,y_t = libsvm_to_metrix.libsvm_to_metrix(path2)
    x_t,y_t = libsvm2array.svm_read_problem(path2)
    #x_t = X_1(x_t)
    print('----  done  ----')
    
    X_train,y_train = x,y
    X_test,y_test = x_t,y_t
    
    model = RDSVMbyADMM()
    model.fit(X_train,y_train)
    print('test accuracy:',model.get_accuracy(X_test,y_test))
 
    
    
    