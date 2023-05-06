#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:21:32 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
import libsvm2array
import time
import math
#%%
class TR(object):
    def __init__(self,X,Y,gtol=1e-5,eta1=1e-1,eta2=0.9,gamma1=0.5,gamma2=2,loss='logistic',iteration=32):
        self.gtol = gtol
        self.eta_1 = eta1
        self.eta_2 = eta2
        self.gamma_1 = gamma1
        self.gamma_2 = gamma2
        self.loss = loss
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n_iteration = iteration
        self.w = np.zeros(self.n_features)
        self.w_s = [self.w]
        self.grad_s = []
        self.loss_s = []
        
        self.delta_bar = np.sqrt(self.n_samples) #初始化信赖域半径的上界为样本个数的平方根
        self.delta = (self.delta_bar / 8.) #初始化信赖域半径是信赖域半径上界的八分之一

        
        if loss == 'logistic':
            self.mu = 1e-2 / len(self.Y) #罗辑回归目标函数的罚因子
    
    
    def obj_loss(self,w):
        if self.loss == 'logistic':
            _sum = 0.
            for i in range(self.n_samples):
                _sum += np.log(1 + np.e**(-self.Y[i]*w@self.X[i])) \
                    + self.mu * np.linalg.norm(w,2)**2
            return (1/self.n_samples) * _sum
    
    def grad(self):
        if self.loss == 'logistic':
            _sum = 0.
            for i in range(self.n_samples):
                _sum += (1/(1 + np.e**(-self.Y[i]*self.w@self.X[i]))) \
                    * np.e**(-self.Y[i]*self.w@self.X[i]) * (-self.Y[i]*self.X[i]) \
                        + 2 * self.mu * self.w
            return (1/self.n_samples) * _sum

    
    #有待改善的地方，hess矩阵如果规模较大不能够直接储存，所以这里应该得到hess矩阵和向量的乘积，保存乘积要更加合理而且存储难度更小一点。
    def hess(self):
        if self.loss == 'logistic':
            _sum = 0.
            for i in range(self.n_samples):
                _sum += (np.e**(-self.Y[i]*self.w@self.X[i])/(1 + np.e**(-self.Y[i]*self.w@self.X[i]))**2) \
                    * (self.X[i].reshape(-1,1)@self.X[i].reshape(-1,1).T)
            return (1/self.n_samples) * _sum + 2 * self.mu * np.eye(self.n_features)
                     
    def fit(self):
        print('----- start training -----')
        self.loss_s.append(self.obj_loss(self.w))
        self.grad_s.append(np.linalg.norm(self.grad(),2))
        #使用截断共轭梯度法求解信赖域子问题
        #两个参数分别记录信赖域半径连续增大或减小的次数，以方便初始值的调整
        consecutive_TRplus = 0
        consecutive_TRminus = 0
        
        k = 0
        loss_dif = 100
        t1 = time.time()
        while k <= self.n_iteration and np.linalg.norm(self.grad(),2) >= self.gtol and loss_dif > self.gtol:
        #while k <= self.n_iteration and np.linalg.norm(self.grad(),2) >= self.gtol:  
            #达到一定精度或者一定迭代次数时停止迭代
            g = self.grad() #梯度
            B = self.hess() #海瑟矩阵
            d,stop_tCG = self.tCG(self.delta,self.grad(),self.hess())
            #print('delta:',self.delta)
            #print('截断共轭梯度法得到的方向：',d)
            print('截断共轭梯度法退出原因：',stop_tCG)
            w_ = self.w + d
            self.w_s.append(w_)
            self.loss_s.append(self.obj_loss(w_))
            #计算比值
            #rho_k = abs(((self.obj_loss(self.w) - self.obj_loss(w_))+0.001)\
            #            / ((self.m_k(self.w,np.zeros(len(d)),g,B) - self.m_k(self.w,d,g,B))+0.001))
            rho_k = abs(((self.m_k(self.w,np.zeros(len(d)),g,B) - self.m_k(self.w,d,g,B))+0.001)\
                        / ((self.obj_loss(self.w) - self.obj_loss(w_))+0.001))

            #rho_k = abs((self.obj_loss(self.w) - self.obj_loss(w_)) / (g@d + 0.5 * d@B@d))
            #print('under rho_k:',self.m_k(self.w,np.zeros(len(d)),g,B) - self.m_k(self.w,d,g,B))
            #确定是否更新信赖域半径
            #print('比值：',rho_k)
            if abs(1 - abs(rho_k - 1)) >= self.eta_1 and abs(1 - abs(rho_k - 1)) <= 1:
                print('rho_k={},此次更新值得信任。'.format(rho_k))
                #接受此次更新，并记录上一步的函数值和梯度
                self.grad_s.append(np.linalg.norm(g,2))
                self.w = self.w + d
                #self.w_s.append(self.w)
                #self.loss_s.append(self.obj_loss(self.w))
                #计算函数值相对变化
            #if loss_dif <= self.gtol:
            #   break
            #调整信赖域半径
            #print(self.loss_s)
            #if abs(1 - abs(rho_k - 1)) <= self.eta_1 or abs(1 - abs(rho_k - 1)) > 1 or \
            #    np.isnan(rho_k) or self.loss_s[-1] >= self.loss_s[-2]:
            if abs(1 - abs(rho_k - 1)) <= self.eta_1 or abs(1 - abs(rho_k - 1)) > 1 or \
                np.isnan(rho_k):
                print('rho_k={},此次更新不值得信任，缩减信赖域半径重新更新。'.format(rho_k))
                self.loss_s.pop()
                self.w_s.pop()
                self.delta = self.gamma_1 * self.delta
                consecutive_TRplus = 0
                consecutive_TRminus = consecutive_TRminus + 1
            #当信赖域连续5次减小时，认为当前的信赖域半径过大，并输出相应的提示信息
            if consecutive_TRminus >= 5:
                consecutive_TRminus = 0
                print(' +++ Detected many consecutive TR- (radius decreases). 信赖域减小次数很多\n')
                print(' +++ Consider decreasing Delta_bar by an order of magnitude. 考虑降低信赖域半径的上界\n')
                print(' +++ Current values: Delta_bar = {} and Delta0 = {}. \
                      当前的信赖域上界和初值分别是{},{}.\n'.format(self.delta_bar,self.delta,self.delta_bar,self.delta))
            
            if (abs(1 - abs(rho_k - 1)) >= self.eta_2 and abs(1 - abs(rho_k - 1)) <= 1) or \
                np.sqrt((np.linalg.norm(d,2) - self.delta)**2)<=self.gtol:
                print('近似效果很好，增大信赖域半径.')
                self.delta = max(self.gamma_2 * self.delta,self.delta_bar)
                consecutive_TRminus = 0
                consecutive_TRplus = consecutive_TRplus + 1
            #当信赖域连续5次增大时，认为当前的信赖域半径过小，并输出相应的提示信息
            if consecutive_TRminus >= 5:
                consecutive_TRplus = 0;
                print(' +++ Detected many consecutive TR+ (radius increases). 信赖域增大次数很多\n')
                print(' +++ Consider increasing Delta_bar by an order of magnitude. 考虑增加信赖域半径的上界\n')
                print(' +++ Current values: Delta_bar = {} and Delta0 = {}. \
                      当前的信赖域上界和初值分别是{},{}.\n'.format(self.delta_bar,self.delta,self.delta_bar,self.delta))
            #除了上面两种情况都不需要对信赖域半径进行调整
            #print(self.loss_s)
            if len(self.loss_s) > 1:
                loss_dif = abs((self.loss_s[-2] - self.loss_s[-1]) / (abs(self.loss_s[-2]) + 1))
            k = k + 1
            print('iteration:',k,'grad:',np.linalg.norm(self.grad(),2),'lossdif',loss_dif,'\n')
            print(stop_tCG)
        #从外层迭代退出记录了函数值，梯度值，目标向量w的变化
        t2 = time.time()
        self.time_consuming = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(self.time_consuming,k))
        print('训练准确率：',self.get_accuracy(self.X, self.Y))
        self.show_loss()
        self.show_grad()
        return 'Done!'
    
    def tCG(self,delta,g,B,kappa=1e-3,theta=1,max_iter=50):
        #kappa,theta是截断共轭梯度法中用于判断收敛的参数，牛顿方程求解精度的参数
        #返回迭代方向d，stop_tCG表示截断共轭梯度法退出的原因
        #初始化
        d = np.zeros(self.n_features) #目标方向
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
            if pHp <= 0:
                #曲率为负，沿负曲率方向搜寻
                tau = (-d@p + np.sqrt((d@p)**2 - p@p * (d@d - delta**2))) / p@p
                d_ = d + tau * p
                stop = 'nagative curvature 曲率是负值'
                return d_,stop
            alpha_i = r@r / pHp #更新步长
            #print('alpha_k',alpha_i,'pbp',pHp,'rr',r@r)
            d_ = d + alpha_i * p #更新迭代向量
            if np.linalg.norm(d_,2) >= delta:
                #超过信赖域边界时，找到tau,到达边界
                #print(d,r,p)
                tau = (-d@p + np.sqrt((d@p)**2 - p@p * (d@d - delta**2))) / p@p
                '''
                print('d',d,'p',p)
                print('dp',d@p,'dd',d@d,'delta',delta**2)
                print('sqrtlimiande:',(d@p)**2 - p@p * (d@d - delta**2),'waimiande:',np.sqrt((d@p)**2 - p@p * (d@d - delta**2)))
                print('pp',p@p)
                print('taotao',tau)
                '''
                d_ = d + tau * p
                stop = 'exceeded trust region 超过了信赖域'
                return d_,stop
            r_ = r + alpha_i*B@p #更新梯度
            if np.linalg.norm(r_,2) <= kappa * np.linalg.norm(r0,2):
                #(self.m_k(self.w,d_,g,B) - self.m_k(self.w,d,g,B))/self.m_k(self.w,d,g,B) <= 1kappa:
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
                
    def m_k(self,w,d,grad,hess):
        return self.obj_loss(w) + d@grad + 0.5 * d@hess@d
    
    def get_accuracy(self,X,Y):
        return (sum(self.predict(X) == Y)) / len(Y)
        
    def predict(self,x):
        if len(list(set(self.Y))) == 2:
            return np.where(x@self.w >= 0.0, 1, -1)
        else:
            pre = []
            for sample in list(x):
                dist = [abs(x@self.w - i) for i in list(set(self.Y))]
                pre.append(list(set(self.Y))[np.argmin(dist)])
            return pre
    
    def show_loss(self):
        plt.scatter(list(range(len(self.loss_s))), self.loss_s,s=5)
        plt.xlabel('Iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n}f_i (x)^2 $')
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
    
    def X_1(X):
        one = np.ones(X.shape[0])
        X_ = np.column_stack((X,one))
        return X_
    
    #'''
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
    x = X_1(x)
    #x_t,y_t = libsvm_to_metrix.libsvm_to_metrix(path2)
    x_t,y_t = libsvm2array.svm_read_problem(path2)
    x_t = X_1(x_t)
    print('----  done  ----')
    
    X_train,y_train = x,y
    X_test,y_test = x_t,y_t
    
    
    tr = TR(X_train,y_train,loss='logistic')
    tr.fit()
    print('test accuracy:',tr.get_accuracy(X_test,y_test))
    #'''
    