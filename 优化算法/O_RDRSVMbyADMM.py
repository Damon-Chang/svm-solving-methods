#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:46:15 2023

@author: damonchang
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
import libsvm2array
from sklearn.model_selection import KFold



#%%
class RDSVMbyADMM(object):
    def __init__(self,iteration = 64):
        
        self.lambda_1 = 1 #L1正则项系数
        self.lambda_2 = 1 #L2正则项系数
        self.gamma = 1. #增广拉格朗日函数罚因子
        self.alpha = 1.8 #松弛因子
        self.delta = 1e-3
        self.n_iter = iteration
        
        self.lambda_candidate = [0.001,0.01,0.1,0.2,0.5,1,2,5,10,100,1000]
                
    
    def _fit(self,X,Y):
        
        n_samples = X.shape[0] #n
        n_features = X.shape[1] #p+1
        X_0 = np.delete(X,-1,axis=1)
        #self.w = w
        self.b = np.zeros(n_features)
        self.e = np.zeros(n_samples)
        #self.z = np.zeros(n_features - 1)
        self.z = np.zeros(n_features)
        self.u_1 = np.zeros(n_samples) #n
        self.u_2 = np.zeros(n_features) #p+1
        self.u = np.array(list(self.u_1) + list(self.u_2))
        self.v = -self.gamma * self.u
        
        
        t = np.array(list(self.e) + list(self.z) + list(self.v))
        T = [100]
        self.T_diff = []
        T.append(t)
        #ADMM的终止准则要看两个残差是否充分小，多快的情况就并非如此了
        k = 0
        while np.linalg.norm(T[-1] - T[-2], 2) > np.sqrt(2*n_samples + 2*n_features)*self.delta and k <= self.n_iter:
            #计算w^k
            Y_d = np.diag(Y)
            H = X
            #compute w^k
            w = (np.row_stack(((X_0.T@Y_d@(self.u_1 - self.e + 1)).reshape(X_0.shape[1],1),\
                              np.ones(n_samples).T@Y_d@(self.u_1 - self.e + 1)))).flatten() - \
                (-self.z + self.u_2)
            #print(w.shape)
            #compute b^k
            eta = self.lambda_2 / (self.gamma + 1)
            if n_samples >= n_features:
                #直接求逆
                self.b = np.linalg.inv(eta*np.eye(n_features) + H.T@H)@w
                #print('bshape',self.b.shape)
            else:
                #使用SWM公式
                self.b = (1/eta)*w - (1/eta**2)(X.T@(np.linalg.inv(np.eye(n_samples) + (1/eta)*X@X.T)@(X@w)))
            
            #update e
            e_old = self.e
            self.e = self._project(self.alpha*(1-Y_d@X@self.b) + (1-self.alpha)*self.e + self.u_1,1/self.gamma)
            z_old = self.z
            self.z = self._fun_soft(self.alpha*self.b+(1-self.alpha)*self.z + self.u_2,self.lambda_1/self.gamma)
            self.u_1 = self.u_1 + self.alpha*(1-Y_d@X@self.b) + (1-self.alpha)*e_old - self.e
            self.u_2 = self.u_2 + self.alpha*self.b + (1-self.alpha)*z_old - self.z
            
            
            t = np.array(list(self.e) + list(self.z) + list(self.v))
            T.append(t)
            self.T_diff.append(np.linalg.norm(T[-1] - T[-2],2))
            k += 1
            
            
        self.show_loss()
        
        #print('训练完成，用时「{}」秒，迭代「{}」次。'.format(_time,k))
        #print('训练准确率：',self.get_accuracy(X,Y))
        return self.get_accuracy(X, Y)   
    
    def fit(self,X,Y):

        self.n_class = len(list(set(Y)))
        n_samples = X.shape[0] #n
        n_features = X.shape[1] #p+1
        X_0 = np.delete(X,-1,axis=1)
        #self.w = w
        
        #self.lambda_1,self.lambda_2 = self._kfold_validation(X, Y, 5)
        #print('五折交叉验证完成，最优乘子为lambda_1=「{}」，lambda_2=「{}」'.format(self.lambda_1,self.lambda_2))
        self.lambda_1,self.lambda_2 = 0.01,100
        #初始化变量
        self.b = np.zeros(n_features)
        self.e = np.zeros(n_samples)
        #self.z = np.zeros(n_features - 1)
        self.z = np.zeros(n_features)
        self.u_1 = np.zeros(n_samples) #n
        self.u_2 = np.zeros(n_features) #p+1
        self.u = np.array(list(self.u_1) + list(self.u_2))
        self.v = -self.gamma * self.u
        
        t = np.array(list(self.e) + list(self.z) + list(self.v))
        T = [100]
        self.T_diff = []
        T.append(t)
        #ADMM的终止准则要看两个残差是否充分小，多快的情况就并非如此了
        k = 0
        t1 = time.time()
        print('---------  start training  -------')
        while np.linalg.norm(T[-1] - T[-2], 2) >= np.sqrt(2*n_samples + 2*n_features)*self.delta and k <= self.n_iter:
           
            #计算w^k
            #A = np.column_stack((-np.eye(n_samples),np.zeros((n_samples,n_features)))).T
            Y_d = np.diag(Y)
            #B = np.column_stack((-X.T@Y_d,np.eye(n_features))).T
            #C = np.column_stack((np.zeros((n_features,n_samples)),-np.eye(n_features))).T
            #d = np.column_stack((np.ones(n_samples).reshape(1,n_samples),np.zeros((1,n_features)))).T
            
            H = X
            #compute w^k
            w = (np.row_stack(((X_0.T@Y_d@(self.u_1 - self.e + 1)).reshape(X_0.shape[1],1),\
                              np.ones(n_samples).T@Y_d@(self.u_1 - self.e + 1)))).flatten() - \
                (-self.z + self.u_2)
            #print(w.shape)
            #compute b^k
            eta = self.lambda_2 / (self.gamma + 1)
            #'''
            if n_samples >= n_features:
                #直接求逆
                self.b = np.linalg.inv(eta*np.eye(n_features) + H.T@H)@w
                #print('bshape',self.b.shape)
            else:
                #使用SWM公式
                self.b = (1/eta)*w - (1/eta**2)(H.T@((np.linalg.inv(np.eye(n_samples) + (1/eta)*H@H.T)@(H@w))))
            #'''
            #self.b, stop = self.tCG(-w, eta*np.eye(n_features) + H.T@H)
            #print(stop)
            
            #update e
            e_old = self.e
            self.e = self._project(self.alpha*(1-Y_d@X@self.b) + (1-self.alpha)*self.e + self.u_1,1/self.gamma)
            z_old = self.z
            self.z = self._fun_soft(self.alpha*self.b+(1-self.alpha)*self.z + self.u_2,self.lambda_1/self.gamma)
            self.u_1 = self.u_1 + self.alpha*(1-Y_d@X@self.b) + (1-self.alpha)*e_old - self.e
            self.u_2 = self.u_2 + self.alpha*self.b + (1-self.alpha)*z_old - self.z
            
            
            t = np.array(list(self.e) + list(self.z) + list(self.v))
            T.append(t)
            self.T_diff.append(np.linalg.norm(T[-1] - T[-2],2))
            k += 1
            print('{}th iteration,loss = 「{}」'.format(k,self.T_diff[-1]))
            
            
        self.show_loss()
        
        t2 = time.time()
        _time = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(_time,k))
        print('训练准确率：',self.get_accuracy(X,Y))
        return self.get_accuracy(X, Y)
    
    def tCG(self,g,B,kappa=1e-3,theta=1,max_iter=50):
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
    
    def _kfold_validation(self,X,Y,k):
        print('------k fold validation------')
        kf = KFold(n_splits=k,shuffle=False)  # 初始化KFold
        group_train = []
        group_test = []
        for train,test in kf.split(X):
            group_train.append(train.tolist())
            group_test.append(test.tolist())
        #kf_index = kf.split(X) # 调用split方法切分数据

        _acc = []
        ij = []
        q = 0
        for i,_lambda_1 in enumerate(list(self.lambda_candidate)):
            for j,_lambda_2 in enumerate(list(self.lambda_candidate)):
                q += 1
                print('{}/{} fold'.format(q,len(self.lambda_candidate)**2))
                self.lambda_1 = _lambda_1
                self.lambda_2 = _lambda_2
                acc = 0
                for l in range(k):
                    #kf_sel = list(range(len(group_test))).remove(k)
                    _X = X[group_train[l]]
                    _Y = Y[group_train[l]]
                    self._fit(_X, _Y)
                    __X = X[group_test[l]]
                    __Y = Y[group_test[l]]
                    acc += self.get_accuracy(__X, __Y)
                _acc.append(acc / k)
                ij.append([i,j])
        id_max = _acc.index(max(_acc))
        lambda_1,lambda_2 = self.lambda_candidate[ij[id_max][0]],\
            self.lambda_candidate[ij[id_max][1]]
        return lambda_1,lambda_2
            
            
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
            return np.where(x@self.b >= 0.0, 1, -1)
        else:
            pre = []
            for sample in list(x):
                dist = [abs(x@self.w - i) for i in list(set(self.Y))]
                pre.append(list(set(self.Y))[np.argmin(dist)])
            return pre
    
    def show_loss(self):
        plt.scatter(list(range(len(self.T_diff))), self.T_diff,s=5)
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

            
if __name__ == '__main__':
    #使用heart数据
    def X_1(X):
        one = np.ones(X.shape[0])
        X_ = np.column_stack((X,one))
        return X_
    
    print('-------choose dataset--------\n')
    print('---    1.heart(240,13)    ---\n')
    print('---    2.ijcnn1(49990,21) ---\n')
    print('---    3.svmguide(3089,4) ---\n')
    print('---    4.w3a(4912,300)    ---\n')
    print('---    5.svmguide3(1243,21)--\n')
    print('-------choose dataset--------\n')
    
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
    
    model = RDSVMbyADMM()
    model.fit(X_train,y_train)
    print('test accuracy:',model.get_accuracy(X_test,y_test))
    #'''
 
    
 