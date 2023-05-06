#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:20:01 2023

@author: damonchang
"""
 
"""
类似sklearn，先输入参数生成一个svm分类器，再通过SVM_training输入数据训练
"""
 
 
import numpy as np
import pickle
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
#import libsvm_to_metrix
import libsvm2array
import time

 
#%%

class SVM(object):
    def __init__(self, C=1, toler=0.001, maxIter=500, kernel_option = ("",0)):
        self.C = C  # 惩罚参数
        self.toler = toler # 迭代的终止条件之一
        self.b = 0  # 阈值
        self.max_iter = maxIter  # 最大迭代次数
        self.kernel_opt = kernel_option # 选用的核函数及其参数
 
    def SVM_training(self, dataSet, labels, ):
        # 1.输入数据集
        # train_x_m, train_y_m = np.mat(train_x), np.mat(train_y)dataSet, labels,
        self.train_x = np.mat(dataSet)  # 训练数据集
        self.train_y = np.mat(labels)   # 测试数据集
        self.train_y = self.train_y.T if np.shape(self.train_y)[0] == 1 else self.train_y # 将其转化为列向量
        self.n_samples = np.shape(dataSet)[0]  # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))  # 拉格朗日乘子（一个全0的列向量）
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))  # 保存E的缓存
        self.kernel_mat = self.calc_kernel(self.train_x, self.kernel_opt)  # 核函数的输出
        # 2.开始训练
        entireSet = True
        alpha_pairs_changed = 0
        iteration = 0
        while iteration<self.max_iter and (alpha_pairs_changed>0 or entireSet):
            print("\t iteration: ",iteration)
            alpha_pairs_changed = 0
 
            if entireSet:   # 对所有样本
                for x in range(self.n_samples):
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            else:   # 对非边界样本
                bound_samples = []
                for i in range(self.n_samples):
                    if self.alphas[i, 0] > 0 and self.alphas[i, 0] < self.C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
 
            if entireSet:
                entireSet = False
            elif alpha_pairs_changed == 0:
                entireSet = True
        return self
 
    def cal_error(self, alpha_index_k):
        """误差值的计算
        :param alpha_index_k(int): 输入的alpha_k的index_k
        :return: error_k(float): alpha_k对应的误差值
        np.multiply(svm.alphas,svm.train_y).T 为一个行向量（αy,αy,αy,αy,...,αy）
        """
        predict_k = float(np.multiply(self.alphas, self.train_y).T * self.kernel_mat[:, alpha_index_k] + self.b)
        error_k = predict_k - float(self.train_y[alpha_index_k])
        return error_k
 
    def select_second_sample_j(self, alpha_index_i, error_i):
        """选择第二个变量
        :param alpha_index_i(float): 第一个变量alpha_i的index_i
        :param error_i(float): E_i
        :return:第二个变量alpha_j的index_j和误差值E_j
        """
        self.error_tmp[alpha_index_i] = [1, error_i] # 用来标记已被优化
        candidate_alpha_list = np.nonzero(self.error_tmp[:, 0].A)[0]  # 因为是列向量，列数[1]都为0，只需记录行数[0]
        max_step,max_step,error_j = 0,0,0
 
        if len(candidate_alpha_list)>1:
            alpha_index_j = -1
            for alpha_index_k in candidate_alpha_list:
                if alpha_index_k == alpha_index_i:
                    continue
                error_k = self.cal_error(alpha_index_k)
                if abs(error_k-error_i)>max_step:
                    max_step = abs(error_k-error_i)
                    alpha_index_j,error_j = alpha_index_k,error_k
        else:   # 随机选择
            alpha_index_j = alpha_index_i
            while alpha_index_j == alpha_index_i:
                alpha_index_j = np.random.randint(0, self.n_samples)
            error_j = self.cal_error(alpha_index_j)
        return alpha_index_j, error_j
 
    def update_error_tmp(self, alpha_index_k):
        """重新计算误差值，并对其标记为已被优化
        :param alpha_index_k: 要计算的变量α
        :return: index为k的alpha新的误差
        """
        error = self.cal_error(alpha_index_k)
        self.error_tmp[alpha_index_k] = [1, error]
 
 
    def choose_and_update(self, alpha_index_i):
        """判断和选择两个alpha进行更新
        :param alpha_index_i(int): 选出的第一个变量的index
        :return:
        """
        error_i = self.cal_error(alpha_index_i) # 计算第一个样本的E_i
        if (self.train_y[alpha_index_i]*error_i<-self.toler) and (self.alphas[alpha_index_i]<self.C) \
                or (self.train_y[alpha_index_i]*error_i>self.toler) and (self.alphas[alpha_index_i]>0):
            # 1.选择第二个变量
            alpha_index_j, error_j = self.select_second_sample_j(alpha_index_i, error_i)
            alpha_i_old = self.alphas[alpha_index_i].copy()
            alpha_j_old = self.alphas[alpha_index_j].copy()
            # 2.计算上下界
            if self.train_y[alpha_index_i] != self.train_y[alpha_index_j]:
                L = max(0, self.alphas[alpha_index_j] - self.alphas[alpha_index_i])
                H = min(self.C, self.C + self.alphas[alpha_index_j] - self.alphas[alpha_index_i])
            else:
                L = max(0, self.alphas[alpha_index_j] + self.alphas[alpha_index_i] - self.C)
                H = min(self.C, self.alphas[alpha_index_j] + self.alphas[alpha_index_i])
            if L == H:
                return 0
            # 3.计算eta
            eta = self.kernel_mat[alpha_index_i, alpha_index_i] + self.kernel_mat[alpha_index_j, alpha_index_j] - 2.0 * self.kernel_mat[alpha_index_i, alpha_index_j]
            if eta <= 0: # 因为这个eta>=0
                return 0
            # 4.更新alpha_j
            self.alphas[alpha_index_j] += self.train_y[alpha_index_j] * (error_i - error_j) / eta
            # 5.根据范围确实最终的j
            if self.alphas[alpha_index_j] > H:
                self.alphas[alpha_index_j] = H
            if self.alphas[alpha_index_j] < L:
                self.alphas[alpha_index_j] = L
 
            # 6.判断是否结束
            if abs(alpha_j_old-self.alphas[alpha_index_j])<0.00001:
                self.update_error_tmp( alpha_index_j)
                return 0
            # 7.更新alpha_i
            self.alphas[alpha_index_i] += self.train_y[alpha_index_i] * self.train_y[alpha_index_j] * (alpha_j_old - self.alphas[alpha_index_j])
            # 8.更新b
            b1 = self.b - error_i - self.train_y[alpha_index_i] * self.kernel_mat[alpha_index_i, alpha_index_i] * (self.alphas[alpha_index_i] - alpha_i_old) \
                 - self.train_y[alpha_index_j] * self.kernel_mat[alpha_index_i, alpha_index_j] * (self.alphas[alpha_index_j] - alpha_j_old)
            b2 = self.b - error_j - self.train_y[alpha_index_i] * self.kernel_mat[alpha_index_i, alpha_index_j] * (self.alphas[alpha_index_i] - alpha_i_old) \
                 - self.train_y[alpha_index_j] * self.kernel_mat[alpha_index_j, alpha_index_j] * (self.alphas[alpha_index_j] - alpha_j_old)
            if 0<self.alphas[alpha_index_i] and self.alphas[alpha_index_i]<self.C:
                self.b = b1
            elif 0<self.alphas[alpha_index_j] and self.alphas[alpha_index_j]<self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            # 9.更新error
            self.update_error_tmp( alpha_index_j)
            self.update_error_tmp(alpha_index_i)
            return 1
        else:
            return 0
 
    def svm_predict(self, test_data_x):
        """对输入的数据预测（预测一个数据）
        :param test_data_x: 要预测的数据（一个）
        :return: 预测值
        """
        kernel_value = self.calc_kernel_value(self.train_x, test_data_x, self.kernel_opt)
        alp = self.alphas
        #print((np.multiply(self.train_y, self.alphas).T).shape,kernel_value.shape)
        predict = np.multiply(self.train_y, self.alphas).T @ kernel_value.reshape(self.train_x.shape[0],1) + self.b
        return predict
 
    def get_prediction(self,test_data):
        '''对样本进行预测（预测多个数据）
        input:  test_data(mat):测试数据
        output: prediction(list):预测所属的类别
        '''
        m = np.shape(test_data)[0]
        prediction = []
        for i in range(m):
            predict = self.svm_predict(test_data[i, :])
            prediction.append(str(np.sign(predict)[0, 0]))
        return prediction
 
    def cal_accuracy(self, test_x, test_y):
        """计算准确率
        :param test_x:
        :param test_y:
        :return:
        """
        #test_x = test_x.reshape(-1,1).T
        n_samples = np.shape(test_x)[0]
        correct = 0.0
        for i in range(n_samples):
            predict = self.svm_predict(test_x[i, :])
            if np.sign(predict) == np.sign(test_y[i]):
                correct += 1
        accuracy = correct / n_samples
        #return (sum(np.where(self.svm_predict(test_x)>=0,1,-1).flatten() == test_y)) / len(test_y)
        return  accuracy
 
    def get_train_accracy(self):
        accuracy = self.cal_accuracy(self.train_x, self.train_y)
        return accuracy
 
    def calc_kernel(self, train_x, kernel_option):
        """计算核函数的矩阵
        :param train_x(matrix): 训练样本的特征值
        :param kernel_option(tuple):  核函数的类型以及参数
        :return: kernel_matrix(matrix):  样本的核函数的值
        """
        m = np.shape(train_x)[0]
        kernel_matrix = np.mat(np.zeros((m,m)))
        for i in range(m):
            kernel_matrix[:,i] = self.calc_kernel_value(train_x, train_x[i,:], kernel_option)
        return kernel_matrix
 
    def calc_kernel_value(self,train_x, train_x_i, kernel_option):
        """样本之间的核函数值
        :param train_x(matrix): 训练样本
        :param train_x_i(matrix):   第i个训练样本 一个行向量
        :param kernel_option(tuple):   核函数的类型以及参数
        :return: kernel_value(matrix):  样本之间的核函数值
        """
        kernel_type = kernel_option[0]
        m = np.shape(train_x)[0]
        kernel_value = np.mat(np.zeros((m,1)))
        if kernel_type == "rbf":  # 高斯核函数
            sigma = kernel_option[1]
            if sigma == 0:
                sigma = 1.0
            for i in range(m):
                diff = train_x[i, :] - train_x_i
                kernel_value[i] = np.exp(diff*diff.T/(-2.0*sigma**2))  # 分子为差的2范数的平方
        elif kernel_type == "polynomial":
            p = kernel_option[1]
            for i in range(m):
                kernel_value[i] = (train_x[i, :]*train_x_i.T + 1)**p
        else:
            kernel_value = train_x@train_x_i.T  # 直接一个m*m矩阵×一个m*1的矩阵
        return kernel_value
 
 
 
    def save_svm_model(self, model_file):
        with open(model_file, "w") as f:
            pickle.dump(self, f)
 
    def load_svm_model(self, model_file):
        with open(model_file, "r") as f:
            svm_model = pickle.load(f)
        return svm_model
 
    def save_prediction(self, result_file, prediction):
        '''保存预测的结果
        input:  result_file(string):结果保存的文件
                prediction(list):预测的结果
        '''
        f = open(result_file, 'w')
        f.write(" ".join(prediction))
        f.close()
 
def load_data(data_file):
    data_set, labels = [], []
    with open(data_file,"r") as f:
        textlist = f.readlines()
        for line in textlist:
            tmp = []
            line = line.strip().split(" ")
            labels.append(float(line[0]))
            i = 1
            for word in line[1:]:
                feature,value = word.split(":")
                while int(feature) != i:
                    tmp.append(float(0))
                    i += 1
                tmp.append(float(value))
                i += 1
            data_set.append(tmp)
 
    return (np.mat(data_set),np.mat(labels).T)
#%% 
 
if __name__ == "__main__":
    #train_x,train_y = load_data("heart_scale")
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
    #x = X_1(x)
    #x_t,y_t = libsvm_to_metrix.libsvm_to_metrix(path2)
    x_t,y_t = libsvm2array.svm_read_problem(path2)
    #x_t = X_1(x_t)
    print('----  done  ----')
    
    X_train,y_train = x,y
    X_test,y_test = x_t,y_t 

    # print(train_y,train_x)
    #svm = SVM(C=0.6,kernel_option=("rbf",0.431029))
    svm = SVM(C=0.6,kernel_option='linear')
    t1 = time.time()
    svm = svm.SVM_training(X_train,y_train,)
    t2 = time.time()
    # print(svm.alphas,svm.b)
    accuracy = svm.get_train_accracy()
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    print('Training time: {} s'.format(t2-t1))
    print('Test accuracy: {}'.format(svm.cal_accuracy(X_test, y_test)))
    
    
    