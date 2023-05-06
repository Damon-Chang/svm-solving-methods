#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:06:58 2023

@author: damonchang
"""

#SSGD+SMO
#SSGD用BB步长
#SSGD使用hinge损失函数？
#损失函数中的惩罚项？doubly regularized？
#
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
import libsvm_to_metrix
import time

#%%

class SSGD:
    def __init__(self,iteration=256,epsilon = 1e-5):
        
        self.n_iter = iteration
        self.epsilon = epsilon
        
        
        
        
    def fit(self,X,Y):
        t1 = time.time()
        print('------ Start training... ------')
        self.X = X
        self.Y = Y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.w = np.zeros(self.n_features)
        self._lambda = self.n_samples / 10 * np.ones(self.n_samples)
        
        self.tau = 1 #[2**6,2**4,1,2**(-2),2**(-4),2**(-6),0]
        
        self.w_s = []
        self.loss = []
        self.i = []
        diff = 10
        t = 0
        while diff >= self.epsilon and t <= self.n_iter:
            t += 1
            #i_t = np.random.randint(self.n_samples,size=5)
            i_t = np.random.randint(self.n_samples)
            self.i.append(i_t)
            l_r = 1 / (t + self.tau)
            #print(self.w,l_r,self._lambda[i_t])
            #print(np.sum(np.array([self.grad_l_i(i) for i in list(i_t)]),axis=0))
            #self.w += l_r * self.w + l_r *\
            #    np.sum(np.array([self.grad_l_i(i)*self._lambda[i] for i in list(i_t)]),axis=0)
            self.w -= l_r * self.w + l_r*self.grad_l_i(i_t)*self._lambda[i_t]
            self.w_s.append(self.w)
            
            diff = np.sum(self.hinge(X, Y))
            self.loss.append(diff)
            
            #K_t = 
        
        self.T = t
        
        t2 = time.time()
        t_all = t2 - t1
        print('训练完成，用时「{}」秒，迭代「{}」次。'.format(t_all,t))
        print('训练准确率：',self.get_accuracy(X,Y))
        
        
        self.show_loss()
        
        return self.w[0:-1],self.w[-1]
    
    def sub_grid(self,i):
        result = []
        for i in range(self.n_samples):
            result += self.grad_l_i(i)
        
        return result
        
    def grad_l_i(self,i):
        if self.Y[i]*self.activation(self.X[i]) < 1:
            #print(len(self.Y[i]),len(self.w))
            return -self.Y[i]*self.X[i] + self._lambda[i]*self.w
        else:
            return self._lambda[i] * self.w
        
    def hinge(self,X,y):
        return sum([max(0,1-y[i]*self.activation(X[i])) for i in range(len(y))])
        
    def activation(self,x):
        #f(x)=wx+b
        return x@self.w
    
    def get_accuracy(self,X,Y):
        return (np.sum(self.predict(X) == Y)) / len(Y)
        
    def predict(self,x):
        return np.where(x@self.w >= 0.0, 1, -1)
    
    def show_loss(self):
        plt.scatter(list(range(len(self.loss))), self.loss,s=5)
        plt.xlabel('Iteration')
        plt.ylabel(r'$ \sum_{i=1}^{n} max({0,y_i f(x_i)}) $')
        plt.title('LOSS')
        plt.show()
    
    def K(self):
        K = []
        for j in range(self.n_samples):
            k = 0
            for t_ in range(self.T):
                if self.i[t_] == j and max(0,1-self.Y[j]*self.w_s[t_]@self.X[j]) > 0:
                    k += 1
            K.append(k)
        return np.array(K)
#%%
'''
path = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/heart/heart.txt'
x,y = libsvm_to_metrix.libsvm_to_metrix(path)
cut = round(0.8*x.shape[0])
def X_1(X):
    one = np.ones(X.shape[0])
    X_ = np.column_stack((X,one))
    return X_
x = X_1(x)
X_train,y_train,X_test,y_test = x[:cut],y[:cut],x[cut:],y[cut:]

model0 = SSGD()
model0.fit(X_train, y_train)
w0,b0 = model0.w[0:-1],model0.w[-1]
#%%
k = model0.K()
alpha0 = (1./(model0.T + model0.tau)) * model0._lambda*k
#alpha0 = (1./((model0.T)*model0.n_samples)) * model0._lambda*k
'''
#%%
class SMOStruct:
    """ 按照John Platt的论文构造SMO的数据结构"""
    def __init__(self, X, y, C, kernel, alphas, b, errors, user_linear_optim):
        self.X = X              # 训练样本
        self.y = y              # 类别 label
        self.C = C              # regularization parameter  正则化常量，用于调整（过）拟合的程度
        self.kernel = kernel    # kernel function   核函数，实现了两个核函数，线性和高斯（RBF）
        self.alphas = alphas    # lagrange multiplier 拉格朗日乘子，与样本一一相对
        self.b = b              # scalar bias term 标量，偏移量
        self.errors = errors    # error cache  用于存储alpha值实际与预测值得差值，与样本数量一一相对
        
        self.m, self.n = np.shape(self.X)    # store size(m) of training set and the number of features(n) for each example  
                                             #训练样本的个数和每个样本的features数量

        self.user_linear_optim = user_linear_optim    # 判断模型是否使用线性核函数
        self.w = np.zeros(self.n)     # 初始化权重w的值，主要用于线性核函数
        #self.b = 0               


   

def linear_kernel(x,y,b=1):
    #线性核函数
    """ returns the linear combination of arrays 'x' and 'y' with
    the optional bias term 'b' (set to 1 by default). """
    result = x @ y.T + b
    return result # Note the @ operator for matrix multiplications


def gaussian_kernel(x,y, sigma=1):
    #高斯核函数
    """Returns the gaussian similarity of arrays 'x' and 'y' with
    kernel width paramenter 'sigma' (set to 1 by default)"""

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(-(np.linalg.norm(x-y,2))**2/(2*sigma**2))
    elif(np.ndim(x)>1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y)>1):
        result = np.exp(-(np.linalg.norm(x-y, 2, axis=1)**2)/(2*sigma**2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1 :
        result = np.exp(-(np.linalg.norm(x[:, np.newaxis]- y[np.newaxis, :], 2, axis = 2) ** 2)/(2*sigma**2))
    return result


# 判别函数一，用于单一样本
def decision_function_output(model,i):
    if model.user_linear_optim:
        #Equation (J1)
        #return float(np.dot(model.w.T, model.X[i])) - model.b
        return float(model.w.T @ model.X[i]) - model.b
    else:
        #Equation (J10)
        return np.sum([model.alphas[j] * model.y[j] * model.kernel(model.X[j], model.X[i]) for j in range(model.m)]) - model.b


# 判别函数二，用于多个样本
def decision_function(alphas, target, kernel, X_train, x_test, b):
    """ Applies the SVM decision function to the input feature vectors in 'x_test'.
    """
    result = (alphas * target) @ kernel(X_train, x_test) - b   # *，@ 两个Operators的区别?
    
    return result

def plot_decision_boundary(model, ax, resolution = 100, colors=('b','k','r'), levels = (-1, 0, 1)):
    """
    绘出分割平面及支持平面，
    用的是等高线的方法，怀疑绘制的分割平面和支持平面的准确性
 
    """

    #Generate coordinate grid of shape [resolution x resolution]
    #and evalute the model over the entire space
    xrange = np.linspace(model.X[:,0].min(), model.X[:, 0].max(), resolution)
    yrange = np.linspace(model.X[:,1].min(), model.X[:, 1].max(), resolution)
    grid = [[decision_function(model.alphas,model.y, model.kernel, model.X,
                               np.array([xr,yr]), model.b) for xr in xrange] for yr in yrange]
   
    grid = np.array(grid).reshape(len(xrange), len(yrange))


    #Plot decision contours using grid and
    #make a scatter plot of training data
    ax.contour(xrange, yrange, grid, levels=levels, linewidths = (1,1,1),
               linestyles = ('--', '-', '--'), colors=colors)
    ax.scatter(model.X[:,0], model.X[:, 1],
               c=model.y, cmap = plt.cm.viridis, lw=0, alpha =0.25)

    #Plot support vectors (non-zero alphas)
    #as circled points (linewidth >0)
    mask = np.round(model.alphas, decimals = 2) !=0.0
    ax.scatter(model.X[mask,0], model.X[mask,1],
               c=model.y[mask], cmap=plt.cm.viridis, lw=1, edgecolors='k')

    return grid, ax

# 选择了alpha2, alpha1后开始第一步优化，然后迭代， “第二层循环，内循环”
# 主要的优化步骤在这里发生
def take_step(i1, i2, model):
   
    #skip if chosen alphas are the same
    if i1 == i2:
        return 0, model
    # a1, a2 的原值，old value，优化在于产生优化后的值，新值 new value
    # 如下的alph1,2, y1,y2,E1, E2, s 都是论文中出现的变量，含义与论文一致
    alph1 = model.alphas[i1]
    alph2 = model.alphas[i2]
   
    y1 = model.y[i1]
    y2 = model.y[i2]

    E1 = get_error(model, i1)
    E2 = get_error(model, i2)
    s = y1 * y2

    # 计算alpha的边界，L, H
    # compute L & H, the bounds on new possible alpha values
    if(y1 != y2):   
        #y1,y2 异号，使用 Equation (J13)
        L = max(0, alph2 - alph1)
        H = min(model.C, model.C + alph2 - alph1)
    elif (y1 == y2):
        #y1,y2 同号，使用 Equation (J14)
        L = max(0, alph1+alph2 - model.C)
        H = min(model.C, alph1 + alph2)
    if (L==H):
        return 0, model

    #分别计算啊样本1, 2对应的核函数组合，目的在于计算eta
    #也就是求一阶导数后的值，目的在于计算a2new
    k11 = model.kernel(model.X[i1], model.X[i1])
    k12 = model.kernel(model.X[i1], model.X[i2])
    k22 = model.kernel(model.X[i2], model.X[i2])
    #计算 eta，equation (J15)
    eta = k11 + k22 - 2*k12
    
    #如论文中所述，分两种情况根据eta为正positive 还是为负或0来计算计算a2 new
    
    if(eta>0): 
        #equation (J16) 计算alpha2
        a2 = alph2 + y2 * (E1 - E2)/eta
        #clip a2 based on bounds L & H
        #把a2夹到限定区间 equation （J17）
        if L < a2 < H:
            a2 = a2
        elif (a2 <= L):
            a2 = L
        elif (a2 >= H):
            a2 = H
    #如果eta不为正（为负或0）
    #if eta is non-positive, move new a2 to bound with greater objective function value
    else:
        # Equation （J19）
        # 在特殊情况下，eta可能不为正not be positive
        f1 = y1*(E1 + model.b) - alph1*k11 - s*alph2*k12
        f2 = y2 * (E2 + model.b) - s* alph1 * k12 - alph2 * k22

        L1 = alph1 + s*(alph2 - L)
        H1 = alph1 + s*(alph2 - H)

        Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 \
               + 0.5 * (L**2) * k22 + s * L * L1 * k12
               
        Hobj = H1 * f1 + H * f2 + 0.5 * (H1**2) * k11 \
               + 0.5 * (H**2) * k22 + s * H * H1 * k12
               
        if Lobj < Hobj - eps:
            a2 = L
        elif Lobj > Hobj + eps:
            a2 = H
        else:
            a2 = alph2

    #当new a2 千万分之一接近C或0是，就让它等于C或0
    if a2 <1e-8:
        a2 = 0.0
    elif a2 > (model.C - 1e-8):
        a2 = model.C
    #超过容差仍不能优化时，跳过
    #If examples can't be optimized within epsilon(eps), skip this pair
    if (np.abs(a2 - alph2) < eps * (a2 + alph2 + eps)):
        return 0, model

    #根据新 a2计算 新 a1 Equation(J18)
    a1 = alph1 + s * (alph2 - a2)

    #更新 bias b的值 Equation (J20)
    b1 = E1 + y1*(a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + model.b
    #equation (J21)
    b2 = E2 + y1*(a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + model.b

    # Set new threshoold based on if a1 or a2 is bound by L and/or H
    if 0 < a1 and a1 < C:
        b_new =b1
    elif 0 < a2 and a2 < C:
        b_new = b2
    #Average thresholds if both are bound
    else:
        b_new = (b1 + b2) * 0.5

    #update model threshold
    model.b = b_new

    # 当所训练模型为线性核函数时
    #Equation (J22) 计算w的值
    if model.user_linear_optim:
         model.w = model.w + y1 * (a1 - alph1)*model.X[i1] + y2 * (a2 - alph2) * model.X[i2]
    #在alphas矩阵中分别更新a1, a2的值
    #Update model object with new alphas & threshold
    model.alphas[i1] = a1
    model.alphas[i2] = a2

    #优化完了，更新差值矩阵的对应值
    #同时更新差值矩阵其它值
    model.errors[i1] = 0
    model.errors[i2] = 0
    #更新差值 Equation (12)
    for i in range(model.m):
        if 0 < model.alphas[i] < model.C:
            model.errors[i] += y1*(a1 - alph1)*model.kernel(model.X[i1],model.X[i]) + \
                            y2*(a2 - alph2)*model.kernel(model.X[i2], model.X[i]) + model.b - b_new

    return 1, model


def get_error(model, i1):
    if 0< model.alphas[i1] <model.C:
        return model.errors[i1]
    else:
        return decision_function_output(model,i1) - model.y[i1]

def get_w(alphas, dataset, labels):
    ''' 通过已知数据点和拉格朗日乘子获得分割超平面参数w
    '''
    #alphas, dataset, labels = np.array(alphas), np.array(dataset), np.array(labels)
    #yx = labels.reshape(1, -1).T*np.array([1, 1])*dataset
    #w = np.dot(yx.T, alphas)
    w = (np.array(alphas).flatten()*labels)@dataset
    return w

def get_accuracy(X,Y,w,b):
    return (sum(predict(X,w,b) == Y)) / len(Y)
    
def predict(x,w,b):
    return np.where(x@w - b >= 0.0, 1, -1)

def examine_example(i2, model):

    y2 = model.y[i2]
    alph2 = model.alphas[i2]
    E2 = get_error(model, i2)
    r2 = E2 * y2

    #重点：这一段的重点在于确定 alpha1, 也就是old a1，并送到take_step去analytically 优化
    # 下面条件之一满足，进入if开始找第二个alpha，送到take_step进行优化
    if ((r2 < -tol and alph2 < model.C) or (r2 > tol and alph2 > 0)):
        if len(model.alphas[(model.alphas != 0) & (model.alphas != model.C)]) > 1:
            #选择Ei矩阵中差值最大的先进性优化
            # 要想|E1-E2|最大，只需要在E2为正时，选择最小的Ei作为E1
            # 在E2为负时选择最大的Ei作为E1
            if model.errors[i2] > 0:
                i1 = np.argmin(model.errors)
            elif model.errors[i2] <= 0:
                i1 = np.argmax(model.errors)

            step_result,model = take_step(i1,i2, model)
            if step_result:
                return 1, model

        # 循环所有非0 非C alphas值进行优化，随机选择起始点
        for i1 in np.roll(np.where((model.alphas != 0) & (model.alphas != model.C))[0],
                          np.random.choice(np.arange(model.m))):
            step_result, model = take_step(i1, i2, model)
            if step_result:
                return 1, model
        
        #a2确定的情况下，如何选择a1? 循环所有(m-1) alphas, 随机选择起始点
        for i1 in np.roll(np.arange(model.m), np.random.choice(np.arange(model.m))):
            #print("what is the first i1",i1)
            step_result, model = take_step(i1, i2, model)
           
            if step_result:
                return 1, model
    #先看最上面的if语句，如果if条件不满足，说明KKT条件已满足，找其它样本进行优化，则执行下面这句，退出
    return 0, model


def fit(model):
   
    numChanged = 0
    examineAll = 1

    #loop num record
    #计数器，记录优化时的循环次数
    loopnum = 0
    loopnum1 = 0
    loopnum2 = 0

    # 当numChanged = 0 and examineAll = 0时 循环退出
    # 实际是顺序地执行完所有的样本，也就是第一个if中的循环，
    # 并且 else中的循环没有可优化的alpha，目标函数收敛了： 在容差之内，并且满足KKT条件
    # 则循环退出，如果执行3000次循环仍未收敛，也退出
    # 重点：这段的重点在于确定 alpha2，也就是old a 2, 或者说alpha2的下标，old a2和old a1都是heuristically 选择
    while(numChanged > 0) or (examineAll): 
        numChanged = 0
        if loopnum == 2000:
           break
        loopnum = loopnum + 1
        if examineAll:
            loopnum1 = loopnum1 + 1 # 记录顺序一个一个选择alpha2时的循环次数
            # # 从0,1,2,3,...,m顺序选择a2的，送给examine_example选择alpha1，总共m(m-1)种选法
            for i in range(model.alphas.shape[0]): 
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        else:  #上面if里m(m-1)执行完的后执行 
            loopnum2 = loopnum2 + 1
            # loop over examples where alphas are not already at their limits
            for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                examine_result, model = examine_example(i, model)
                numChanged += examine_result
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
    print("loopnum012",loopnum,":", loopnum1,":", loopnum2)   
    return model

# can be replaced as per different model u want to show


#make_blobs需要解释一下

# 生成测试数据，训练样本
#X_train, y = make_blobs(n_samples = 1000, centers =2, n_features=2, random_state = 2)
#%%
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
    
    x,y = libsvm_to_metrix.libsvm_to_metrix(path)
    cut = round(0.8*x.shape[0])
    
    X_train,y_train,X_test,y_test = x[:cut],y[:cut],x[cut:],y[cut:]
    
    model0 = SSGD()
    w0,b0 = model0.fit(X_train, y_train)
    
    k = model0.K()
    alpha0 = (1./(model0.T + model0.tau)) * model0._lambda*k
#%%
    #alphas, b = platt_smo(X_train,y_train,0.8,40)
    # set model parameters and initial values
    C = 1.3
    m = len(X_train)
    
    initial_alphas = np.zeros(m)
    initial_b = 0.
    
    #initial_alphas = alpha0
    #initial_b = - b0

    #set tolerances
    tol = 0.01 # error tolerance
    eps = 0.01 # alpha tolerance

    #Instaantiate model

    model = SMOStruct(X_train, y_train, C, linear_kernel, initial_alphas, initial_b, np.zeros(m),user_linear_optim=True)
    #model = SMOStruct(X_train, y_train, C, gaussian_kernel, initial_alphas, initial_b, np.zeros(m),user_linear_optim=False)

    #print("model created ...")
    #initialize error cache

    initial_error = decision_function(model.alphas, model.y, model.kernel, model.X, model.X, model.b) - model.y
    model.errors = initial_error
    #np.random.seed(0)
    
    print("starting to fit...")
    #开始训练
    t3 = time.time()
    output = fit(model)
    t4 = time.time()
    
    # stop here
#%%
    w = get_w(model.alphas,X_train,y_train)
    b = model.b
    
    print('SMO训练用时: {}'.format(t4-t3))
    print('train accuracy: ',get_accuracy(X_train, y_train, w, b))
    print('test accuracy: ',get_accuracy(X_test, y_test, w, b))


