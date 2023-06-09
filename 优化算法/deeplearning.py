#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:05:26 2023

@author: damonchang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
 
#%%
#数据处理
def process_svm_data():
    import numpy as np
    import sys
    sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/数据处理')
    sys.path.append('/Users/damonchang/Desktop/研究生/课题/SVM/测试代码/优化算法')
    import vector2array
    #%%
    path1 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/SVM新闻数据/vector_js.txt'
    path2 = '/Users/damonchang/Desktop/研究生/课题/SVM/数据集/SVM新闻数据/vector_njs.txt'
    x_js,y_js = vector2array.vector2array(path1)
    x_njs,y_njs = vector2array.vector2array(path2)
    X = np.append(x_js,x_njs,axis=0)
    Y = np.append(y_js,y_njs,axis=0)
    
    return list(zip(X,Y))

datadata = process_svm_data()


def simple_gradient():
    # print the gradient of 2x^2 + 5x
    x = 2*torch.ones(2, 2, requires_grad=True)  #Variable(torch.ones(2, 2) * 2, requires_grad=True)
    z = 2 * (x * x) + 5 * x
    # run the backpropagation
    z.backward(torch.ones(2, 2))
    print(x.grad)
 
 
def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
              log_interval=10):
    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    '''
    train_loader = torch.utils.data.DataLoader(
                        dataset= datadata,
                        batch_size= 32,#批处理大小
                        shuffle=True #是否打乱排序
                                            )
    test_loader = train_loader
    '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)
    '''
    
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(128, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 10)
 
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x,-1)
 
    net = Net()
    print(net)
 
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()
 
    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))
 
    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()
 
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 
 
if __name__ == "__main__":
    run_opt = 2
    if run_opt == 1:
        simple_gradient()
    elif run_opt == 2:
        create_nn()
        
        