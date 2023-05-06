#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:39:19 2023

@author: damonchang
"""

import numpy as np
#%%
class father(object):
    def __init__(self,a=1,b=2):
        self.a = a
        self.b = b
        self.cv = 'father'
        #child_method = getattr(self, '_test')# 获取子类的out()方法
        #child_method(4) # 执行子类的out()方法
    def _sum(self,k):
        self.a = self.a + 100
        return self.a + k*self.b
    
class son(father):
    def __init__(self,c,d):
        super().__init__()
        #super()._sum(k)
        #father().__init_(self)
        self.c = c
        self.d = d
        self.cv = 'son'
        print('now',self.a)
    def _test(self,l):
        _f = self._sum(3)
        result = self.a + l*self.c + _f
        print('then',self.a)
        return result
#%%
Son = son(5,6)
print(Son._test(8))
f = father()
    