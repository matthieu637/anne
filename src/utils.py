# -*- coding: UTF-8 -*-
'''
Created on 14 fevr. 2012

@author: matthieu637
'''

import numpy as np

def findMax(l):
    '''
    returns the index of the max value in the list l
    '''
    m = 0
    for i in range(1, len(l)):
        if (l[i] > l[m]):
            m = i
    return m
