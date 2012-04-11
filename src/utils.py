# -*- coding: UTF-8 -*-
'''
Created on 14 February 2012

@author: Matthieu Zimmer

Provides some basics functions
'''

from random import random

def index_max(l):
    '''
    returns the index of the max value in the list l
    '''
    m = 0
    for i in range(1, len(l)):
        if (l[i] > l[m]):
            m = i
    return m

def index_min(l):
    '''
    returns the index of the max value in the list l
    '''
    m = 0
    for i in range(1, len(l)):
        if (l[i] < l[m]):
            m = i
    return m

def index_max_nth(l, n):
    cl = list(l)
    vmin = min(cl) - 1
#    - 1
    for _ in range(n):
        imax = index_max(cl)
        cl[imax] = vmin
#        print(i, cl)
    return index_max(cl)

def randmm(vmin, vmax):
    '''
    returns a float between [vmin, vmax[
    '''
    return vmin + random() * abs(vmax - vmin)

def compare(list_model, list_test, threshold=0.5):
    for i in range(len(list_model)):
        if(list_model[i] == 0 and list_test[i] > threshold) or \
            (list_model[i] == 1 and list_test[i] < threshold):
            return 0
    return 1

def compare_f(list_model, list_test, threshold=0.05):
    for i in range(len(list_model)):
        if(abs(list_model[i] - list_test[i]) > threshold):
            return 0
    return 1

