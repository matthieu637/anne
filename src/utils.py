# -*- coding: UTF-8 -*-
'''
Created on 14 February 2012

@author: Matthieu Zimmer

Provides some basics functions and constants
'''

from random import random

NB_CORE = 4

def index_max(l):
    '''
    returns the index of the max value in the list l
    '''
    m = 0
    for i in range(1, len(l)):
        if (l[i] > l[m]):
            m = i
    return m

def last_index_max(l):
    '''
    returns the index of the max value in the list l by the end
    '''
    m = len(l) - 1
    ln = len(l)
    for i in range(1, len(l)+1):
        if (l[ln - i] > l[m]):
            m = ln - i
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
            return False
    return True

def multithread_repartition(n, depth=NB_CORE):
    res = [n // depth] * depth
    for i in range(n % depth):
        res[i] += 1
    return res
    
def charge_to_indice(charge):
    buffer = 0
    i = 0
    res = []
    for c in charge:
        res.append((i ,buffer, buffer + c))
        i += 1
        buffer += c
    return res

def print_liste(l, n):
    i = 0
    while (i < len(l)) :
        print("".join(["x" if l[j] < 0.5 else "." for j in range(i, i+n)]))
        i += n
    
if __name__ == '__main__':
    print(multithread_repartition(42, 4))
    print(multithread_repartition(41, 5))
    print(multithread_repartition(41, 1))
    print(multithread_repartition(45, 5))

    print(charge_to_indice(multithread_repartition(42, 4)))
    print(charge_to_indice(multithread_repartition(41, 5)))
    print(charge_to_indice(multithread_repartition(41, 1)))
    print(charge_to_indice(multithread_repartition(45, 5)))
    
    
    print(index_max([1, 0, 0, 1, 0]))
    print(last_index_max([1, 0, 0, 1, 0]))

    print_liste([0.1, 0.6, 0.4, 0.8], 2)
