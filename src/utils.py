# -*- coding: UTF-8 -*-
'''
Created on 14 fevr. 2012

@author: matthieu637
'''

import numpy as np

def findMax(activationValues):
    m = 0
    for i in range(1, len(activationValues)):
        if (activationValues[i] > activationValues[m]):
            m = i
    return m


def RMS(values):
    '''
    Root Mean square
    '''
    a = np.array(values)
    return np.sqrt(np.mean(a**2))