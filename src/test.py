# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637

Article test
'''

from digit import Factory as DigitsFactory
from network import MultilayerNetwork


def findMax(activationValues):
    m = 0
    for i in range(1, len(activationValues)):
        if (activationValues[i] > activationValues[m]):
            m = i
    return m

if __name__ == '__main__':

    digits = [DigitsFactory.digitToMatrix(k, (5, 4)) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10)

    #learning
    for epoch in range(1000):
        for ex in range(10):
            example = {}
            example["inputs"] = digits[ex].ravel().tolist()
            example["outputs"] = [-1]*10
            example["outputs"][ex] = 1;

            mn.learn(example["inputs"], example["outputs"])
            
    #testing
    for ex in range(10):
        print digits[ex]
        print mn.calc_output(digits[ex].ravel().tolist())
        print findMax(mn.calc_output(digits[ex].ravel().tolist()))