# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637

Article test
'''

from digit import Factory as DigitsFactory
from network import MultilayerNetwork

if __name__ == '__main__':

    digits = [DigitsFactory.digitToMatrix(k, (5, 4)) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10)

    #learning
    for epoch in range(20):
        for ex in range(10):
            example = {}
            example["inputs"] = digits[ex].ravel().tolist()
            example["outputs"] = [0]*10
            example["outputs"][ex] = 1;
            
            mn.learn(example["inputs"], example["outputs"]);
            
    #testing
    for ex in range(10):
        print mn.calc_output(digits[ex].ravel().tolist())