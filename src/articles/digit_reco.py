# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637

Article test
'''

from digit import Factory as DigitsFactory
from network import MultilayerNetwork
from utils import findMax, RMS

if __name__ == '__main__':

    digits = [DigitsFactory.digitToMatrix(k, (5, 4)) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10, momemtum=.9)

    #learning
    for epoch in range(1000):
        errs = []
        for ex in range(10):
            example = {}
            example["inputs"] = digits[ex].ravel().tolist()
            example["outputs"] = [-1]*10
            example["outputs"][ex] = 1;

            mn.learn(example["inputs"], example["outputs"])
            err = reduce(lambda x, y:x + y, map(lambda x, y: abs(x - y), \
                        mn.calc_output(example["inputs"]), example["outputs"]))
            errs.append(err/10)
        print max(errs), RMS(errs), RMS(errs)/max(errs), sum(errs)/10
        
    #testing
    for ex in range(10):
        print digits[ex]
        print mn.calc_output(digits[ex].ravel().tolist())
        print findMax(mn.calc_output(digits[ex].ravel().tolist()))