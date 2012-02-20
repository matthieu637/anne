# -*- coding: UTF-8 -*-
'''
Created on 14 fevr. 2012

@author: matthieu637
'''

from digit import Factory as DigitsFactory
from network import MultilayerNetwork
from utils import findMax
from numpy import random

if __name__ == '__main__':

    mode = MultilayerNetwork.R0to1
    digits = [DigitsFactory.digitToMatrix(k, (5, 4), mode) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10, learning_rate=.2, grid=mode)

    #create example
    examples = [{} for _ in range(10)]
    for ex in range(10):
        examples[ex]["inputs"] = digits[ex].ravel().tolist()
        examples[ex]["outputs"] = [mode] * 10
        examples[ex]["outputs"][ex] = 1

    #learning
    print "Start learning..."
    y = []
    for epoch in range(1000): 
        for ex in random.randint(0, 10, 10):
#        for ex in range(10):
            mn.train(examples[ex]["inputs"], examples[ex]["outputs"])
            
    #testing
    for ex in range(10):
        print "inputs : \n", digits[ex]
        print "outputs state : "
        print "\n".join(i.__str__() for i in mn.calc_output(examples[ex]["inputs"]))
        print "recognition : ", findMax(mn.calc_output(digits[ex].ravel().tolist()))
        print 


#    inputs : 
#    [[ 1.  1.  1.  1.]
#     [ 1.  0.  0.  1.]
#     [ 1.  1.  1.  1.]
#     [ 1.  0.  0.  1.]
#     [ 1.  1.  1.  1.]]
#    outputs state : 
#    0.0820195905921
#    0.00412312830418
#    0.122202100904
#    0.00389510539182
#    0.00259115283238
#    0.136033783387
#    0.12416393787
#    0.000254385101939
#    0.345878527463
#    0.140505006205
#    recognition :  8
#        
