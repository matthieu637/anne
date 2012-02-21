# -*- coding: UTF-8 -*-
'''
Created on 14 fevr. 2012

@author: matthieu637
'''

from data import DataFile
from network import MultilayerNetwork
from utils import findMax
from numpy import random

if __name__ == '__main__':

    mode = MultilayerNetwork.R1to1
    mn = MultilayerNetwork(20, 5, 10, learning_rate=.1, grid=mode)

    #create example
    examples = DataFile("../data/digit_shape.txt", mode)

    #learning
    print "Start learning..."
    y = []
    for epoch in range(1000): 
        for ex in random.randint(0, 10, 10):
#        for ex in range(10):
            mn.train(examples.inputs[ex], examples.outputs[ex])
            
    #testing
    for ex in range(10):
        print "inputs : \n", examples.inputs[ex]
        print "outputs state : "
        print "\n".join(i.__str__() for i in mn.calc_output(examples.inputs[ex]))
        print "recognition : ", findMax(mn.calc_output(examples.inputs[ex]))
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
