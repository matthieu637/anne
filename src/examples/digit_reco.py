# -*- coding: UTF-8 -*-
'''
Created on 14 fevr. 2012

@author: matthieu637
'''


from digit import Factory as DigitsFactory
from network import MultilayerNetwork
from utils import findMax
from numpy import random
from math import sqrt

import matplotlib.pyplot as plt

if __name__ == '__main__':

    mode = MultilayerNetwork.R0to1
    digits = [DigitsFactory.digitToMatrix(k, (5, 4), mode) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10, momentum=0.9, learning_rate=.1, grid=mode)

    #create example
    examples = [{} for _ in range(10)]
    for ex in range(10):
        examples[ex]["inputs"] = digits[ex].ravel().tolist()
        examples[ex]["outputs"] = [mode] * 10
        examples[ex]["outputs"][ex] = 1

    #learning
    print "Start learning..."
    y=[]
    max_err = 0
    for epoch in range(1000): 
        sum_rms = 0.
#        for ex in random.randint(0,10,10):
        for ex in range(10):
            rms =reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y,2), \
                    mn.calc_output(examples[ex]["inputs"]), examples[ex]["outputs"]))
            #rms = sqrt(rms/10)
#            print rms
            sum_rms += rms
            mn.train(examples[ex]["inputs"], examples[ex]["outputs"])
            
        print sum_rms
        if sum_rms > max_err:
            max_err=sum_rms
            
#        if epoch in [0,25,50,100,200,500,999]:
        y.append(sum_rms)
            
#    y = map(lambda x : x/max_err, y)
    
#    plt.plot([0,25,50,100,200,500,999],y)
    plt.plot(y)
    plt.show()
            
        
    #testing
    for ex in range(10):
        print "inputs : \n", digits[ex]
        print "outputs state : "
        print "\n".join(i.__str__() for i in mn.calc_output(examples[ex]["inputs"]))
        print "recognition : ", findMax(mn.calc_output(digits[ex].ravel().tolist()))
        print 


    #inputs : 
    #[[ 1.  1.  1.  1.]
    # [ 1. -1. -1.  1.]
    # [ 1.  1.  1.  1.]
    # [ 1. -1. -1.  1.]
    # [ 1.  1.  1.  1.]]
    #outputs state : 
    #-0.969166301891
    #-0.994377081186
    #-0.996921577559
    #-0.99355142251
    #-0.999945081245
    #-0.999902126028
    #-0.937741355889
    #-0.993966995205
    #0.944524268975
    #-0.933971110649
    #recognition :  8
    