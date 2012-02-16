# -*- coding: UTF-8 -*-
'''
Created on 14 fevr. 2012

@author: matthieu637
'''


from digit import Factory as DigitsFactory
from network import MultilayerNetwork
from utils import findMax

if __name__ == '__main__':

    digits = [DigitsFactory.digitToMatrix(k, (5, 4)) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10)

    #create example
    examples = [{} for _ in range(10)]
    for ex in range(10):
        examples[ex]["inputs"] = digits[ex].ravel().tolist()
        examples[ex]["outputs"] = [-1] * 10
        examples[ex]["outputs"][ex] = 1

    #learning
    print "Start learning..."
    for epoch in range(1000):
        for ex in range(10):
            mn.train(examples[ex]["inputs"], examples[ex]["outputs"])
        
    #testing
    for ex in range(10):
        print "inputs : \n", digits[ex]
        print "outputs state : "
        print "\n".join(i.__str__() for i in mn.calc_output(examples[ex]["inputs"]))
        print "recognition : ", findMax(mn.calc_output(digits[ex].ravel().tolist()))
        print 
