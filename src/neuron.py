# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

from __future__ import division
import random
import math as n

#TODO:comment
class Neuron:
    '''
    A single neuron
    '''
#TODO:separate hidden/outputs 
    def __init__(self, nbr_input, learning_rate = 0.1, momemtum = 0.9):
        '''
        Constructor
        '''
        self.weights = [random.random() for _ in range(nbr_input)]
        self.last_weights = self.weights;
        self.learning_rate = learning_rate
        self.momemtum = momemtum

    def calc_output(self, inputs):
        a = reduce( lambda x,y:x+y,  map( lambda x,y:x*y, inputs, self.weights) )
        return self._sigmoid(a)
    
    def learn(self, y, x):
        self.last_weights = self.weights
        self.weights = map(lambda wt, xi, wtm: 
                           wt + self.learning_rate * y * xi + self.momemtum*(wt - wtm), self.weights, x, self.last_weights)

    def _sigmoid (self, a):
        return (n.exp(a*0.9) - 1 )/(n.exp(a*0.9) + 1 )
    #TODO: use & calc derivated
    def _derivated_sigmoid (self, a):
        return 1/(1 + n.exp(-a))


if __name__ == '__main__':
    #AND test
    n = Neuron(2)
    
#TODO:implement single perceptron learning
#    for epoch in range(1000):
#        n.learn([0,0], [0])
#        n.learn([0,1], [0])
#        n.learn([1,0], [0])
#        n.learn([1,1], [1])
        
    print n.calc_output([0,0])
    print n.calc_output([0,1])
    print n.calc_output([1,0])
    print n.calc_output([1,1])
    