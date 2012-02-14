# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

from __future__ import division
import random
import math as m

#TODO:comment
class Neuron:
    '''
    A single neuron
    '''
    (Hidden, Output) = range(2)
    
#TODO:separate hidden/outputs 
    def __init__(self, nbr_input, learning_rate=0.1, momemtum=0.0, ntype=Output):
        '''
        Constructor
        '''
        self.weights = [random.random() for _ in range(nbr_input)]
        self.last_weights = self.weights;
        self.learning_rate = learning_rate
        self.momemtum = momemtum
        self.ntype = ntype
        self.a = 0.
        self.state = 0.
        self.gradient = 1.

    def calc_output(self, inputs):
        a = reduce(lambda x, y:x + y, map(lambda x, y:x * y, inputs, self.weights))
        self.a = a
        self.state = self._sigmoid(a)
        return self.state
    
    def learn(self, inputs, wanted):
        self.calc_output(inputs)
        self.last_weights = self.weights
        
        if self.ntype == Neuron.Output:
#            y = 2 * self._derivated_sigmoid(self.a) * (wanted - self.state)
            print self.state, wanted
            y = (wanted - self.state)
            self.weights = map(lambda wt, xi, wtm: 
                           wt + self.learning_rate * y * xi + self.momemtum * (wt - wtm), self.weights, inputs, self.last_weights)
        
        print self.weights

    def _sigmoid (self, x):
        return 1 / (1 + m.exp(-self.gradient * x))
    #TODO: use & calc derivated
    def _derivated_sigmoid (self, x):
        return (self.gradient / m.pow(1 + m.exp(-self.gradient * x), 2)) * m.exp(-self.gradient * x)


if __name__ == '__main__':
    #AND test
    n = Neuron(2)
    
#TODO:implement single perceptron learning
    for epoch in range(1000):
        n.learn([0, 0], 0)
        n.learn([0, 1], 0)
        n.learn([1, 0], 0)
        n.learn([1, 1], 1)
        
    print n.calc_output([0, 0])
    print n.calc_output([0, 1])
    print n.calc_output([1, 0])
    print n.calc_output([1, 1])
    
