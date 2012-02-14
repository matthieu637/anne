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
    
    def __init__(self, nbr_input, learning_rate=0.1, momemtum=0., ntype=Output):
        '''
        Constructor
        '''
        self.weights = [random.random() for _ in range(nbr_input+1)]
        self.last_weights = self.weights;
        self.learning_rate = learning_rate
        self.momemtum = momemtum
        self.ntype = ntype
        self.a = 0.
        self.state = 0.
        self.gradient = 1.

    def calc_output(self, inputs):
        if len(inputs) != len(self.weights):
            inputs.append(-1)
        a = reduce(lambda x, y:x + y, map(lambda x, y:x * y, inputs, self.weights))
        self.a = a
        self.state = self._sigmoid(a)
        return self.state
    
    def learn(self, inputs, wanted):
        self.calc_output(inputs)
        self.last_weights = self.weights
        
        if self.ntype == Neuron.Output:
            err = wanted - self.state
            y = 2 * self.learning_rate * self._derivated_sigmoid(self.a) * err
            self.weights = map(lambda wt, xi, wtm: 
                           wt +  y * xi + self.momemtum * (wt - wtm), 
                           self.weights, inputs, self.last_weights)
        elif self.ntype == Neuron.Hidden:
            #TODO:learning hidden neurons
            pass

    def _sigmoid (self, x):
        return (m.exp(-self.gradient * x) - 1) / (1 + m.exp(-self.gradient * x))
    #TODO: calc derivated
    def _derivated_sigmoid (self, x):
        return derivative(self._sigmoid)(x)

def derivative(f, epsilon=1e-6):
    eps = epsilon/2
    def g(x):
        return (f(x+eps) - f(x-eps)) / epsilon
    return g

if __name__ == '__main__':
    #AND example
    n = Neuron(2)
    for epoch in range(200):
        n.learn([-1,-1], -1)
        n.learn([-1,1], -1)
        n.learn([1,-1], -1)
        n.learn([1,1], 1)
        
    print n.calc_output([-1,-1])
    print n.calc_output([-1,1])
    print n.calc_output([1,-1])
    print n.calc_output([1,1])
    
    
    
