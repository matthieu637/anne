# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

from __future__ import division
import random
import math as m

class Neuron:
    '''
    a single perceptron
    '''
    (Hidden, Output) = range(2)
    
    def __init__(self, nbr_input, learning_rate=0.1, momemtum=0., ntype=Output):
        '''
        Constructor
        '''
        self.weights = [random.random() for _ in range(nbr_input + 1)] #+1 for bias node
        self.last_weights = self.weights
        self.learning_rate = learning_rate
        self.momemtum = momemtum
        self.ntype = ntype
        self.a = 0.
        self.state = 0.
        self.stateUpdated = False
        self.gradient = 1.

    def calc_output(self, inputs):
        if len(inputs) != len(self.weights):
            inputs.append(-1)
        a = reduce(lambda x, y:x + y, map(lambda x, y:x * y, inputs, self.weights))
        self.a = a
        self.state = self._sigmoid(a)
        self.stateUpdated = True
        return self.state
    
    def learn(self, inputs, wanted=0., w_sum=0.):
        if not self.stateUpdated:
            self.calc_output(inputs)
        self.stateUpdated = False
        self.last_weights = self.weights
        
        y = 0.
        if self.ntype == Neuron.Output:
            err = wanted - self.state
            y = 2 * self._derivated_sigmoid(self.a) * err
        elif self.ntype == Neuron.Hidden:
            y = self._derivated_sigmoid(self.a) * w_sum
        
        self.weights = map(lambda wt, xi, wtm: 
               wt + self.learning_rate * y * xi + self.momemtum * (wt - wtm),
               self.weights, inputs, self.last_weights)
        return y

    def _sigmoid (self, x):
        return (m.exp(self.gradient * x) - 1) / (1 + m.exp(self.gradient * x))
    def _derivated_sigmoid (self, x):
        return (2 * m.exp(x)) / (m.pow(m.exp(x) + 1, 2))

class Neuron0to1(Neuron):
    def _sigmoid (self, x):
        return 1 / (1 + m.exp(-self.gradient * x))
    def _derivated_sigmoid (self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

if __name__ == '__main__':
    #AND example on {-1, 1}
    n = Neuron(2)
    for epoch in range(200):
        n.learn([-1, -1], -1)
        n.learn([-1, 1], -1)
        n.learn([1, -1], -1)
        n.learn([1, 1], 1)
        
    print n.calc_output([-1, -1])
    print n.calc_output([-1, 1])
    print n.calc_output([1, -1])
    print n.calc_output([1, 1])
    
    print
    
    #AND example on {0,1}
    n = Neuron0to1(2)
    for epoch in range(200):
        n.learn([0, 0], 0)
        n.learn([0, 1], 0)
        n.learn([1, 0], 0)
        n.learn([1, 1], 1)
        
    print n.calc_output([0, 0])
    print n.calc_output([0, 1])
    print n.calc_output([1, 0])
    print n.calc_output([1, 1])
