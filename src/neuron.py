# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

from __future__ import division
from random import random
from math import exp, pow

class Neuron:
    '''
    a single perceptron with a weight list
    '''
    
    #defines neuron type, read __init__ doc
    (Hidden, Output) = range(2)
    
    def __init__(self, nbr_input, learning_rate=0.1, momemtum=0., gradient=1., ntype=Output):
        '''
        initialize a single perceptron with nbr_input random weights
        nType can be Neuron.Hidden or Neuron.Output it specifies the layer of neuron 
            ( it is required because train algo is different for each one )
        '''
        self.learning_rate = learning_rate
        self.momemtum = momemtum
        self.gradient = gradient
        self.ntype = ntype
        
        self.weights = [random() for _ in range(nbr_input + 1)] #+1 for bias node
        self.last_weights = self.weights #last_weights is used by the momentum algo
        
        #these fields are used simply to reuse the output during the training ( if it has been calculated )
        self.a = 0.
        self.state = 0.
        self.stateUpdated = False
        self.last_input = []

    def calc_output(self, inputs):
        '''
        returns the output to the data inputs according to the formula:
        $output \leftarrow g(a)=g\left( \sum \limits_{i} w_{i}\times e_{i}\right)$
        $with \left\lbrace \begin{array}{lll} w : weights\\e : inputs\\ w_{0} : bias\\ e_{0} : -1 \end{array}\right.$
        (The output is a real in [-1, 1]) 
        '''
        # add -1 to inputs for bias node
        if len(inputs) == len(self.weights) - 1:
            inputs.append(-1)
            
        #$a = \sum \limits_{i = 0}^{len(weights)} weights_{i}\times inputs_{i}$
        a = reduce(lambda x, y:x + y, map(lambda x, y:x * y, inputs, self.weights)) 
        
        self.a = a
        self.state = self._sigmoid(a)
        self.stateUpdated = True
        self.last_input = inputs
        return self.state
    
    def train(self, inputs, wanted):
        '''
        train the neuron by using the backpropagation algorithm
        inputs : list of values to match with wanted

        In case of hidden neurons, wanted must be :
        $wanted = \sum \limits_{k} w_{k}\times y_{k}$
        $with \left\lbrace \begin{array}{lll} w_{k} : weight\ between\ this\ neuron\ and\ k^{th}\ neuron\ in\ output\ layer\\ y_{k} : return\ of\ this\ function\ for\ the\ k^{th}\ neuron\ on\ output\ layer \end{array}\right.$
        In case of output neurons, wanted is just the value to match for inputs
        
        return $y :\left \lbrace \begin{array}{lll} 2 \times g'(a) \times (wanted - e_{k}) : for\ ouput\ neurons\\g'(a) \times wanted : for\ hidden\ neurons \end{array}\right.$
        '''
        if not self.stateUpdated or self.last_input != inputs:
            self.calc_output(inputs)
        self.stateUpdated = False
        self.last_weights = self.weights
        
        y = 0.
        if self.ntype == Neuron.Output:
            y = 2 * self._derivated_sigmoid(self.a) * (wanted - self.state)
        elif self.ntype == Neuron.Hidden:
            y = self._derivated_sigmoid(self.a) * wanted
        else :
            raise Exception("%d : unknown neuron type (ntype param)" % self.ntype)
         
        #update weights
        #$w_{j} (t+1) = w_{j}(t) + learning\_rate \times y \times inputs_j + momemtum \times [w_{j}(t) - w_{j}(t-1) ]$
        self.weights = map(lambda wt, xi, wtm: 
               wt + self.learning_rate * y * xi + self.momemtum * (wt - wtm),
               self.weights, inputs, self.last_weights)
        return y

    def _sigmoid (self, x):
        '''
        return $\frac{ e^{\theta x} - 1}{1 + e^{ \theta x}}$
        returned values are in [-1 ; 1]
        '''
        return (exp(self.gradient * x) - 1) / (1 + exp(self.gradient * x))
    def _derivated_sigmoid (self, x):
        return (2 * exp(x)) / (pow(exp(x) + 1, 2))


class Neuron0to1(Neuron):
    '''
    this class of neuron can be used on a grid {0, 1} ( unlike parent on {-1, 1} )
    to do this we simply must redefine the sigmoid function
    '''
    def _sigmoid (self, x):
        '''
        return $\frac{ e^{\theta x} - 1}{1 + e^{ \theta x}}$
        returned values are in [0 ; 1]
        '''
        return 1 / (1 + exp(-self.gradient * x))
    def _derivated_sigmoid (self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))


if __name__ == '__main__':
    #AND example on {-1, 1}
    n = Neuron(2)
    for epoch in range(200):
        n.train([-1, -1], -1)
        n.train([-1, 1], -1)
        n.train([1, -1], -1)
        n.train([1, 1], 1)
        
    print n.calc_output([-1, -1])
    print n.calc_output([-1, 1])
    print n.calc_output([1, -1])
    print n.calc_output([1, 1])
    
    print
    
    #AND example on {0,1}
    n = Neuron0to1(2)
    for epoch in range(200):
        n.train([0, 0], 0)
        n.train([0, 1], 0)
        n.train([1, 0], 0)
        n.train([1, 1], 1)
        
    print n.calc_output([0, 0])
    print n.calc_output([0, 1])
    print n.calc_output([1, 0])
    print n.calc_output([1, 1])
