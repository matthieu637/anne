# -*- coding: UTF-8 -*-
'''
Created on 10 fevr. 2012

@author: matthieu637
'''

from __future__ import division
from random import random
from math import exp

class Neuron:
    '''
    a single perceptron with a weight list
    '''
    
    #defines neuron type, read __init__ doc
    (Hidden, Output) = range(2)
    
    def __init__(self, nbr_input, learning_rate=0.1, momentum=0.,
                 gradient=1., ntype=Output, random=True, enableBias=True):
        '''
        initialize a single perceptron with nbr_input random weights
        nType can be Neuron.Hidden or Neuron.Output it specifies the layer of neuron 
            ( it is required because train algo is different for each one )
        '''
        self.nbr_input = nbr_input
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.gradient = gradient
        self.ntype = ntype
        self.enableBias = enableBias
        
        if(random):
            self.init_random_weights()
        
        #these fields are here simply to reuse the output during the training ( if it has been calculated )
        self.stateUpdated = False
        self.last_input = []
        
    def init_random_weights(self):
        '''
        initialize the perceptron with random weights between [-0.5 ; 0.5]
        '''
        self.weights = [random() - 0.5 for _ in range(self.nbr_input)]
        
        if(self.enableBias):
            self.bias = random() - 0.5
        self.last_weights = list(self.weights) #last_weights is used by the momentum algo
        self.last_bias = self.bias
        
    def init_weights(self, val):
        '''
        assigns the value val to all the weights of the perceptron
        '''
        self.weights = [val for _ in range(self.nbr_input)]
        if(self.enableBias):
            self.bias = val
            self.last_bias = self.bias
        self.last_weights = list(self.weights) #last_weights is used by the momentum algo
        
    def calc_output(self, inputs):
        '''
        returns the output to the data inputs according to the formula:
        $output \leftarrow g(a)=g\left( \sum \limits_{i} w_{i}\times e_{i}\right)$
        $with \left\lbrace \begin{array}{lll} w : weights\\e : inputs\\ w_{0} : enableBias\\ e_{0} : -1 \end{array}\right.$
        (The output is a real in [-1, 1]) 
        '''
        
        #$a = \sum \limits_{i = 1}^{len(weights)} weights_{i}\times inputs_{i}$
        a = 0.
        for i in range(len(self.weights)):
            a += inputs[i] * self.weights[i]
        
        if self.enableBias:
            a += self.bias
            
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
        $with \left\lbrace \begin{array}{lll} w_{k} : weight\ between\ this\ neuron\ and\ k^{th}\ neuron\ in\ output\ layer\\ y_{k} : return\ of\ this\ function\ for\ the\ k^{th}\ neuron\ in\ output\ layer \end{array}\right.$
        In case of output neurons, wanted is just the value to match for inputs
        
        returns $y :\left \lbrace \begin{array}{lll} 2 \times g'(a) \times (wanted - e_{k}) : for\ ouput\ neurons\\g'(a) \times wanted : for\ hidden\ neurons \end{array}\right.$
        '''
        #recalculates the output if necessary
        if not self.stateUpdated or self.last_input != inputs:
            self.calc_output(inputs)
        self.stateUpdated = False
        
        y = self._calc_y(wanted)
         
        tmp_weights = list(self.weights)
        #update weights
        #$w_{j} (t+1) = w_{j}(t) + learning\_rate \times y \times inputs_j + momentum \times [w_{j}(t) - w_{j}(t-1) ]$
        for j in range(len(self.weights)):
            dw = self.weights[j] - self.last_weights[j]
            self.weights[j] += self.learning_rate * y * inputs[j] + self.momentum * dw
        
        if self.enableBias:
            tmp_bias = self.bias 
            self.bias += self.learning_rate * y + self.momentum * (self.bias - self.last_bias)
            self.last_bias = tmp_bias

        
        self.last_weights = tmp_weights
        
        return y
    def _calc_y(self, wanted):
        if self.ntype == Neuron.Output:
            return  2 * self._derivated_sigmoid(self.a) * (wanted - self.state)
        elif self.ntype == Neuron.Hidden:
            return  self._derivated_sigmoid(self.a) * wanted
        else :
            raise Exception("%d : unknown neuron type (ntype param)" % self.ntype)
    def _sigmoid (self, x):
        '''
        returns $\frac{ e^{\theta x} - 1}{1 + e^{ \theta x}}$
        returned values are in [-1 ; 1]
        '''
        return (exp(self.gradient * x) - 1) / (1 + exp(self.gradient * x))
    def _derivated_sigmoid (self, x):
        return (2 * exp(x)) / ((exp(x) + 1) ** 2)

class NeuronR0to1(Neuron):
    '''
    this class of neuron can be used on a grid [0, 1] ( unlike parent on [-1, 1] )
    to do this we simply must redefine the sigmoid function
    '''
    def _sigmoid (self, x):
        '''
        returns $\frac{ e^{\theta x} - 1}{1 + e^{ \theta x}}$
        returned values are in [0 ; 1]
        '''
        return 1 / (1 + exp(-self.gradient * x))
    def _derivated_sigmoid (self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))


class NeuronN0to1(Neuron):
    '''
    this class of neuron can be used on a grid {0, 1} ( unlike parent on [-1, 1] )
    the train method is also different and use Hebb's rule
    '''
    def __init__(self, nbr_input, learning_rate=0.1, momemtum=0., random=True):
        '''
        notice that :
        - enableBias ( the last weight ) is mandatory and corresponds to the threshold
        - this type of neuron cannot be in hidden layout ( backpropagation don't work on {0,1} ) 
        - gradient has no meaning ( there isn't anymore sigmoid )
        '''
        Neuron.__init__(self, nbr_input, learning_rate, momemtum, Neuron.Output, random, enableBias=True)
    def _sigmoid (self, x):
        '''
        this function is not anymore a sigmoid but a Heaviside function
        '''
        return 1 if x >= 0 else 0
    def _derivated_sigmoid (self, x):
        raise NotImplemented
    def _calc_y(self, wanted):
        '''
        train now follow the Hebb's rule ( with a possible momentum ) 
        '''
        return wanted - self.state

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    #AND example on [-1, 1]
    n = Neuron(2)
    for epoch in range(200):
        n.train([-1, -1], -1)
        n.train([-1, 1], -1)
        n.train([1, -1], -1)
        n.train([1, 1], 1)
        
    print(n.calc_output([-1, -1]))
    print(n.calc_output([-1, 1]))
    print(n.calc_output([1, -1]))
    print(n.calc_output([1, 1]))
    #-0.999440929256
    #-0.876805869893
    #-0.877244822514
    #0.877681922152
    
    print
    
    #AND example on [0,1]
    n = NeuronR0to1(2)
    for epoch in range(500):
        n.train([0, 0], 0)
        n.train([0, 1], 0)
        n.train([1, 0], 0)
        n.train([1, 1], 1)
        
    print(n.calc_output([0, 0]))
    print(n.calc_output([0, 1]))
    print(n.calc_output([1, 0]))
    print(n.calc_output([1, 1]))
    #0.0150476832098
    #0.186800858015
    #0.187850888618
    #0.776676205452

    print
    
    #OR example without training
    n = NeuronN0to1(2, random=False)
    n.weights = [1, 1] 
    n.bias = -1 # corresponds to the threshold
    print(n.calc_output([0, 0]))
    print(n.calc_output([0, 1]))
    print(n.calc_output([1, 0]))
    print(n.calc_output([1, 1]))
    
    #0
    #1
    #1
    #1
    
    print
    
    #OR example on {0, 1} with training
    n = NeuronN0to1(2)
    for epoch in range(500):
        n.train([0, 0], 0)
        n.train([0, 1], 1)
        n.train([1, 0], 1)
        n.train([1, 1], 1)
        
    print(n.calc_output([0, 0]))
    print(n.calc_output([0, 1]))
    print(n.calc_output([1, 0]))
    print(n.calc_output([1, 1]))
    print(n.weights , n.bias)
    
    #0
    #1
    #1
    #1
    #[0.7567742076501507, 0.8121707616454082] 0.687264146396
