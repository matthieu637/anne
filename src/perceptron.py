# -*- coding: UTF-8 -*-
'''
Created on 10 February 2012

@author: Matthieu Zimmer

Provides classes to simulate different type of perceptron
'''

from utils import randmm
from math import exp

class Perceptron:
    '''
    a single perceptron with a weight list
    these outputs are reals between -1. and 1.
    '''
    
    #defines neuron type, read __init__ doc
    (HIDDEN, OUTPUT) = range(2)
    
    def __init__(self, nbr_input, learning_rate=0.1, momentum=0.,
                 temperature=1., ntype=OUTPUT, init_w_randomly=True, enable_bias=True):
        '''
        initialize a single perceptron with nbr_input weights
        - _ntype can be Perceptron.HIDDEN or Perceptron.OUTPUT it specifies the layer of neuron 
            ( it is required because train algo is different for each one )
        - temperature : the slope of the sigmoid
        - init_w_randomly : boolean to define the weights randomly between [ -0.25 ; 0.25 ]
            ( if false you should call init_weights_randomly or create your own list )
        - enable_bias : do you want to enable a bias ?
        '''
        self.nbr_input = nbr_input
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature
        self._ntype = ntype
        self._enable_bias = enable_bias
        
        if(init_w_randomly):
            self.init_weights_randomly()
            
        if(ntype != self.HIDDEN and ntype != self.OUTPUT):
            raise Exception("%d : unknown neuron type (_ntype param)" % ntype)
        
        #these fields are here simply to avoid unnecessary computations
        self._weights_updated = True
        self._last_inputs = []
        
    def init_weights_randomly(self, vmin= -0.25, vmax=0.25):
        '''
        initialize the perceptron with random weights between [vmin ; vmax]
        returns None
        '''
        self.weights = [randmm(vmin, vmax) for _ in range(self.nbr_input)]
        
        if(self._enable_bias):
            self.bias = randmm(vmin, vmax)
        self._last_weights = list(self.weights) #_last_weights is used by the momentum algo
        self._last_bias = self.bias
        self._weights_updated = True
        
    def init_weights(self, val):
        '''
        assigns the value val to all the weights of the perceptron
        returns None
        '''
        self.weights = [val for _ in range(self.nbr_input)]
        if(self._enable_bias):
            self.bias = val
            self._last_bias = self.bias
        self._last_weights = list(self.weights) #_last_weights is used by the momentum algo
        
    def calc_output(self, inputs):
        '''
        returns the output _state of the neuron to the data inputs according to the formula:
        $output \leftarrow g(a)=g\left( \sum \limits_{i = 0}^{len(weights)} weights_{i}\times inputs_{i}\right)$
        $with \left\lbrace \begin{array}{lll} w : weights\\e : inputs\\ w_{0} : bias\\ e_{0} : 1 \end{array}\right.$
        (The output is a real in [-1, 1]) 
        '''

        #avoids unnecessary computations
        if not self._weights_updated and self._last_inputs == inputs:
            return self._state
        self._last_inputs = inputs
        self._weights_updated = False

        #calculates the output
        a = 0.
        for i in range(len(self.weights)):
            a += inputs[i] * self.weights[i]
        
        if self._enable_bias:
            a += self.bias
            
        self._a = a
        self._state = self._sigmoid(a)
        return self._state
    
    def calc_error_propagation(self, wanted):
        '''
        calculates the error for gradient descent
        
        In case of hidden neurons, wanted must be :
        $wanted = \sum \limits_{k} w_{k}\times y_{k}$

        $with \left\lbrace \begin{array}{lll} w_{k} : weight\ between\ this\ neuron\ and\ k^{th}\ neuron\ in\ output\ layer\\ y_{k} : return\ of\ this\ function\ for\ the\ k^{th}\ neuron\ in\ output\ layer \end{array}\right.$

        In case of output neurons, wanted is just the output value to match to the neuron
        
        returns the error to propagate
        $\left \lbrace \begin{array}{lll} 2 \times g'(a) \times (wanted - e_{k}) : for\ ouput\ neurons\\g'(a) \times wanted : for\ hidden\ neurons \end{array}\right.$
        '''
        if(self._weights_updated):
            raise Exception("You are trying to calculate the error with an old _state,\
                 please call calc_output(..) first")
        
        if self._ntype == Perceptron.OUTPUT:
            return  self._derivated_sigmoid(self._a) * (wanted - self._state)
        else: # self._ntype == Perceptron.HIDDEN:
            return self._derivated_sigmoid(self._a) * wanted
        
    def update_weights(self, error, inputs):
        '''
        updates all the weights of the neuron according to the formula :
        $w_{j} (t+1) = w_{j}(t) + learning\_rate \times error \times inputs_j + momentum \times [w_{j}(t) - w_{j}(t-1) ]$
        returns None
        '''
        self.calc_output(inputs)
        
        tmp_weights = list(self.weights)

        for j in range(len(self.weights)):
            dw = self.weights[j] - self._last_weights[j] 
            self.weights[j] += self.learning_rate * error * inputs[j] + self.momentum * dw
        
        if self._enable_bias:
            tmp_bias = self.bias 
            self.bias += self.learning_rate * error + self.momentum * (self.bias - self._last_bias)
            self._last_bias = tmp_bias

        
        self._last_weights = tmp_weights
        self._weights_updated = True
        
    def train(self, inputs, output): 
        '''
        only to use with single perceptron ( not with multilayer network )
        '''
        if(self._ntype != Perceptron.OUTPUT):
            raise Exception("Perceptron.train is only for single perceptron")
        
        self.calc_output(inputs)
        self.update_weights(output - self._state, inputs)
    def calc_sum_dw(self):
        s = 0.
        for i in range(len(self.weights)):
            s += abs(self.weights[i] - self._last_weights[i])
        return s
    def _sigmoid (self, x):
        '''
        returns $\frac{ e^{\theta x} - 1}{1 + e^{ \theta x}}$
        returned values are in [-1 ; 1]
        '''
        return (exp(self.temperature * x) - 1) / (1 + exp(self.temperature * x))
    def _derivated_sigmoid (self, x):
        return (2 * self.temperature * exp(self.temperature * x)) / ((exp(x * self.temperature) + 1) ** 2)

class PerceptronR0to1(Perceptron):
    '''
    this class of neuron can be used on a grid [0, 1]
    to do this we simply must redefine the sigmoid function
    '''
    def _sigmoid (self, x):
        '''
        returns $\frac{ 1}{1 + e^{ - \theta x}}$
        returned values are in [0 ; 1]
        '''
        return 1 / (1 + exp(-self.temperature * x))
    def _derivated_sigmoid (self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x)) * self.temperature


class PerceptronN0to1(Perceptron):
    '''
    this class of neuron can be used on a grid {0, 1}
    the train method is also different and the Perceptron learning rule
    '''
    def __init__(self, nbr_input, learning_rate=0.1, momentum=0., random=True):
        '''
        notice that :
        - _enable_bias ( the last weight ) is mandatory and corresponds to the threshold
        - this type of neuron cannot be in hidden layout ( backpropagation don't work on {0,1} ) 
        - temperature has no meaning ( there isn't anymore sigmoid )
        '''
        Perceptron.__init__(self, nbr_input, learning_rate, momentum, 0., Perceptron.OUTPUT, random, True)
    def _sigmoid (self, x):
        '''
        this function is not anymore a sigmoid but a Heaviside function
        '''
        return 1 if x >= 0 else 0
    def _derivated_sigmoid (self, x):
        raise NotImplemented
    def calc_error_propagation(self, wanted):
        '''
        train now follow the Perceptron learning rule
        '''
        return wanted - self._state

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    #AND example on [-1, 1]
    n = Perceptron(2)
    for epoch in range(50):
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
    
    print()
    
    #AND example on [0,1]
    n = PerceptronR0to1(2)
    for epoch in range(200):
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

    print()
    
    #OR example without training
    n = PerceptronN0to1(2, random=False)
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
    
    print()
    
    #OR example on {0, 1} with training
    n = PerceptronN0to1(2)
    for epoch in range(100):
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
