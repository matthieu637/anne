# -*- coding: UTF-8 -*-
'''
Created on 17 April 2012

@author: Matthieu Zimmer

'''

from utils import randmm
from random import random
from math import exp

class PRenforcement:
    
    def __init__(self, nbr_input, mu_plus=0.1, mu_minus=0.01, 
                 temperature=2., init_w_randomly=True):
        self.nbr_input = nbr_input
        self.mu_plus = mu_plus
        self.mu_minus = mu_minus
        self.temperature = temperature
        
        if(init_w_randomly):
            self.init_weights_randomly()
        
    def init_weights_randomly(self, vmin= -0.25, vmax=0.25):
        self.weights = [randmm(vmin, vmax) for _ in range(self.nbr_input)]
        self.bias = randmm(vmin, vmax)

    def calc_output(self, inputs):
        #calculates the output
        a = 0.
        for i in range(len(self.weights)):
            a += inputs[i] * self.weights[i]
        a += self.bias
        
        self._last_inputs = list(inputs)
        self._ga = self._sigmoid(a)
        self._out = 1 if random() <= self._ga else 0
        return self._out
        
    def train(self, reward): 
        m = (1 * self._ga) + (-1 * (1 - self._ga))
        d = -1 if self._out == 0 else 1
        
        dw = 0.
        if(reward == True):
            dw = self.mu_plus * (d - m)
        else :
            dw = self.mu_minus * (-d - m)
            
        for j in range(len(self.weights)):
            self.weights[j] += dw * self._last_inputs[j]
            
#        print("dw : ", dw, " ga-m : ",self._ga - m, " m : ",  m, " ga", self._ga, reward, " -> ", self.bias)
        
        self.bias += dw * 1

    def _sigmoid (self, x):
        return 1 / (1 + exp(- self.temperature * x))

if __name__ == '__main__':
    #
    #Some examples :
    #
    
    #AND example
    n = PRenforcement(2, 0.2, 0.02, 2., True)
    for epoch in range(500):
        r = random()
        if( r <= 0.25):
            n.train(n.calc_output([0, 0]) == 0)
        elif (r <= 0.5):
            n.train(n.calc_output([0, 1]) == 1)
        elif (r <= 0.75) :
            n.train(n.calc_output([1, 0]) == 1)
        else :
            n.train(n.calc_output([1, 1]) == 1)
        
    print(n.calc_output([0, 0]))
    print(n.calc_output([0, 1]))
    print(n.calc_output([1, 0]))
    print(n.calc_output([1, 1]))
    
    print(n.weights + [n.bias])
  