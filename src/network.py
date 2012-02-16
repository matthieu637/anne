# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637
'''

import neuron as n

class MultilayerNetwork:
    
    #defines grid values, R1to1 means real in {-1, 1}
    (R1to1, R0to1) = range(2)
    
    '''
    Describe a 
    '''
    def __init__(self, nbr_input, nbr_hidden, nbr_output, grid, learning_rate=0.1, momemtum=0., gradient=1.):
        '''
        Constructor
        '''
        self.hiddenNeurons = [n.Neuron(nbr_input, learning_rate, momemtum, gradient, ntype=n.Neuron.Hidden) for _ in range(nbr_hidden)]
        self.outputNeurons = [n.Neuron(nbr_hidden, learning_rate, momemtum, gradient) for _ in range(nbr_output)]
        self.stateOutputNeurons = []
        self.stateHiddenNeurons = []
        
    def calc_output(self, inputs):
        #determine the state of hidden neurons
        stateHidden = []
        for i in range(len(self.hiddenNeurons)) :
            stateHidden.append(self.hiddenNeurons[i].calc_output(inputs))
        self.stateHiddenNeurons = stateHidden
        
        #then the output layer
        stateOutputs = []
        for i in range(len(self.outputNeurons)) :
            stateOutputs.append(self.outputNeurons[i].calc_output(stateHidden))
        self.stateOutputNeurons = stateOutputs
        return stateOutputs
    
    def train(self, inputs, outputs):
        self.calc_output(inputs)
        
        y = []
        for i in range(len(self.outputNeurons)) :
            y.append(self.outputNeurons[i].train(self.stateHiddenNeurons, outputs[i]))
        
        for i in range(len(self.hiddenNeurons)):
            w_sum = 0.
            for j in range(len(self.outputNeurons)) :
                w_sum += self.outputNeurons[j].weights[i] * y[j]
            self.hiddenNeurons[i].train(inputs, w_sum)
            
if __name__ == '__main__':
    #XOR test
    n = MultilayerNetwork(2, 3, 1)
    
    for epoch in range(1000):
        n.train([-1, -1], [-1])
        n.train([-1, 1], [1])
        n.train([1, -1], [1])
        n.train([1, 1], [-1])
        
    print n.calc_output([-1, -1])
    print n.calc_output([-1, 1])
    print n.calc_output([1, -1])
    print n.calc_output([1, 1])
    
