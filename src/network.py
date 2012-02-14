# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637
'''

import neuron as n

class MultilayerNetwork:
    '''
    Describe a 
    '''
    def __init__(self, nbr_input, nbr_hidden, nbr_output, learning_rate=0.1, momemtum=0.9):
        '''
        Constructor
        '''
        self.hiddenNeurons = [n.Neuron(nbr_input, learning_rate, momemtum) for _ in range(nbr_hidden)]
        self.outputNeurons = [n.Neuron(nbr_hidden, learning_rate, momemtum) for _ in range(nbr_output)]
        self.outputs_status = []
        self.stateHiddenNeurons = []
        
    def calc_output(self, inputs):
        #determine the status of hidden neurons
        hidden_status = []
        for i in range(len(self.hiddenNeurons)) :
            hidden_status.append(self.hiddenNeurons[i].calc_output(inputs))
        self.stateHiddenNeurons = hidden_status
        
        #then the output layer
        outputs_status = []
        for i in range(len(self.outputNeurons)) :
            outputs_status.append(self.outputNeurons[i].calc_output(hidden_status))
        self.outputs_status = outputs_status
        return outputs_status
    
    def learn(self, inputs, outputs):
        self.calc_output(inputs)
        
        for i in range(len(self.outputNeurons)) :
            y = self.outputs_status[i] * \
                    (1. - self.outputs_status[i]) * \
                    (outputs[i] - self.outputs_status[i])
            self.outputNeurons[i].learn(y, self.stateHiddenNeurons)
        
        for i in range(len(self.hiddenNeurons)):
            wki = 0.
            for j in range(len(self.outputNeurons)):
                y = self.stateHiddenNeurons[i] * \
                        (1. - self.stateHiddenNeurons[i]) * \
                        (outputs[j] - self.stateHiddenNeurons[i])
                wki += y * self.outputNeurons[j].weights[i]
            self.hiddenNeurons[i].learn(wki * self.stateHiddenNeurons[i] * (1 - self.stateHiddenNeurons[i]), inputs)
            
if __name__ == '__main__':
    #XOR test
    n = MultilayerNetwork(2, 3, 1)
    
    
    for epoch in range(1000):
        n.learn([0, 0], [0])
        n.learn([0, 1], [1])
        n.learn([1, 0], [1])
        n.learn([1, 1], [0])
        
    print n.calc_output([0, 0])
    print n.calc_output([0, 1])
    print n.calc_output([1, 0])
    print n.calc_output([1, 1])
    
