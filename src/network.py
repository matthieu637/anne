# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637
'''

from neuron import Neuron,NeuronR0to1

class MultilayerNetwork:
    '''
    describes a neural network with 2 layers ( hidden and output )
    '''
    
    #defines grid values, R1to1 means Real in [-1, 1], R0to1 means Real in [0,1]
    (R1to1, R0to1) = (-1,0)

    def __init__(self, nbr_input, nbr_hidden, nbr_output, grid=R1to1, learning_rate=0.1,
                  momentum=0., gradient=1., random=True, enableBias=True):
        '''
        builds a neural network with 2 layers
        nbr_input is the number of inputs to the neurons in the hidden layer
        see Neuron.__init__() for more information about other parameters
        '''
        if grid == MultilayerNetwork.R1to1:
            self.hiddenNeurons = \
                [Neuron(nbr_input, learning_rate, momentum, gradient, Neuron.Hidden, random, enableBias) \
                                  for _ in range(nbr_hidden)]
            self.outputNeurons = \
                [Neuron(nbr_hidden, learning_rate, momentum, gradient, Neuron.Output, random, enableBias) \
                                  for _ in range(nbr_output)]
        elif grid == MultilayerNetwork.R0to1:
            self.hiddenNeurons = \
                [NeuronR0to1(nbr_input, learning_rate, momentum, gradient, Neuron.Hidden, random, enableBias) \
                                  for _ in range(nbr_hidden)]
            self.outputNeurons = \
                [NeuronR0to1(nbr_hidden, learning_rate, momentum, gradient, Neuron.Output, random, enableBias) \
                                  for _ in range(nbr_output)]
        self.stateOutputNeurons = []
        self.stateHiddenNeurons = []

    def init_random_weights(self):
        '''
        assigns a random value to all the weights of the network
        '''
        for neuron in self.hiddenNeurons+self.outputNeurons:
            neuron.init_random_weights()
       
    def init_weights(self, val):
        '''
        assigns the value val to all the weights of the network
        '''
        for neuron in self.hiddenNeurons+self.outputNeurons:
            neuron.init_weights(val)
        
    def calc_output(self, inputs):
        '''
        returns the responses list of the output neurons to these data inputs
        '''
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
        '''
        trains the network to associate inputs to outputs ( by using the backpropagation algorithm )
        '''
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
    #XOR test on [-1, 1]
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
    
    #[-0.9424440588256111]
    #[0.9583671093424109]
    #[0.9581018151457293]
    #[-0.9435578546516472]
    
    print
    
    #XOR test on [0, 1]
    n = MultilayerNetwork(2, 3, 1, grid=MultilayerNetwork.R0to1)
    
    for epoch in range(2000):
        n.train([0, 0], [0])
        n.train([0, 1], [1])
        n.train([1, 0], [1])
        n.train([1, 1], [0])
        
    print n.calc_output([0, 0])
    print n.calc_output([0, 1])
    print n.calc_output([1, 0])
    print n.calc_output([1, 1])
    
    #[0.18486650386859885]
    #[0.8124738577513108]
    #[0.7566651974008758]
    #[0.257136216923515]

