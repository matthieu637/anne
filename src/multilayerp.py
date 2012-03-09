# -*- coding: UTF-8 -*-
'''
Created on 13 February 2012

@author: Matthieu Zimmer
'''

from math import sqrt
from perceptron import Perceptron, PerceptronR0to1

class MultilayerPerceptron:
    '''
    describes a neural network with 2 layers ( hidden and output )
    '''
    
    #defines grid values, R1to1 means Real in [-1, 1], R0to1 means Real in [0,1]
    (R1to1, R0to1) = (-1, 0)

    def __init__(self, nbr_input, nbr_hidden, nbr_output, grid=R1to1, learning_rate=0.1,
                  momentum=0., temperature=1., random=True, enable_bias=True):
        '''
        builds a neural network with 2 layers
        nbr_input is the number of inputs to the neurons in the hidden layer
        see Perceptron.__init__() for more information about other parameters
        '''
        
        if(grid != self.R1to1 and grid != self.R0to1):
            raise Exception("%d : unknown grid. Please use : %d or %d" % (grid, self.R1to1, self.R0to1))
        
        self.hiddenNeurons = []
        self.outputNeurons = []
        
    
        for _ in range(nbr_hidden):
            if grid == self.R1to1:
                ph = Perceptron(nbr_input, learning_rate, momentum, temperature, Perceptron.HIDDEN, random, enable_bias)
            else:
                ph = PerceptronR0to1(nbr_input, learning_rate, momentum, temperature, Perceptron.HIDDEN, random, enable_bias)
            self.hiddenNeurons.append(ph)
            
        for _ in range(nbr_output):
            if grid == self.R1to1:
                po = Perceptron(nbr_hidden, learning_rate, momentum, temperature, Perceptron.OUTPUT, random, enable_bias)
            else:
                po = PerceptronR0to1(nbr_hidden, learning_rate, momentum, temperature, Perceptron.OUTPUT, random, enable_bias)
            self.outputNeurons.append(po)
                
        self.stateOutputNeurons = []
        self.stateHiddenNeurons = []
        self._last_inputs = []
        self._network_updated = True

    def init_weights_randomly(self, vmin= -0.25, vmax=0.25):
        '''
        assigns a random value between [ vmin, vmax [ to all the weights of the network 
        '''
        for neuron in self.hiddenNeurons + self.outputNeurons:
            neuron.init_weights_randomly(vmin, vmax)
        self._network_updated = True
       
    def init_weights(self, val):
        '''
        assigns the value val to all the weights of the network
        '''
        for neuron in self.hiddenNeurons + self.outputNeurons:
            neuron.init_weights(val)
        
    def calc_output(self, inputs):
        '''
        returns the responses list of the output neurons to these data inputs
        '''
        #avoids unnecessary computations
        if(not self._network_updated and self._last_inputs == inputs):
            return self.stateOutputNeurons
        self._network_updated = False
        self._last_inputs = inputs
        
        #determine the state of hidden neurons
        stateHidden = []
        for neuron in self.hiddenNeurons :
            stateHidden.append(neuron.calc_output(inputs))
        self.stateHiddenNeurons = stateHidden
        
        #then the output layer
        stateOutputs = []
        for neuron in self.outputNeurons :
            stateOutputs.append(neuron.calc_output(stateHidden))
        self.stateOutputNeurons = stateOutputs
        return stateOutputs
    
    def calc_RMS(self, inputs, outputs):
        '''
        returns the RMS ( Root Mean Square ) for the entire output layer
        '''
        return self.calc_RMS_range(inputs, outputs, 0, len(outputs))
    
    def calc_RMS_range(self, inputs, outputs, imin, imax):
        '''
        returns the RMS ( Root Mean Square ) according to the formula :
        $ \sqrt{ \frac{1}{n} \sum \limits_{i=1}^{n} ( o_{i} - d_{i} )^2 } $
        $ with \left\lbrace \begin{array}{lll} n : number\ of\ neurons\ on\ the\ output\ layer\\ o : values\ obtained \\ d : values\ desired \end{array} \right.$
        (it is used to determine the total error of the network)
        '''
        self.calc_output(inputs)
        
        s = 0.
        for i in range(imin, imax):
            s += (self.stateOutputNeurons[i] - outputs[i]) ** 2
        return sqrt(s / (imax - imin))
        
    def train(self, inputs, outputs):
        '''
        trains the network to associate inputs to outputs ( by using the backpropagation algorithm )
        '''
        self.calc_output(inputs)
        
        #build y the error vector to propagate
        y = []
        for i in range(len(self.outputNeurons)) :
            y.append(self.outputNeurons[i].calc_error_propagation(outputs[i]))
        
        yy = []
        for i in range(len(self.hiddenNeurons)):
            w_sum = 0.
            for j in range(len(self.outputNeurons)) :
                w_sum += self.outputNeurons[j].weights[i] * y[j]
            yy.append(self.hiddenNeurons[i].calc_error_propagation(w_sum))
            
        #updates all weights of the network
        for i in range(len(self.hiddenNeurons)) :
            self.hiddenNeurons[i].update_weights(yy[i] , inputs)
 
        for i in range(len(self.outputNeurons)) :
            self.outputNeurons[i].update_weights(y[i] , self.stateHiddenNeurons)
            
        self._network_updated = True
        
if __name__ == '__main__':
    #XOR test on [-1, 1]
    n = MultilayerPerceptron(2, 3, 1, grid=MultilayerPerceptron.R1to1)
    n.init_weights_randomly(-1, 1)
    
    for epoch in range(700):
        n.train([-1, -1], [-1])
        n.train([-1, 1], [1])
        n.train([1, -1], [1])
        n.train([1, 1], [-1])
        
    print(n.calc_output([-1, -1]))
    print(n.calc_output([-1, 1]))
    print(n.calc_output([1, -1]))
    print(n.calc_output([1, 1]))
    
    #[-0.9424440588256111]
    #[0.9583671093424109]
    #[0.9581018151457293]
    #[-0.9435578546516472]
    
    print
    
    #XOR test on [0, 1]
    n = MultilayerPerceptron(2, 3, 1, grid=MultilayerPerceptron.R0to1, momentum=0.2)
    
    for epoch in range(1000):
        n.train([0, 0], [0])
        n.train([0, 1], [1])
        n.train([1, 0], [1])
        n.train([1, 1], [0])
        
    print(n.calc_output([0, 0]))
    print(n.calc_output([0, 1]))
    print(n.calc_output([1, 0]))
    print(n.calc_output([1, 1]))
    
    #[0.18486650386859885]
    #[0.8124738577513108]
    #[0.7566651974008758]
    #[0.257136216923515]

