# -*- coding: UTF-8 -*-
'''
Created on 19 March 2012

@author: Matthieu Zimmer

'''

from perceptron import PerceptronR0to1, Perceptron
from multilayerp import MultilayerPerceptron
from utils import index_max
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile

class AdHock(MultilayerPerceptron):
    def __init__(self, nbr_input, nbr_hidden, nbr_output, learning_rate=0.1,
                  momentum=0., temperature=1., random=True, enable_bias=True):
        self.hiddenNeurons = []
        self.outputNeurons = []
        
    
        for _ in range(nbr_hidden):
            ph = PerceptronR0to1(nbr_input, learning_rate, momentum, temperature, Perceptron.HIDDEN, random, enable_bias)
            self.hiddenNeurons.append(ph)
            
        for _ in range(nbr_output):
            po = PerceptronR0to1(nbr_hidden + 2, learning_rate, momentum, temperature, Perceptron.OUTPUT, random, enable_bias)
            self.outputNeurons.append(po)
                
        self.stateOutputNeurons = []
        self.stateHiddenNeurons = []
        self._last_inputs = []
        self._network_updated = True

    def calc_hidden(self, inputs):
        #determine the state of hidden neurons
        stateHidden = []
        for neuron in self.hiddenNeurons :
            stateHidden.append(neuron.calc_output(inputs))
        self.stateHiddenNeurons = stateHidden

    def calc_output(self, addition):
        #then the output layer
        stateOutputs = []
        for neuron in self.outputNeurons :
            stateOutputs.append(neuron.calc_output(self.stateHiddenNeurons + addition))
        self.stateOutputNeurons = stateOutputs
        return stateOutputs
    
    def train(self, inputs, outputs, addition):
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
            self.outputNeurons[i].update_weights(y[i] , self.stateHiddenNeurons+ addition)
            
        self._network_updated = True
        

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 3
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = AdHock(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum)
        high_order_h = MultilayerPerceptron(100, 20, 2, learning_rate=0.1, momentum=0., grid=mode)
        
        first_order.init_weights_randomly(-1, 1)
#        high_order_h.init_weights_randomly(-1, 1)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'feedback' : []}
    
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'feedback' : 0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_hidden(examples.inputs[ex])
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
                network['first_order'].calc_output([0, 0])
                
                cell = [0, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, 0]
                
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max(cell)):
                    perfo['high_order_h'] += 1

                if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                    perfo['wager_proportion'] += 1
                    
                
                network['first_order'].calc_output(network['high_order_h'].stateOutputNeurons)
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['feedback'] += 1
                
                #learn
                tmp = list(network['first_order'].stateHiddenNeurons)
                network['first_order'].train(examples.inputs[ex],
                             examples.outputs[ex], network['high_order_h'].stateOutputNeurons)
                
                
                network['high_order_h'].train(tmp,  cell)

        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    

    plt.title("Feedback by merging")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
