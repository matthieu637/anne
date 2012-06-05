# -*- coding: UTF-8 -*-
'''
Created on 19 April 2012

@author: Matthieu Zimmer

'''

from perceptron import PerceptronR0to1, Perceptron
from multilayerp import MultilayerPerceptron
from utils import index_max, randmm
from random import shuffle, seed
import matplotlib.pyplot as plt
from data import DataFile
from copy import deepcopy


nbSH=50



class AdHockP(PerceptronR0to1):
    def __init__(self, control):
        self.nbr_input = control.nbr_input
        self.learning_rate = control.learning_rate
        self.momentum = control.momentum
        self.temperature = control.temperature
        self._ntype = control._ntype
        self._enable_bias = control._enable_bias
        

        self.weights = []        
        for w in control.weights:
            self.weights.append(w)
        
        for _ in range(nbSH):
            self.weights.append(randmm(-1, 1))
        
        self.bias = control.bias
        self._last_bias = control._last_bias
        
        self._last_weights = list(self.weights)
        
        self._weights_updated = True
        self._last_inputs = []


class AdHock(MultilayerPerceptron):
    def __init__(self, control):
        self.hiddenNeurons = []
        self.hiddenNeurons2 = []
        self.outputNeurons = []
        
        nbr_hidden = len(control.hiddenNeurons)
        nbr_output = len(control.outputNeurons)
    
        for i in range(nbr_hidden):
            ph = deepcopy(control.hiddenNeurons[i])
            self.hiddenNeurons.append(ph)
            
        for j in range(nbr_output):
            po = AdHockP(control.outputNeurons[j])
            self.outputNeurons.append(po)
            
        for _ in range(nbSH):
            self.hiddenNeurons2.append(PerceptronR0to1(nbr_hidden, learning_rate=0.1, momentum=0.,
                 temperature=1., ntype=Perceptron.HIDDEN, init_w_randomly=True, enable_bias=True))
            
                
        self.stateOutputNeurons = []
        self.stateHiddenNeurons = []
        self.stateHiddenNeurons2 = []
        self._last_inputs = []
        self._network_updated = True

    def calc_output(self, inputs):
        #determine the state of hidden neurons
        stateHidden = []
        for neuron in self.hiddenNeurons :
            stateHidden.append(neuron.calc_output(inputs))
        self.stateHiddenNeurons = stateHidden
        
        stateHidden2 = []
        for neuron in self.hiddenNeurons2 :
            stateHidden2.append(neuron.calc_output(self.stateHiddenNeurons))
        self.stateHiddenNeurons2 = stateHidden2
        
        
        #then the output layer
        stateOutputs = []
        for neuron in self.outputNeurons :
            stateOutputs.append(neuron.calc_output(self.stateHiddenNeurons + self.stateHiddenNeurons2))
        self.stateOutputNeurons = stateOutputs
        return stateOutputs
    
    def train(self, inputs, outputs):
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
            
    
        yyy = []
        for i in range(nbSH):
            w_sum = 0.
            for j in range(len(self.outputNeurons)) :
                w_sum += self.outputNeurons[j].weights[len(self.hiddenNeurons) + i] * y[j]
            yyy.append(self.hiddenNeurons2[i].calc_error_propagation(w_sum))
            
        
        #updates all weights of the network
        for i in range(len(self.hiddenNeurons)) :
            self.hiddenNeurons[i].update_weights(yy[i] , inputs)
            
        for i in range(len(self.hiddenNeurons2)) :
            self.hiddenNeurons2[i].update_weights(yyy[i] , self.stateHiddenNeurons)
 
        for i in range(len(self.outputNeurons)) :
            self.outputNeurons[i].update_weights(y[i] , self.stateHiddenNeurons +self.stateHiddenNeurons2)
            
        self._network_updated = True

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        seed(i)
        control = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode,
                                       temperature=1, random=False, enable_bias=True)
        control.init_weights_randomly(-1, 1)
        
        first_order = AdHock(control)
        
        networks[i] = {'first_order' : first_order,
                    'control': control}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'feedback' : [],
              'control': [],
              'diff': []}
    seed(100)
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'feedback' : 0.,
                 'control': 0.,
                 'diff': 0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['control'].calc_output(examples.inputs[ex])
                network['first_order'].calc_output(examples.inputs[ex])
                
                if(index_max(network['control'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['control'] += 1
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                    
                #learn
#                print(control.hiddenNeurons[0].weights[0])
                network['first_order'].train(examples.inputs[ex],
                             examples.outputs[ex])
                
                
#                print(control.hiddenNeurons[0].weights[0])
                network['control'].train(examples.inputs[ex], examples.outputs[ex])

        perfo['diff'] = (perfo['feedback'] - perfo['control'])
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    
    print("score : ", sum(y_perfo['diff']) / len(y_perfo['diff']))

    plt.title("Feedback by merging (gradient)")
#    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
#    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['control'][3::5], label="control network", linewidth=2)
    
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="feedback", linewidth=2)
#    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
