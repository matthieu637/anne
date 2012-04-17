# -*- coding: UTF-8 -*-
'''
Created on 19 March 2012

@author: Matthieu Zimmer

'''

from perceptron import PerceptronR0to1
from multilayerp import MultilayerPerceptron
from utils import index_max, randmm
from random import shuffle, seed
import matplotlib.pyplot as plt
from data import DataFile
from copy import deepcopy

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
        
        for _ in range(8):
            self.weights.append(randmm(-1, 1))
        
        self.bias = control.bias
        self._last_bias = control._last_bias
        
        self._last_weights = list(self.weights)
        
        self._weights_updated = True
        self._last_inputs = []
    
    def update_weights_1(self, error, inputs, addition):
        self.calc_output(inputs + addition)
        
        ii = len(inputs)
        tmp_weights = list(self.weights[0:ii])
        
        for j in range(len(inputs)):
            dw = self.weights[j] - self._last_weights[j] 
#            print(dw)
            self.weights[j] += self.learning_rate * error * inputs[j] + self.momentum * dw
        
        if self._enable_bias:
            tmp_bias = self.bias 
            self.bias += self.learning_rate * error + self.momentum * (self.bias - self._last_bias)
            self._last_bias = tmp_bias
        
        
        self._last_weights[0:ii] = tmp_weights
        self._weights_updated = True
        
    def update_weights_2(self, output, beg, inputs, hel):
        self.calc_output(hel + inputs)
        
        ii = len(inputs)
        tmp_weights = list(self.weights[beg:beg + ii])
        

        for j in range(ii):
            dw = self.weights[beg + j] - self._last_weights[beg + j] 
            self.weights[beg + j] += (self.learning_rate * (output - self._state) * inputs[j] + self.momentum * dw) / (len(inputs))

        
        self._last_weights[beg:beg + ii] = tmp_weights
        self._weights_updated = True


class AdHock(MultilayerPerceptron):
    def __init__(self, control):
        self.hiddenNeurons = []
        self.outputNeurons = []
        
        nbr_hidden = len(control.hiddenNeurons)
        nbr_output = len(control.outputNeurons)
    
        for i in range(nbr_hidden):
            ph = deepcopy(control.hiddenNeurons[i])
#            PerceptronR0to1(nbr_input, learning_rate, momentum, temperature, Perceptron.HIDDEN, random, enable_bias)
            self.hiddenNeurons.append(ph)
            
        for j in range(nbr_output):
            po = AdHockP(control.outputNeurons[j])
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
            self.outputNeurons[i].update_weights_1(y[i] , self.stateHiddenNeurons, addition)
            self.outputNeurons[i].update_weights_2(outputs[i], len(self.stateHiddenNeurons), addition, self.stateHiddenNeurons)
            
        self._network_updated = True
        
def ampli(l, n):
    ll = [0 for _ in range(n)]
    for i in range(n // 2):
        ll[i] = max(l) if index_max(l) == 0 else min(l)
    for i in range(n // 2, n):
        ll[i] = max(l) if index_max(l) == 1 else min(l)
    return ll

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    seed(100)
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        control = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode,
                                       temperature=1, random=False, enable_bias=True)
        control.init_weights_randomly(-1, 1)
        
        first_order = AdHock(control)
        high_order_h = MultilayerPerceptron(100, 20, 2, learning_rate=0.1, momentum=0., grid=mode)
#        high_order_h.init_weights_randomly(-1, 1)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h,
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
                network['first_order'].calc_hidden(examples.inputs[ex])
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
                network['first_order'].calc_output([0] * 8)
                
                cell = [0, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, 0]
                
                if(index_max(network['control'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['control'] += 1
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max(cell)):
                    perfo['high_order_h'] += 1

                if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                    perfo['wager_proportion'] += 1
                    
                
                cell2 = [0, 0]
                network['first_order'].calc_output(ampli(network['high_order_h'].stateOutputNeurons, 8))
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['feedback'] += 1
                    cell2[1] = 1
                else :
                    cell2[0] = 1
                
                
                #learn
#                print(control.hiddenNeurons[0].weights[0])
                tmp = list(network['first_order'].stateHiddenNeurons)
                network['first_order'].train(examples.inputs[ex],
                             examples.outputs[ex], ampli(network['high_order_h'].stateOutputNeurons, 8))
                
                
                network['high_order_h'].train(tmp, cell2)
#                print(control.hiddenNeurons[0].weights[0])
                network['control'].train(examples.inputs[ex], examples.outputs[ex])

        perfo['diff'] = (perfo['feedback'] - perfo['control'])
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    
    print("score : ", sum(y_perfo['diff']) / len(y_perfo['diff']))

    plt.title("Feedback by merging")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=1)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['control'][3::5], label="control network", linewidth=2)
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    
    plt.title("Diff by merging")
    plt.plot(display_interval , y_perfo['diff'][3::5], label="diff", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()
    
