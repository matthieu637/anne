# -*- coding: UTF-8 -*-
'''
Created on 18 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from perceptron import PerceptronR0to1
from utils import index_max
from random import shuffle, seed
import matplotlib.pyplot as plt
from data import DataFile


DEBUG = False

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1 if DEBUG else 5
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        seed(i)
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        first_order.init_weights_randomly(-1, 1)
        
        high_order_h = MultilayerPerceptron(100, 20, 2, learning_rate=0.1, momentum=0., grid=mode)
        feedback = [PerceptronR0to1(100+20, 0.1, 0., True) for _ in range(10)]
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h,
                    'feedback': feedback}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'feedback' : []}
    
    seed(100)
    
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
                network['first_order'].calc_output(examples.inputs[ex])
                cell = [mode, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, mode]
                
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)

                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max(cell)):
                    perfo['high_order_h'] += 1
                    
                
                res = [ network['feedback'][i].calc_output(network['first_order'].stateHiddenNeurons + 
                                                           network['high_order_h'].stateHiddenNeurons) for i in range(10)]
                if(index_max(res) == index_max(examples.outputs[ex])):
                    perfo['feedback'] += 1
                    
                if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                    perfo['wager_proportion'] += 1
                
                #learn
                for i in range(10):
                    network['feedback'][i].train(network['first_order'].stateHiddenNeurons + 
                                                 network['high_order_h'].stateHiddenNeurons
                                                 , examples.outputs[ex][i])
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])

        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    

    plt.title("Feedback with a third perceptron network (hidden FoN/SoN) ")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    if(DEBUG):
        plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
        plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
