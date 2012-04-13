# -*- coding: UTF-8 -*-
'''
Created on 30 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from utils import index_max
from random import shuffle
from random import seed
import matplotlib.pyplot as plt
from data import DataFileR

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    nbEpoch = 300
    nbTry = 20
    display_interval = range(nbEpoch)[::5]
    #seed(0)
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(4, 10, 3, learning_rate=0.05, momentum=0.3, grid=mode)
        high_order_h = MultilayerPerceptron(10, 10, 2, learning_rate=0.2, momentum=0.5, grid=mode)
        
        first_order.init_weights_randomly(-1, 1)
        high_order_h.init_weights_randomly(-1, 1)
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h}

    #create example
    examples = DataFileR("iris.txt", mode)
    
    print(len(examples.inputs))

    #3 curves
    y_plot = {'first_order' : [] ,
              'high_order_h' : [],
              'high_order_l': []}

    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'high_order_l': []}
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                   'high_order_h' : 0.,
                   'high_order_l': 0.}
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
                    
                    
                if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                    perfo['high_order_l'] += 1
                
                #learn
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            
        y_perfo['first_order'].append(perfo['first_order'] / (nbTry * nbr_network))
        y_perfo['high_order_h'].append(perfo['high_order_h'] / (nbTry * nbr_network))
        y_perfo['high_order_l'].append(perfo['high_order_l'] / (nbTry * nbr_network))
        
        print(epoch)
    

    plt.title("Performance of first-order and higher-order networks")
    plt.plot(display_interval , y_perfo['first_order'][::5], label="first-order network")
    plt.plot(display_interval , y_perfo['high_order_h'][::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_perfo['high_order_l'][::5], label="proportion of high wagers")
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
