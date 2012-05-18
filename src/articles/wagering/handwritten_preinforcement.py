# -*- coding: UTF-8 -*-
'''
Created on 18 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from utils import index_max
from random import shuffle, seed
import matplotlib.pyplot as plt
from prenforcement import PRenforcement
from data import DataFile

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
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        first_order.init_weights_randomly(-1, 1)
        high_order_h = PRenforcement(100, temperature=1.)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

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
            seed(100+epoch*(nbr_network+1)+networks.index(network))
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(examples.inputs[ex])
                cell = 1 \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else 0
                
                res = network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)

                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                if(res == cell):
                    perfo['high_order_h'] += 1
                    
                    
                if(res == 1):
                    perfo['high_order_l'] += 1
                
                #learn
                network['high_order_h'].train(res == cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            
        y_perfo['first_order'].append(perfo['first_order'] / (nbTry * nbr_network))
        y_perfo['high_order_h'].append(perfo['high_order_h'] / (nbTry * nbr_network))
        y_perfo['high_order_l'].append(perfo['high_order_l'] / (nbTry * nbr_network))
        
        print(epoch)
    

    plt.title("Performance of first-order and higher-order networks (prenforcement)")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_l'][3::5], label="proportion of high wagers")
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
