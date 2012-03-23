# -*- coding: UTF-8 -*-
'''
Created on 23 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from utils import index_max
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from data import DataFile


if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 16 * 4, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        high_order_h = MultilayerPerceptron(16 * 4, 16 * 4 * 2 , 2, learning_rate=0.1, momentum=0., grid=mode)
        
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
                network['first_order'].calc_output(examples.inputs[ex])
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
                
                cell = [mode, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, mode]
                
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max(cell)):
                    perfo['high_order_h'] += 1
                    
                
                

                        
                if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                    perfo['wager_proportion'] += 1
                
                #learn
                tmp = list(network['high_order_h'].stateOutputNeurons)
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                
                if(index_max(tmp) == 0):
                    network['first_order'].set_learning_rate(0.15)
                else:
                    network['first_order'].set_learning_rate(0.05)
                network['first_order'].train(examples.inputs[ex],
                                         examples.outputs[ex])
                
        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)

    plt.title("Performance of first-order and higher-order networks with feedback ( Master )")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
#    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    

    
    plb.hist
    plb.show()
