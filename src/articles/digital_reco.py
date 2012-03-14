# -*- coding: UTF-8 -*-
'''
Created on 21 February 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from utils import index_max
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 20
    momentum = 0.5
    nbEpoch = 201
    display_interval = range(nbEpoch)[6::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(7, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode )
        high_order_h = MultilayerPerceptron(100, 100, 2, learning_rate=0.1, momentum=0, grid=mode )
        high_order_l = MultilayerPerceptron(100, 100, 2, learning_rate=10e-7, momentum=0, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h,
                    'high_order_l' : high_order_l}

    for network in networks:
        for k in network.keys():
            network[k].init_weights_randomly(-1, 1)

    #create example
    examples = DataFile("../data/digital_shape.txt", mode)

    #3 curves
    y_plot = {'first_order' : [] ,
              'high_order_h' : [],
              'high_order_l': []}

    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'high_order_l': []}
    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'first_order' : 0. ,
                   'high_order_h' : 0.,
                   'high_order_l': 0.}
        perfo = {'first_order' : [] ,
                 'high_order_h' : [],
                 'high_order_l': []}
        for network in networks:
            perfo_i = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'high_order_l': 0.}
            
            l_exx = list(range(10))
            shuffle(l_exx)
            for ex in l_exx:
                sum_rms['first_order'] += network['first_order'].calc_RMS(
                                            examples.inputs[ex],
                                            examples.outputs[ex])
                cell = [mode, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, mode]
                
                sum_rms['high_order_h'] += network['high_order_h'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                             cell)

                sum_rms['high_order_l'] += network['high_order_l'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                            cell)
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo_i['first_order'] += 1
                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max(cell)):
                    perfo_i['high_order_h'] += 1
                if(index_max(network['high_order_l'].stateOutputNeurons) == index_max(cell)):
                    perfo_i['high_order_l'] += 1
                
                #learn
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['high_order_l'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            
            perfo['first_order'].append(perfo_i['first_order'] / 10)
            perfo['high_order_l'].append(perfo_i['high_order_l'] / 10)
            perfo['high_order_h'].append(perfo_i['high_order_h'] / 10)
            
        y_perfo['first_order'].append(sum(perfo['first_order']) / nbr_network)
        y_perfo['high_order_h'].append(sum(perfo['high_order_h']) / nbr_network)
        y_perfo['high_order_l'].append(sum(perfo['high_order_l']) / nbr_network)

        y_plot['first_order'].append(sum_rms['first_order'])
        y_plot['high_order_h'].append(sum_rms['high_order_h'])
        y_plot['high_order_l'].append(sum_rms['high_order_l'])
        
        print(epoch)
        
    # divided by the maximum error
    max_err = (max(y_plot['first_order']),
               max(y_plot['high_order_h']),
               max(y_plot['high_order_l']))

    for i in range(nbEpoch):
        y_plot['first_order'][i] /= max_err[0]
        y_plot['high_order_h'][i] /= max_err[1]
        y_plot['high_order_l'][i] /= max_err[2]
    
    #displays
    plt.title("Square error of first-order and higher-order networks")
    plt.plot(display_interval , y_plot['first_order'][6::5], label="first-order network")
    plt.plot(display_interval , y_plot['high_order_h'][6::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_plot['high_order_l'][6::5], label="high-order network (low learning rate)")
    plt.ylabel('MEAN SQUARE ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    

    plt.title("Performance of first-order and higher-order networks")
    plt.plot(display_interval , y_perfo['first_order'][6::5], label="first-order network")
    plt.plot(display_interval , y_perfo['high_order_h'][6::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_perfo['high_order_l'][6::5], label="high-order network (low learning rate)")
    plt.ylabel('SUCCESS')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    #testing
    for ex in range(10):
        for network in networks:
            print(examples.inputs[ex])
            print(network['first_order'].calc_output(examples.inputs[ex]))
            print(index_max(network['first_order'].calc_output(examples.inputs[ex])))

