# -*- coding: UTF-8 -*-
'''
Created on 21 fevr. 2012

@author: matthieu637

Article test
'''

from network import MultilayerNetwork
from utils import findMax
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile

if __name__ == '__main__':
    mode = MultilayerNetwork.R0to1
    nbr_network = 1
    momentum = 0.5
    nbEpoch = 201
    display_interval = range(nbEpoch)[6::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerNetwork(7, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        high_order_h = MultilayerNetwork(100, 100, 2, learning_rate=0.1, momentum=momentum, grid=mode)
        high_order_l = MultilayerNetwork(100, 100, 2, learning_rate=10e-7, momentum=momentum, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h,
                    'high_order_l' : high_order_l}

    for network in networks:
        for k in network.keys():
                network[k].init_random_weights(-1,1)

#    plt.title("  0.70 0")
    #create example
    examples = DataFile("../data/digital_shape.txt", mode)

    #3 curves
    y_plot = {'first_order' : [] ,
              'high_order_h' : [],
              'high_order_l': []}

    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'first_order' : 0. ,
                   'high_order_h' : 0.,
                   'high_order_l': 0.}
        for network in networks:
#            for ex in range(10):
#            for ex in np.random.randint(0, 10, 10):
            l_exx = range(10)
            shuffle(list(l_exx))
            for ex in l_exx:
                sum_rms['first_order'] += network['first_order'].calc_RMS(
                                            examples.inputs[ex],
                                            examples.outputs[ex])
                cell = [mode, 1] \
                        if findMax(network['first_order'].stateOutputNeurons) == findMax(examples.outputs[ex]) \
                        else [1, mode]
                
#                convert = list(map(lambda x:, network['first_order'].stateHiddenNeurons))
                
                sum_rms['high_order_h'] += network['high_order_h'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                             cell)

                sum_rms['high_order_l'] += network['high_order_l'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                            cell)
                
                
                #learn
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['high_order_l'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])


        y_plot['first_order'].append(sum_rms['first_order'])
        y_plot['high_order_h'].append(sum_rms['high_order_h'])
        y_plot['high_order_l'].append(sum_rms['high_order_l'])
        
    # divided by the maximum error
    max_err = (max(y_plot['first_order']),
               max(y_plot['high_order_h']),
               max(y_plot['high_order_l']))
    m = max(y_plot['first_order'])
    for i in range(nbEpoch):
        y_plot['first_order'][i] /= max_err[0]
        y_plot['high_order_h'][i] /= max_err[1]
        y_plot['high_order_l'][i] /= max_err[2]
    
    #displays
    plt.plot(display_interval , y_plot['first_order'][6::5], label="first-order network")
    plt.plot(display_interval , y_plot['high_order_h'][6::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_plot['high_order_l'][6::5], label="high-order network (low learning rate)")
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend()
    plt.show()
    

    #testing
    for ex in range(10):
        for network in networks:
            print(examples.inputs[ex])
            print(network['first_order'].calc_output(examples.inputs[ex]))
            print(findMax(network['first_order'].calc_output(examples.inputs[ex])))
