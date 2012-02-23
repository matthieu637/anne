# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637

Article implementation
$<img src="../../datadoc/digit_reco.png" />$
'''

from network import MultilayerNetwork
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile

if __name__ == '__main__':
    mode = MultilayerNetwork.R0to1
    nbr_network = 5
    momentum = 0.9
    lrate = 0.1
    nbEpoch = 1000
    display_interval = [0, 25, 50, 100, 200, 500, 999]
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerNetwork(20, 5, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerNetwork(5, 10, 35, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_5 = MultilayerNetwork(5, 5, 35, learning_rate=lrate, momentum=momentum, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10,
                        'high_order_5' : high_order_5}

    #create inputs/outputs to learn
    examples = DataFile("../data/digit_shape.txt", mode)


    #3 curves
    y_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_5': []}

    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'first_order' : 0. ,
                   'high_order_10' : 0.,
                   'high_order_5': 0.}
        
        for network in networks:
            l_exx = list(range(10))
            shuffle(l_exx)
            for ex in l_exx:              
                #add RMS
                sum_rms['first_order'] += network['first_order'].calc_RMS(
                                            examples.inputs[ex],
                                            examples.outputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                
                sum_rms['high_order_10'] += network['high_order_10'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                             entire_first_order)

                sum_rms['high_order_5'] += network['high_order_5'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                            entire_first_order)
                
                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['high_order_5'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])

        #add plot
        y_plot['first_order'].append(sum_rms['first_order'])
        y_plot['high_order_10'].append(sum_rms['high_order_10'])
        y_plot['high_order_5'].append(sum_rms['high_order_5'])
        

    # divided by the maximum error
    max_err = (max(y_plot['first_order']),
               max(y_plot['high_order_10']),
               max(y_plot['high_order_5']))
    for i in range(nbEpoch):
        y_plot['first_order'][i] /= max_err[0]
        y_plot['high_order_10'][i] /= max_err[1]
        y_plot['high_order_5'][i] /= max_err[2]
    
    #displays
    plt.plot(display_interval, [y_plot['first_order'][i] for i in display_interval], 
             label="first-order network")
    
    plt.plot(display_interval, [y_plot['high_order_10'][i] for i in display_interval], 
             label="high-order network (10 hidden units)")
    
    plt.plot(display_interval, [y_plot['high_order_5'][i] for i in display_interval], 
             label="high-order network (5 hidden units)")
    
    plt.title('Error proportion of first-order and high-order networks')
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend()
    plt.show()
    
    '''
    result
    $<img src="../../results/digit_reco.png" />$
    '''
