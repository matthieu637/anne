# -*- coding: UTF-8 -*-
'''
Created on 13 March 2012

@author: Matthieu Zimmer

Article implementation
'''

from multilayerp import MultilayerPerceptron
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 10
    momentum = 0.9
    lrate = 0.1
    nbEpoch = 1000
    display_interval = [0, 25, 50, 100, 200, 500, 999]
    display_interval2 = range(nbEpoch)[::4]
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(20, 5, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(5, 10, 35, learning_rate=lrate, momentum=momentum, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}

    #create inputs/outputs to learn
    examples = DataFile("../../data/digit_shape.txt", mode)


    #3 curves
    rms_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'dw_1': [],
              'dw_2': [],
              'dw_3': [],
              'dw_4': []}

    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'first_order' : 0. ,
                   'high_order_10' : 0.,
                   'high_order_5': 0.,
                   'dw_1': 0.,
                   'dw_2': 0.,
                   'dw_3': 0.,
                   'dw_4': 0.}
        
        for network in networks:
            l_exx = list(range(10))
            shuffle(l_exx)
            for ex in l_exx:              
                #RMS
                sum_rms['first_order'] += network['first_order'].calc_RMS(
                                            examples.inputs[ex],
                                            examples.outputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                
                sum_rms['high_order_10'] += network['high_order_10'].calc_RMS(
                                            network['first_order'].stateHiddenNeurons,
                                             entire_first_order)
                

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                                             
                sum_rms['dw_1'] += network['first_order'].calc_sum_dw_hidden()
                sum_rms['dw_2'] += network['first_order'].calc_sum_dw_outputs()
                sum_rms['dw_3'] += network['high_order_10'].calc_sum_dw_hidden()
                sum_rms['dw_4'] += network['high_order_10'].calc_sum_dw_outputs()
                

        #add plot
        rms_plot['first_order'].append(sum_rms['first_order'])
        rms_plot['high_order_10'].append(sum_rms['high_order_10'])
        rms_plot['dw_1'].append(sum_rms['dw_1'])
        rms_plot['dw_2'].append(sum_rms['dw_2'])
        rms_plot['dw_3'].append(sum_rms['dw_3'])
        rms_plot['dw_4'].append(sum_rms['dw_4'])
        
        print(epoch)

    # divided by the maximum error
    max_err = (max(rms_plot['first_order']),
               max(rms_plot['high_order_10']),
               max(rms_plot['dw_1']),
               max(rms_plot['dw_2']),
               max(rms_plot['dw_3']),
               max(rms_plot['dw_4'])
               )
    
    #displays rms
             
    plt.plot(display_interval, [rms_plot['dw_1'][i] for i in display_interval],
             label="first-order hidden")
             
    plt.plot(display_interval, [rms_plot['dw_2'][i] for i in display_interval],
             label="first-order outputs")
    
    plt.plot(display_interval, [rms_plot['dw_3'][i] for i in display_interval],
             label="high-order hidden")
             
    plt.plot(display_interval, [rms_plot['dw_4'][i] for i in display_interval],
             label="high-order outputs")
    
    plt.title('Sum dw of first-order and high-order networks')
    plt.ylabel('SUM DW')
    plt.xlabel("EPOCHS")
#    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    