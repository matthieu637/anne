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
    nbr_network = 20
    momentum = 0.9
    lrate = 0.1
    nbEpoch = 1000
    display_interval = range(nbEpoch)[::15]
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(20, 5, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(5, 10, 35, learning_rate=lrate, momentum=momentum, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}

    #create inputs/outputs to learn
    examples = DataFile("digit_shape.txt", mode)


    #3 curves
    rms_plot = {'dw_1': [],
              'dw_2': [],
              'dw_3': [],
              'dw_4': [],
              'dw_5': [],
              'dw_6': [],
              'dw_7': [],
              'dw_8': []}

    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'dw_1': 0.,
                   'dw_2': 0.,
                   'dw_3': 0.,
                   'dw_4': 0.,
                   'dw_5': 0.,
                   'dw_6': 0.,
                   'dw_7': 0.,
                   'dw_8': 0.}
        
        for network in networks:
            l_exx = list(range(10))
            shuffle(l_exx)
            for ex in l_exx:              
                #RMS
                network['first_order'].calc_output(examples.inputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                
                network['high_order_10'].calc_output(network['first_order'].stateHiddenNeurons)
                

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                                             
                sum_rms['dw_1'] += network['first_order'].calc_sum_dw_hidden()
                sum_rms['dw_2'] += network['first_order'].calc_sum_dw_outputs()
                sum_rms['dw_3'] += network['high_order_10'].calc_sum_dw_hidden()
                sum_rms['dw_4'] += network['high_order_10'].calc_sum_dw_outputs()
                
                sum_rms['dw_5'] += network['first_order'].calc_sum_dw_hidden()/5
                sum_rms['dw_6'] += network['first_order'].calc_sum_dw_outputs()/10
                sum_rms['dw_7'] += network['high_order_10'].calc_sum_dw_hidden()/10
                sum_rms['dw_8'] += network['high_order_10'].calc_sum_dw_outputs()/35
                
        #add plot
        for k in rms_plot.keys():
            rms_plot[k].append(sum_rms[k] / (10 * nbr_network))
        
        print(epoch)

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
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    
    #displays mean sum dw
    plt.plot(display_interval, [rms_plot['dw_5'][i] for i in display_interval],
             label="first-order hidden")
             
    plt.plot(display_interval, [rms_plot['dw_6'][i] for i in display_interval],
             label="first-order outputs")
    
    plt.plot(display_interval, [rms_plot['dw_7'][i] for i in display_interval],
             label="high-order hidden")
             
    plt.plot(display_interval, [rms_plot['dw_8'][i] for i in display_interval],
             label="high-order outputs")
    
    plt.title('Mean sum dw of first-order and high-order networks')
    plt.ylabel('MEAN SUM DW')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()
    