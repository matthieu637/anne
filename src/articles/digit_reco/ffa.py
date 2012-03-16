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
from utils import index_max, compare

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.9
    lrate = 0.1
    nbShape = 10
    nbEpoch = 1000
    display_interval = [0, 25, 50, 100, 200, 500, 999]
    display_interval2 = range(nbEpoch)[::6]
    
    
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
    rms_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_5' : [],
              'high_order_20' : [],
              'high_order': []}

    err_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_5': [],
              'high_order_20': []}

    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'first_order' : 0. ,
                   'high_order_10' : 0.,
                   'high_order_5': 0.,
                   'high_order_20': 0.,
                   'high_order':0.}
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.,
                           'high_order_5': 0.,
                           'high_order_20': 0.}
        
        for network in networks:
            l_exx = list(range(nbShape))
            shuffle(l_exx)
            for ex in l_exx:              
                #RMS
                sum_rms['first_order'] += network['first_order'].calc_ME(
                                            examples.inputs[ex],
                                            examples.outputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                
                sum_rms['high_order_10'] += network['high_order_10'].calc_ME_range(
                                            network['first_order'].stateHiddenNeurons,
                                             entire_first_order, 25, 35)

                sum_rms['high_order_5'] += network['high_order_10'].calc_ME_range(
                                            network['first_order'].stateHiddenNeurons,
                                            entire_first_order, 20, 25)
                
                sum_rms['high_order_20'] += network['high_order_10'].calc_ME_range(
                                            network['first_order'].stateHiddenNeurons,
                                            entire_first_order, 0, 20)
                
                sum_rms['high_order'] += network['high_order_10'].calc_ME(
                                            network['first_order'].stateHiddenNeurons,
                                             entire_first_order)
                
                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
                if(index_max(network['high_order_10'].stateOutputNeurons[25:35]) != index_max(network['first_order'].stateOutputNeurons)):
                    err_one_network['high_order_10'] += 1

                err_one_network['high_order_5'] += network['high_order_10'].calc_ME_range(network['first_order'].stateHiddenNeurons, 
                                                                                          entire_first_order,
                                                                                          20, 25)
                err_one_network['high_order_20'] += 1 - compare(examples.inputs[ex], network['high_order_10'].stateOutputNeurons[0:20])
                

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            

        #add plot
        rms_plot['first_order'].append(sum_rms['first_order'] / (nbShape * nbr_network))
        rms_plot['high_order_10'].append(sum_rms['high_order_10'] / (nbShape * nbr_network))
        rms_plot['high_order_5'].append(sum_rms['high_order_5'] / (nbShape * nbr_network))
        rms_plot['high_order_20'].append(sum_rms['high_order_20'] / (nbShape * nbr_network))
        rms_plot['high_order'].append(sum_rms['high_order'] / (nbShape * nbr_network))
        
        
        err_plot['first_order'].append(err_one_network['first_order'] / (nbShape * nbr_network))
        err_plot['high_order_10'].append(err_one_network['high_order_10'] / (nbShape * nbr_network))
        err_plot['high_order_20'].append(err_one_network['high_order_20'] / (nbShape * nbr_network))
        err_plot['high_order_5'].append(err_one_network['high_order_5'] / (nbShape * nbr_network))
        
        
        print(epoch)
        
    # divided by the maximum error 
    mh5 = max(rms_plot['high_order_5'])
    mh10 = max(rms_plot['high_order_10'])
    mh20 = max(rms_plot['high_order_20'])
    mf = max(rms_plot['first_order'])
    mh = max(rms_plot['high_order'])
    
#    for i in range(nbEpoch):
##        rms_plot['first_order'][i] /= 10
#        rms_plot['high_order_10'][i] *= 10/35
#        rms_plot['high_order_5'][i] *= 5/35
#        rms_plot['high_order_20'][i] *= 20/35
##        rms_plot['high_order'][i] /= 10
    
    #displays rms
    plt.plot(display_interval2, [rms_plot['first_order'][i] for i in display_interval2],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval2, [rms_plot['high_order_10'][i] for i in display_interval2],
             label="10 outputs")
    
    plt.plot(display_interval2, [rms_plot['high_order_5'][i] for i in display_interval2],
             label="5 hidden")
    
    plt.plot(display_interval2, [rms_plot['high_order_20'][i] for i in display_interval2],
             label="20 inputs")
    
    plt.plot(display_interval2, [rms_plot['high_order'][i] for i in display_interval2],
             label="high-order network (10 hidden units)",
             linewidth=2)
    
    plt.title('Sum square error of first-order and high-order networks')
    plt.ylabel('SQUARE ERROR')
    plt.xlabel("EPOCHS")
#    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    
    
    
    plt.plot(display_interval2, [err_plot['first_order'][i] for i in display_interval2],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval2, [err_plot['high_order_10'][i] for i in display_interval2],
             label="10 outputs")
    
    plt.plot(display_interval2, [err_plot['high_order_5'][i] for i in display_interval2],
             label="5 hidden")
    
    plt.plot(display_interval2, [err_plot['high_order_20'][i] for i in display_interval2],
             label="20 inputs")

    
    
    plt.title('Sum square error of first-order and high-order networks')
    plt.ylabel('SQUARE ERROR')
    plt.xlabel("EPOCHS")
#    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    