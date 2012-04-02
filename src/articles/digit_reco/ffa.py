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
from articles.digit_reco.discretize import discretis


if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.9
    lrate = 0.1
    nbShape = 10
    nbEpoch = 1000
    display_interval = range(nbEpoch)[::6]
    
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
                
                if(discretis(network['high_order_10'].stateOutputNeurons[20:25], 4) != discretis(network['first_order'].stateHiddenNeurons, 4)):
                    err_one_network['high_order_5'] += 1
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
    
    #displays rms
    plt.plot(display_interval, [rms_plot['first_order'][i] for i in display_interval],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval, [rms_plot['high_order_10'][i] for i in display_interval],
             label="10 outputs")
    
    plt.plot(display_interval, [rms_plot['high_order_5'][i] for i in display_interval],
             label="5 hidden")
    
    plt.plot(display_interval, [rms_plot['high_order_20'][i] for i in display_interval],
             label="20 inputs")
    
    plt.plot(display_interval, [rms_plot['high_order'][i] for i in display_interval],
             label="high-order network (10 hidden units)",
             linewidth=2)
    
    plt.title('Mean error of one neuron')
    plt.ylabel('MEAN ERROR')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    
    plt.plot(display_interval, [err_plot['first_order'][i] for i in display_interval],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval, [err_plot['high_order_10'][i] for i in display_interval],
             label="10 outputs ( winner take all )")
    
    plt.plot(display_interval, [err_plot['high_order_5'][i] for i in display_interval],
             label="5 hidden  ( | x - o | <= 0.1 )")
    
    plt.plot(display_interval, [err_plot['high_order_20'][i] for i in display_interval],
             label="20 inputs ( x > 0.5 => activation  )")
    
    plt.title('Error ratio of first-order and high-order networks')
    plt.ylabel('ERROR RATIO')
    plt.xlabel("EPOCHS")
#    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    