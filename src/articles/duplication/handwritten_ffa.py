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
from utils import index_max, compare, compare_f, print_liste

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbEpoch = 1000
    nbTry = 10
    display_interval = range(nbEpoch)[::50]
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 16 * 4, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(16 * 4, 16 * 4 * 2, 16 * 16 + 16 * 4 + 10, learning_rate=lrate, momentum=momentum, grid=mode)

        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}

    #create inputs/outputs to learn
    examples = DataFile("digit_handwritten_16.txt", mode)
#    examples = DataFile("digit_shape_16.txt")

    #3 curves
    rms_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_1': [],
              'high_order_2': [],
              'high_order_3': []}
    err_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_5': [],
              'high_order_20': []}

    #learning
    for epoch in range(nbEpoch):
        sum_rms = {'first_order' : 0. ,
                   'high_order_10' : 0.,
                   'high_order_5': 0.,
                   'high_order_1' : 0.,
                   'high_order_2' : 0.,
                   'high_order_3' : 0.}
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.,
                           'high_order_5': 0.,
                           'high_order_20': 0.}
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
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
                sum_rms['high_order_1'] += network['high_order_10'].calc_RMS_range(
                            network['first_order'].stateHiddenNeurons,
                             entire_first_order, 0, 16 * 16)
                sum_rms['high_order_2'] += network['high_order_10'].calc_RMS_range(
                            network['first_order'].stateHiddenNeurons,
                             entire_first_order, 16 * 16, 16 * 16 + 16 * 4)
                sum_rms['high_order_3'] += network['high_order_10'].calc_RMS_range(
                            network['first_order'].stateHiddenNeurons,
                             entire_first_order, 16 * 16 + 16 * 4, 16 * 16 + 16 * 4 + 10)


                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1

                err_one_network['high_order_20'] += 1 - compare(examples.inputs[ex], network['high_order_10'].stateOutputNeurons[0:16 * 16])

                if( not compare_f(network['first_order'].stateHiddenNeurons, 
                                  network['high_order_10'].stateOutputNeurons[16 * 16:16 * 16 + 16 * 4], 0.3) ):
                    err_one_network['high_order_5'] += 1
                    
                if(index_max(network['high_order_10'].stateOutputNeurons[16 * 16 + 16 * 4:16 * 16 + 16 * 4 + 10]) != 
                    index_max(network['first_order'].stateOutputNeurons)):
                    err_one_network['high_order_10'] += 1

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            

        #add plot
        rms_plot['first_order'].append(sum_rms['first_order']/ (nbTry * nbr_network))
        rms_plot['high_order_10'].append(sum_rms['high_order_10']/ (nbTry * nbr_network))
        rms_plot['high_order_1'].append(sum_rms['high_order_1']/ (nbTry * nbr_network * (16 * 16 + 16 * 4 + 10) / (16 * 16)))
        rms_plot['high_order_2'].append(sum_rms['high_order_2']/ (nbTry * nbr_network * (16 * 16 + 16 * 4 + 10) / (16 * 4)))
        rms_plot['high_order_3'].append(sum_rms['high_order_3']/ (nbTry * nbr_network * (16 * 16 + 16 * 4 + 10) / (10)))

        err_plot['first_order'].append(err_one_network['first_order'] / (nbTry * nbr_network))
        err_plot['high_order_10'].append(err_one_network['high_order_10'] / (nbTry * nbr_network))
        err_plot['high_order_20'].append(err_one_network['high_order_20'] / (nbTry * nbr_network))
        err_plot['high_order_5'].append(err_one_network['high_order_5'] / (nbTry * nbr_network))
        
        print(epoch, " rms :", rms_plot['first_order'][epoch], " err : ", err_plot['first_order'][epoch])
    
    #displays rms
    plt.plot(display_interval, [rms_plot['first_order'][i] for i in display_interval],
             label="first-order network", linewidth=2)
    
    plt.plot(display_interval, [rms_plot['high_order_10'][i] for i in display_interval],
             label="high-order network (10 hidden units)", linewidth=2)
    
    plt.plot(display_interval, [rms_plot['high_order_1'][i] for i in display_interval],
             label="input layer of high-order")
        
    plt.plot(display_interval, [rms_plot['high_order_2'][i] for i in display_interval],
             label="hidden layer of high-order")
            
    plt.plot(display_interval, [rms_plot['high_order_3'][i] for i in display_interval],
             label="output layer of high-order")
    
    plt.title('Error RMS of first-order and high-order networks')
    plt.ylabel('RMS ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    #displays errors
    plt.plot(display_interval, [err_plot['first_order'][i] for i in display_interval],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval, [err_plot['high_order_10'][i] for i in display_interval],
             label="output layer ( winner take all )")
    
    plt.plot(display_interval, [err_plot['high_order_5'][i] for i in display_interval],
             label="hidden layer ( | x - o | <= 0.3 )")
    
    plt.plot(display_interval, [err_plot['high_order_20'][i] for i in display_interval],
             label="input layer ( x > 0.5 => activation  )")
    
    plt.title('Error ratio of first-order and high-order networks')
    plt.ylabel('ERROR RATIO')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    
    for j in range(10):
        for k in range(len(examples.inputs)):
            if(index_max(examples.outputs[k]) == j):
                print_liste(examples.inputs[k], 16)
                print()
                networks[0]['first_order'].calc_output(examples.inputs[k])
                o = networks[0]['high_order_10'].calc_output(networks[0]['first_order'].stateHiddenNeurons)[0:16*16]
                print_liste(o, 16)
                
                break
