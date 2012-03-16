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
from utils import index_max

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbEpoch = 201
    nbTry = 50
    display_interval = [0, 25, 50, 100, 200] #, 500, 999]
    display_interval2 = range(nbEpoch)[::10]
    
    
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16*16, 16*4, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(16*4, 16*4*2, 16*16+16*4+10, learning_rate=lrate, momentum=momentum, grid=mode)
#        high_order_5 = MultilayerPerceptron(20, 20, 32*32+20+10, learning_rate=lrate, momentum=momentum, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}
#                        ,
#                        'high_order_5' : high_order_5}

    #create inputs/outputs to learn
    examples = DataFile("../../data/digit_handwritten_16.txt", mode)

#    print(len(examples.inputs[0]), len(examples.inputs))
#    print(len(examples.outputs[0]), len(examples.outputs))
#    
#    l_ex = list(range(len(examples.inputs)))
#    shuffle(l_ex)
#    n_inp=[]
#    n_out=[]
#    for ex in l_ex[0:500]:
#        n_inp.append(examples.inputs[ex])
#        n_out.append(examples.outputs[ex])
#    
#    examples.inputs = n_inp
#    examples.outputs = n_out
    
    print(len(examples.inputs[0]), len(examples.inputs))
    print(len(examples.outputs[0]), len(examples.outputs))
    
    
    #3 curves
    rms_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_1': [],
              'high_order_2': [],
              'high_order_3': []}
    err_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'high_order_5': []}

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
                           'high_order_5': 0.}
        
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
                             entire_first_order, 16*16, 16*16+10)
                sum_rms['high_order_3'] += network['high_order_10'].calc_RMS_range(
                            network['first_order'].stateHiddenNeurons,
                             entire_first_order, 16*16 +16*4, 16*16+16*4+10)

#                sum_rms['high_order_5'] += network['high_order_5'].calc_RMS(
#                                            network['first_order'].stateHiddenNeurons,
#                                            entire_first_order)
#                
                #error
                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
#                if(index_max(network['high_order_5'].stateOutputNeurons[25:35]) != index_max(examples.outputs[ex])):
#                    err_one_network['high_order_5'] += 1
#                if(index_max(network['high_order_10'].stateOutputNeurons[25:35]) != index_max(examples.outputs[ex])):
#                    err_one_network['high_order_10'] += 1

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
#                network['high_order_5'].train(network['first_order'].stateHiddenNeurons,
#                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            

        #add plot
        rms_plot['first_order'].append(sum_rms['first_order'])
        rms_plot['high_order_10'].append(sum_rms['high_order_10'])
        rms_plot['high_order_1'].append(sum_rms['high_order_1'])
        rms_plot['high_order_2'].append(sum_rms['high_order_2'])
        rms_plot['high_order_3'].append(sum_rms['high_order_3'])

        err_plot['first_order'].append(err_one_network['first_order'] / (nbTry * nbr_network))
#        err_plot['high_order_10'].append(err_one_network['high_order_10'] / (len(examples.inputs) * nbr_network))
#        err_plot['high_order_5'].append(err_one_network['high_order_5'] / (len(examples.inputs) * nbr_network))
        
#        print(err_plot['first_order'])
        print(epoch, " rms :", rms_plot['first_order'][epoch], " err : ", err_plot['first_order'][epoch])

    # divided by the maximum error
    max_err = (max(rms_plot['first_order']),
               max(rms_plot['high_order_10']),
               max(rms_plot['high_order_1']),
               max(rms_plot['high_order_2']),
               max(rms_plot['high_order_3'])
               )
#               ,
#               max(rms_plot['high_order_5']))
    for i in range(nbEpoch):
        rms_plot['first_order'][i] /= max_err[0]
        rms_plot['high_order_10'][i] /= max_err[1]
        rms_plot['high_order_1'][i] /= max_err[2]
        rms_plot['high_order_2'][i] /= max_err[3]
        rms_plot['high_order_3'][i] /= max_err[4]
#        rms_plot['high_order_5'][i] /= max_err[2]
    
    #displays rms
    plt.plot(display_interval, [rms_plot['first_order'][i] for i in display_interval],
             label="first-order network", linewidth=2)
    
    plt.plot(display_interval, [rms_plot['high_order_10'][i] for i in display_interval],
             label="high-order network (10 hidden units)", linewidth=2)
    
    plt.plot(display_interval, [rms_plot['high_order_1'][i] for i in display_interval],
             label="inputs")
        
    plt.plot(display_interval, [rms_plot['high_order_2'][i] for i in display_interval],
             label="hidden")
            
    plt.plot(display_interval, [rms_plot['high_order_3'][i] for i in display_interval],
             label="outputs")
    
    plt.plot(display_interval2, [err_plot['first_order'][i] for i in display_interval2],
             label="error first-order",
             linewidth=2)
    
#    plt.plot(display_interval, [rms_plot['high_order_5'][i] for i in display_interval],
#             label="high-order network (5 hidden units)")
    
    plt.title('Error proportion (RMS) of first-order and high-order networks')
    plt.ylabel('ERROR RMS')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    #displays errors
#    plt.plot(display_interval2, [err_plot['first_order'][i] for i in display_interval2],
#             label="first-order network")
#    
#    plt.plot(display_interval2, [err_plot['high_order_10'][i] for i in display_interval2],
#             label="high-order network (10 hidden units)")
#    
#    plt.plot(display_interval2, [err_plot['high_order_5'][i] for i in display_interval2],
#             label="high-order network (5 hidden units)")
#    
#    plt.title('Error proportion of first-order and high-order networks')
#    plt.ylabel('ERROR')
#    plt.xlabel("EPOCHS")
#    plt.axis((0, nbEpoch, 0, 1.))
#    plt.legend(loc='best', frameon=False)
#    plt.show()