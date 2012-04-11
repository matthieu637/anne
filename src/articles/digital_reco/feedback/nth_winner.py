# -*- coding: UTF-8 -*-
'''
Created on 19 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from utils import index_max, index_max_nth
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile
from random import random
from perceptron import PerceptronR0to1, Perceptron

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    nbEpoch = 200
    nbTry = 50
    display_interval = range(nbEpoch)[1::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        high_order_h = MultilayerPerceptron(100, 100, 10, learning_rate=0.1, momentum=0.5, grid=mode)
#        high_order_h = [PerceptronR0to1(100, learning_rate=0.1, momentum=0.,
#                 temperature=1., ntype=Perceptron.OUTPUT, init_w_randomly=True, enable_bias=True)
#                        for _ in range(10)]
        
        
        first_order.init_weights_randomly(-1, 1)
        high_order_h.init_weights_randomly(-1, 1)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'osti':[],
              'feedback' : []}
    
    stats = []
    
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'feedback' : 0.,
                 'osti':0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(examples.inputs[ex])
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
                
                cell = [0 for _ in range(10)]
                

                for k in range(10):
                    if index_max_nth(network['first_order'].stateOutputNeurons, k) == index_max(examples.outputs[ex]):
                        cell[k] = 1
                        break

                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1

#                res = [ network['high_order_h'][i].calc_output(network['first_order'].stateHiddenNeurons) for i in range(10)]
                res = network['high_order_h'].stateOutputNeurons

                if(index_max(res) == 0):
                    perfo['wager_proportion'] += 1
                
                if(cell == [0 for _ in range(10)]):
                    print(network['first_order'].stateOutputNeurons)
                    for k in range(10):
                        print(index_max_nth(network['first_order'].stateOutputNeurons, k), index_max(examples.outputs[ex]))
                    
                    
                    print("here", epoch, ex)
                    print(cell)
                    exit()
                
                if(index_max(cell) == index_max(res)):
                    perfo['high_order_h'] += 1
                    for k in range(10):
                        if(index_max(res) == k and index_max_nth(network['first_order'].stateOutputNeurons, k) == index_max(examples.outputs[ex])):
                            perfo['feedback'] += 1
                            break
                                
                
                stats.append(index_max(cell))
                
                #learn
                
#                for i in range(10):
#                    network['high_order_h'][i].train(network['first_order'].stateHiddenNeurons
#                                                     , cell[i])
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    
    iu = 0
    for i in range(10):
        t = 0
        for j in range(len(stats)):
            if(i == stats[j]):
                t += 1
        if(i == 0):
            iu = t
        print(i , ' -> ', t / len(stats))
        
    
    print()
    for i in range(10):
        t = 0
        for j in range(len(stats)):
            if(i == stats[j]):
                t += 1
        print(i , ' -> ', t / (len(stats) - iu))

    plt.title("Performance of first-order and higher-order networks with feedback ( nth W-T-A )")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
