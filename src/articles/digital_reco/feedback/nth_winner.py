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

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 3
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        high_order_h = MultilayerPerceptron(100, 20, 5, learning_rate=0.1, momentum=0., grid=mode)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h}

    for network in networks:
        for k in network.keys():
            if(k == 'first_order'):
                network[k].init_weights_randomly(-1, 1)

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'feedback' : []}
    
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'feedback' : 0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(examples.inputs[ex])
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
                
                cell = [0 for _ in range(5)]
                
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    cell[0] = 1
                elif(index_max_nth(network['first_order'].stateOutputNeurons, 1), index_max(examples.outputs[ex])):
                    cell[1] = 1
                elif(index_max_nth(network['first_order'].stateOutputNeurons, 2), index_max(examples.outputs[ex])):
                    cell[2] = 1
                elif(index_max_nth(network['first_order'].stateOutputNeurons, 3), index_max(examples.outputs[ex])):
                    cell[3] = 1
                else:
                    cell[4] = 1


                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
#                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max_nth(examples.outputs[ex],index_max(cell))):
#                    perfo['high_order_h'] += 1
                    
                
                
                if(index_max(cell) == 0):
                    if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                        perfo['feedback'] += 1
                    if(index_max(network['high_order_h'].stateOutputNeurons) == 0):
                        perfo['high_order_h'] += 1
                elif(index_max(cell) == 1) :
                    if(index_max_nth(network['first_order'].stateOutputNeurons, 1) == index_max(examples.outputs[ex])):
                        perfo['feedback'] += 1
                    if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                        perfo['high_order_h'] += 1
                elif(index_max(cell) == 2) :
                    if(index_max_nth(network['first_order'].stateOutputNeurons, 2) == index_max(examples.outputs[ex])):
                        perfo['feedback'] += 1
                    if(index_max(network['high_order_h'].stateOutputNeurons) == 2):
                        perfo['high_order_h'] += 1 
                elif(index_max(cell) == 3) :
                    if(index_max_nth(network['first_order'].stateOutputNeurons, 3) == index_max(examples.outputs[ex])):
                        perfo['feedback'] += 1
                    if(index_max(network['high_order_h'].stateOutputNeurons) == 3):
                        perfo['high_order_h'] += 1
                else :
                    if(index_max_nth(network['first_order'].stateOutputNeurons, 4) == index_max(examples.outputs[ex])):
                        perfo['feedback'] += 1
                    if(index_max(network['high_order_h'].stateOutputNeurons) == 4):
                        perfo['high_order_h'] += 1
                    
#                if(index_max(network['high_order_h'].stateOutputNeurons) == 0):
#                    perfo['wager_proportion'] += 1
                
                #learn
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    

    plt.title("Performance of first-order and higher-order networks with feedback ( nth W-T-A )")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
#    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
