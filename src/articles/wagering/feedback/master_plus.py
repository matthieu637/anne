# -*- coding: UTF-8 -*-
'''
Created on 12 April 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from prenforcement import PRenforcement
from utils import index_max, last_index_max
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile
from copy import deepcopy
from random import seed


def l_to_lr(l):
    w = 0.
    for i in range(3):
        w += l[i]*(2**i)
    return w

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.5
    nbEpoch = 201
    lrate = 0.15
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    seed(50)
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 50, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        first_order.init_weights_randomly(-1, 1)
        high_order_h = [PRenforcement(50, 0.4, 0.04, 2., True) for _ in range(6)]
        
        control = deepcopy(first_order)
        
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h,
                    'control':control}
        


    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'control' : []}
    
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'control' : 0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(examples.inputs[ex])
                network['control'].calc_output(examples.inputs[ex])
                res =  [ network['high_order_h'][i].calc_output(network['first_order'].stateHiddenNeurons) 
                        for i in range(6)]
                
                f_success = False
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                    f_success = True
                    
                if(index_max(network['control'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['control'] += 1
                
                l = l_to_lr(res[0:3]) * 9.5 / 150
                m = l_to_lr(res[3:6]) * 9.5 / 150
            
                #learn
                network['control'].train(examples.inputs[ex], examples.outputs[ex])
                
                
#                [ network['high_order_h'][i].train(f_success) for i in range(6)]

                if(epoch < nbEpoch/2):
                    [ network['high_order_h'][i].train(f_success) for i in range(3)]
                    network['first_order'].set_learning_rate(l)
                    network['first_order'].set_momentum(m)
                    perfo['wager_proportion'] += l
                    perfo['high_order_h'] += momentum
                else :
                    [ network['high_order_h'][i].train(f_success) for i in range(3,6)]
                    network['first_order'].set_learning_rate(lrate)
                    network['first_order'].set_momentum(m)
                    perfo['wager_proportion'] += lrate
                    perfo['high_order_h'] += m
                    

#                network['first_order'].set_learning_rate(l)
#                network['first_order'].set_momentum(m)
                network['first_order'].train(examples.inputs[ex],
                                         examples.outputs[ex])
                
        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)


    plt.title("Performance of first-order and higher-order networks with feedback ( Master without)")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="momentum")
    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="learning rate")
    plt.plot(display_interval , y_perfo['control'][3::5], label="control", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    

    
#    plb.hist
#    plb.show()
