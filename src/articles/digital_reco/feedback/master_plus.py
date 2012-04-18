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


if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    seed(50)
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        first_order.init_weights_randomly(-1, 1)
        high_order_h = [PRenforcement(100, 0.4, 0.04, 2., True) for _ in range(20)]
        
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
                        for i in range(20)]
                
                f_success = False
                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                    f_success = True
                    
                if(index_max(network['control'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['control'] += 1
                
                il = last_index_max(res[0:10]) if res[0:10] != ([0]*10) else 0
                im = last_index_max(res[10:20]) if res[10:20] != ([0]*10) else 0
                
                perfo['wager_proportion'] += il*0.1
                perfo['high_order_h'] += im*0.1

                #learn
                network['control'].train(examples.inputs[ex], examples.outputs[ex])
                
#                for i in range(20):
#                    [ network['high_order_h'][i].calc_output(network['first_order'].stateHiddenNeurons) for i in range(20) ]

#                [ network['high_order_h'][i].train(f_success) for i in range(20)]
                
#                for i in range(0, il):
#                    network['high_order_h'][il].train(f_success)
                
                if(il < 9):
                    for _ in range(5):
                        network['high_order_h'][il+1].calc_output(network['first_order'].stateHiddenNeurons)
                        network['high_order_h'][il+1].train(f_success)
                
                for _ in range(5):
                    network['high_order_h'][il].calc_output(network['first_order'].stateHiddenNeurons)
                    network['high_order_h'][il].train(f_success)
                
                
#                network['high_order_h'][im].train(f_success)

                print(res[0:10])
                network['first_order'].set_learning_rate(il*0.1)
                network['first_order'].set_momentum(im*0.1)
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
