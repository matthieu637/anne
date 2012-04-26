# -*- coding: UTF-8 -*-
'''
Created on 13 March 2012

@author: Matthieu Zimmer

Article implementation
'''

from multilayerp import MultilayerPerceptron, MultilayerPerceptronM
from perceptron import PerceptronR0to1
from random import shuffle, seed
from utils import index_max
import matplotlib.pyplot as plt
from data import DataFile, DataFileR



if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    lrate = 0.15
    nbEpoch = 800
    nbTry = 20
    display_interval = range(nbEpoch)[::6]
    
    
        #create inputs/outputs to learn
    examples = DataFileR("iris.txt", mode)
#    examples = DataFile("digit_handwritten_16.txt", mode)

    nbInputs = len(examples.inputs[0])
    nbOutputs = len(examples.outputs[0])
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        seed(i)
        first_order = [PerceptronR0to1(nbInputs, learning_rate=lrate, momentum=momentum,
                 temperature=1., init_w_randomly=True, enable_bias=True) for _ in range(nbOutputs)]
#        high_order_10 = MultilayerPerceptron(nbInputs, 3 + nbInputs // 4, nbOutputs, learning_rate=lrate, momentum=momentum, grid=mode)
#        high_order_10.init_weights_randomly(-1, 1)
        
        high_order_10 = MultilayerPerceptronM(nbInputs, 3 + nbInputs // 4, nbOutputs, 1, learning_rate=lrate, momentum=momentum, grid=mode)
        

        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}
        
    #3 curves
    err_plot = {'first_order' : [] ,
              'high_order_10' : []}

    seed(100)

    #learning
    for epoch in range(nbEpoch):
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.}
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                resf1 = [network['first_order'][i].calc_output(examples.inputs[ex]) 
                         for i in range(nbOutputs)]
                network['high_order_10'].calc_output(examples.inputs[ex])

                if(index_max(resf1) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
                    
                if(index_max(network['high_order_10'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['high_order_10'] += 1


                network['high_order_10'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                for i in range(nbOutputs):
                    network['first_order'][i].train(examples.inputs[ex],
                                             examples.outputs[ex][i])
            

        #add plot
        err_plot['first_order'].append(err_one_network['first_order'] / (nbTry * nbr_network))
        err_plot['high_order_10'].append(err_one_network['high_order_10'] / (nbTry * nbr_network))
        
        print(epoch, " err : ", err_plot['first_order'][epoch])
        
    
    #displays errors
    plt.plot(display_interval, [err_plot['first_order'][i] for i in display_interval],
             label="perceptron",
             linewidth=2)
    
    plt.plot(display_interval, [err_plot['high_order_10'][i] for i in display_interval],
             label="multi-layer perceptron", linewidth=2)
    
    plt.title('Error ratio of first-order and high-order networks')
    plt.ylabel('ERROR RATIO')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()
