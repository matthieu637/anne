# -*- coding: UTF-8 -*-
'''
Created on 21 fevr. 2012

@author: matthieu637

Article test
'''

from network import MultilayerNetwork
from utils import findMax
from random import shuffle
from math import sqrt
import matplotlib.pyplot as plt
from data import DataFile
from functools import reduce

if __name__ == '__main__':
    mode = MultilayerNetwork.R1to1
    nbr_network = 1
    momentum = 0.5
    
    mn = [MultilayerNetwork(7, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode) for _ in range(nbr_network)]    
    mn2 = [MultilayerNetwork(100, 100, 2, learning_rate=0.1, momentum=momentum, grid=mode) for _ in range(nbr_network)]
    mn3 = [MultilayerNetwork(100, 100, 2, learning_rate=10e-7, momentum=momentum, grid=mode) for _ in range(nbr_network)]
    for n in mn + mn2 + mn3:
        n.init_random_weights(-1,1)


    #create example
    examples = DataFile("../data/digital_shape.txt", mode)


    nbEpoch = 201
    y_plot = [[] for _ in range(3)]

    #learning
    for epoch in range(nbEpoch):
        sum_rms = 0.
        sum_rms2 = 0. 
        sum_rms3 = 0. 
        for network in range(nbr_network):
#            for ex in range(10):
#            for ex in np.random.randint(0, 10, 10):
            l_exx = range(10)
            shuffle(list(l_exx))
            for ex in l_exx:
                cell = [0, 0]
                if findMax(mn[network].calc_output(examples.inputs[ex])) == findMax(examples.outputs[ex]):
                    cell = [mode, 1]
                else:
                    cell = [1, mode]  
                
                rms = reduce(lambda x, y_plot:x + y_plot, map(lambda x, y_plot: pow(x - y_plot, 2), \
                    mn[network].calc_output(examples.inputs[ex]), examples.outputs[ex]))
                sum_rms += sqrt(rms / 10)
                
                rms2 = reduce(lambda x, y_plot:x + y_plot, map(lambda x, y_plot: pow(x - y_plot, 2), \
                    mn2[network].calc_output(mn[network].stateHiddenNeurons), cell))
                sum_rms2 += sqrt(rms2 / 2)
                
                rms3 = reduce(lambda x, y_plot:x + y_plot, map(lambda x, y_plot: pow(x - y_plot, 2), \
                    mn3[network].calc_output(mn[network].stateHiddenNeurons), cell))
                sum_rms3 += sqrt(rms3 / 2)
                
                mn2[network].train(mn[network].stateHiddenNeurons, cell)
                mn3[network].train(mn[network].stateHiddenNeurons, cell)
                
                mn[network].train(examples.inputs[ex], examples.outputs[ex])


        y_plot[0].append(sum_rms)
        y_plot[1].append(sum_rms2)
        y_plot[2].append(sum_rms3)
        
    y_plot[0] = list(map(lambda x : x / max(y_plot[0]), y_plot[0]))
    y_plot[1] = list(map(lambda x : x / max(y_plot[1]), y_plot[1]))
    y_plot[2] = list(map(lambda x : x / max(y_plot[2]), y_plot[2]))
    
    plt.plot(range(201)[6::5] , y_plot[0][6::5], label="first-order network")
    plt.plot(range(201)[6::5] , y_plot[1][6::5], label="high-order network (high learning rate)")
    plt.plot(range(201)[6::5] , y_plot[2][6::5], label="high-order network (low learning rate)")
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend()
    plt.show()
    

    #testing
    for ex in range(10):
        for network in mn:
            print(examples.inputs[ex])
            print(network.calc_output(examples.inputs[ex]))
            print(findMax(network.calc_output(examples.inputs[ex])))
