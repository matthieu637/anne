# -*- coding: UTF-8 -*-
'''
Created on 21 fevr. 2012

@author: matthieu637

Article test
'''

from __future__ import division
from network import MultilayerNetwork
from utils import findMax
from random import shuffle
from math import sqrt
import matplotlib.pyplot as plt
from data import DataFile

if __name__ == '__main__':
    mode = MultilayerNetwork.R0to1
    nbr_network = 5
    momentum = 0.9
    
    mn = [MultilayerNetwork(20, 5, 10, learning_rate=0.1, momentum=momentum, grid=mode) for _ in range(nbr_network)]
    mn2 = [MultilayerNetwork(5, 10, 35, learning_rate=0.1, momentum=momentum, grid=mode) for _ in range(nbr_network)]
    mn3 = [MultilayerNetwork(5, 5, 35, learning_rate=0.1, momentum=momentum, grid=mode) for _ in range(nbr_network)]
    

    #create example
    examples = DataFile("../data/digit_shape.txt", mode)


    nbEpoch = 1000
    y = [[] for _ in range(3)]
    max_err = 0
    max_err2 = 0
    max_err3 = 0

    #learning
    for epoch in range(nbEpoch):
        sum_rms = 0.
        sum_rms2 = 0. 
        sum_rms3 = 0. 
        for network in range(nbr_network):
#            for ex in range(10):
#            for ex in np.random.randint(0, 10, 10):
            l_exx = range(10)
            shuffle(l_exx)
            for ex in l_exx:
                rms = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y, 2), \
                    mn[network].calc_output(examples.inputs[ex]), examples.outputs[ex]))
                rms = sqrt(rms / 10)
                sum_rms += rms
                
                rms2 = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y, 2), \
                    mn2[network].calc_output(mn[network].stateHiddenNeurons), examples.inputs[ex] + \
                    mn[network].stateHiddenNeurons + mn[network].stateOutputNeurons))
                rms2 += sqrt(rms2 / 10)
                sum_rms2 += rms2
                
                rms3 = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y, 2), \
                    mn3[network].calc_output(mn[network].stateHiddenNeurons), examples.inputs[ex] + \
                    mn[network].stateHiddenNeurons + mn[network].stateOutputNeurons))
                rms3 += sqrt(rms3 / 10)
                sum_rms3 += rms3
    
                mn[network].train(examples.inputs[ex], examples.outputs[ex])
                mn2[network].train(mn[network].stateHiddenNeurons, examples.inputs[ex] + \
                    mn[network].stateHiddenNeurons + mn[network].stateOutputNeurons)
                mn3[network].train(mn[network].stateHiddenNeurons, examples.inputs[ex] + \
                    mn[network].stateHiddenNeurons + mn[network].stateOutputNeurons)

            
        if sum_rms > max_err:
            max_err = sum_rms
        if sum_rms2 > max_err2:
            max_err2 = sum_rms2
        if sum_rms3 > max_err3:
            max_err3 = sum_rms3
        
        y[0].append(sum_rms)
        y[1].append(sum_rms2)
        y[2].append(sum_rms3)

    y[0] = map(lambda x : x / max_err, y[0])
    y[1] = map(lambda x : x / max_err2, y[1])
    y[2] = map(lambda x : x / max_err3, y[2])
    
    yy = [y[0][0]] + [y[0][25]] + [y[0][50]] + [y[0][100]] + [y[0][200]] + [y[0][500]] + [y[0][999]]
    yyy = [y[1][0]] + [y[1][25]] + [y[1][50]] + [y[1][100]] + [y[1][200]] + [y[1][500]] + [y[1][999]]
    yyyy = [y[2][0]] + [y[2][25]] + [y[2][50]] + [y[2][100]] + [y[2][200]] + [y[2][500]] + [y[2][999]]

    plt.plot([0, 25, 50, 100, 200, 500, 999], yy, label="first-order network")
    plt.plot([0, 25, 50, 100, 200, 500, 999], yyy, label="high-order network (10 hidden units)")
    plt.plot([0, 25, 50, 100, 200, 500, 999], yyyy, label="high-order network (5 hidden units)")
    plt.plot(y[0], label="line ")
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend()
    plt.show()
    

    #testing
    for ex in range(10):
        for network in mn:
            print examples.inputs[ex]
            print network.calc_output(examples.inputs[ex])
            print findMax(network.calc_output(examples.inputs[ex]))
