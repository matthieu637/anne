'''
Created on 7 mars 2012

@author: matthieu637

Iowa Gambling Task
'''

from network import MultilayerNetwork
from random import random
from utils import index_max, randmm
import matplotlib.pyplot as plt


def deck(nbr):
    if(nbr in (0, 2)):
        return (True, random() < 0.7)
    else: 
        return (False, random() < 0.3)

if __name__ == '__main__':
    
    plt.title("low")
    grid = MultilayerNetwork.R0to1
    nbr_network = 30
    nbr_epoch = 200
    nbr_trial = 10
    
    first_order = []
    high_order = []
    for _ in range(nbr_network):
        fo = MultilayerNetwork(5, 40, 4, grid, 0.002, 0., 1., False, True)
        fo.init_random_weights(-1, 1)
        first_order.append(fo)
        
        #0.0003 or 0.015
        ho = MultilayerNetwork(40, 40, 2, grid, 0.0003, 0., 1., False, True)
        ho.init_random_weights(-1, 1)
        high_order.append(ho)
    
    
    #trials
    wins = []
    wagers = []
    
    last_output = [ [0 for _ in range(5)] for _ in range(nbr_network)]
    for epoch in range(nbr_epoch):
        nbr_win = 0
        nbr_gwager = 0
        for net in range(nbr_network):
            for trial in range(nbr_trial):
                for i in range(len(last_output[net])):
                    last_output[net][i] += randmm(0.01, 0.03)
        
                bet = index_max(first_order[net].calc_output(last_output[net]))
                
                for i in range(len(first_order[net].stateHiddenNeurons)):
                        first_order[net].stateHiddenNeurons[i] += randmm(0.01, 0.03)
                
                wager = index_max(high_order[net].calc_output(first_order[net].stateHiddenNeurons))
                
                win = deck(bet)
                
                if((win[0] and wager == 0) ):
                        nbr_gwager += 1
                
                if(win[0]):                    
                    high_order[net].train(first_order[net].stateHiddenNeurons, [1. , grid])
                    
                    outputs = [grid for _ in range(4)] + [1.]
                    outputs[bet] = 1.
                    first_order[net].train(outputs, outputs)
                    last_output[net] = outputs
                    
                    nbr_win += 1
                else:
                    high_order[net].train(first_order[net].stateHiddenNeurons, [grid , 1.])
                    
                    inputs = [grid for _ in range(4)] + [grid]
                    inputs[bet] = 1.
                    outputs = [1. for _ in range(4)]
                    outputs[bet] = grid
                    first_order[net].train(inputs, outputs)
                    last_output[net] = inputs
                    
        
        print(epoch)
        wins.append(nbr_win / (nbr_network * nbr_trial))
        wagers.append(nbr_gwager / (nbr_network * nbr_trial))

    plt.plot(wins,
             label="first-order network")
    plt.plot(wagers,
             label="high-order network")
    plt.axis((0, nbr_epoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
