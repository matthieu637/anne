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
        return random() < 0.7
    else: 
        return random() < 0.3

if __name__ == '__main__':
    
    grid = MultilayerNetwork.R0to1
    nbr_network = 15
    nbr_epoch = 200
    nbr_trial = 10
    display_interval = range(nbr_epoch)[::5]
    
    first_order = []
    for _ in range(nbr_network):
        fo = MultilayerNetwork(5, 40, 4, grid, 0.2, 0., 1., False, True)
        fo.init_random_weights(-1, 1)
        first_order.append(fo)
    
    
    
    #trials
    wins = []
    
    last_output = [ [randmm(0, 0.02) for _ in range(5)] for _ in range(nbr_network)]
    for epoch in range(nbr_epoch):
        nbr_win = 0
        for net in range(nbr_network):
            for trial in range(nbr_trial):
        #        print(last_output)
#                for i in range(len(last_output[net])):
#                    last_output[net][i] += randmm(0, 0.02)
        
                bet = index_max(first_order[net].calc_output(last_output[net]))
#                print(bet)
                
                win = deck(bet)
                
                if(win):
                    outputs = [0. for _ in range(4)] + [1.]
                    outputs[bet] = 1.
                    first_order[net].train(outputs, outputs)
                    last_output[net] = outputs
                    
                    nbr_win += 1
                else:
                    inputs = [0. for _ in range(4)] + [0.]
                    inputs[bet] = 1.
                    outputs = [1. for _ in range(4)]
                    outputs[bet] = 0.
                    first_order[net].train(inputs, outputs)
                    last_output[net] = inputs
        
        
        wins.append(nbr_win / (nbr_network * nbr_trial))

    plt.plot(display_interval, [wins[i] for i in display_interval],
             label="first-order network")
    plt.axis((0, nbr_epoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
