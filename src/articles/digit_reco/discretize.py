# -*- coding: UTF-8 -*-
'''
Created on 22 March 2012

@author: Matthieu Zimmer

Article implementation
'''

from multilayerp import MultilayerPerceptron
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile
from mpl_toolkits.mplot3d import Axes3D
from utils import index_max


def nbdis(nb):
    if(nb <= 0.25):
        return 0
    elif(nb <= 0.5):
        return 1
    elif(nb <= 0.75):
        return 2
    else:
        return 3
    
def discretis(ll):
    s = 0
    for i in range(5):
        s += (4**i)*nbdis(ll[i])
    return s

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbShape = 10
    nbEpoch = 1000
    display_interval = range(nbEpoch)[::6]
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(20, 5, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(5, 10, 35, learning_rate=lrate, momentum=momentum, grid=mode)
        
        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}

    #create inputs/outputs to learn
    examples = DataFile("digit_shape.txt", mode)


    #3 curves
    dis = [[0 for _ in range(nbEpoch)] for _ in range(nbShape)]
    div = [[0 for _ in range(nbEpoch)] for _ in range(nbShape)]
    valid = [[] for _ in range(10)]

    #learning
    for epoch in range(nbEpoch):
        for network in networks:
            l_exx = list(range(nbShape))
            shuffle(l_exx)
            for ex in l_exx:              
                #RMS
                network['first_order'].calc_ME(
                                            examples.inputs[ex],
                                            examples.outputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                
                network['high_order_10'].calc_ME_range(
                                            network['first_order'].stateHiddenNeurons,
                                             entire_first_order, 25, 35)

                im = index_max(examples.outputs[ex])
                div[im][epoch] += 1
                dis[im][epoch] += discretis(network['first_order'].stateHiddenNeurons)
                
                if(len(valid[im]) == 0):
                    valid[im].append(epoch)
                elif(valid[im][len(valid[im])-1] != epoch):
                    valid[im].append(epoch)

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
            
        print(epoch)
    
    #divided by the number of networks
    for i in range(10):
        for j in range(nbEpoch):
            if(div[i][j] != 0):
                dis[i][j] /= div[i][j]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    colors =[(0.2, 0.8, 0.88), 'b', 'g', 'r', 'c', 'm', 'y', 'k', (0.8,0.1,0.8), (0.8,0.2,0.5)]
    #displays
    for j in range(10):
        ax.scatter([dis[j][k] for k in valid[j]], [j] * len(valid[j]), valid[j], color=colors[j], marker='x')

    ax.set_xlabel('DISCRETIZED VALUE')
    ax.set_ylabel('SHAPE')
    ax.set_zlabel('EPOCH')
    plt.show()

    
    plt.figure()
    plt.title('Discretize hidden layer')
    plt.ylabel('DISCRETIZED VALUE')
    plt.xlabel("EPOCHS")
    
    
    for j in range(nbShape):
        plt.plot(valid[j], [dis[j][k] for k in valid[j]], '.', color=colors[j])

    plt.show()
    