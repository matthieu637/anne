# -*- coding: UTF-8 -*-
'''
Created on 13 March 2012

@author: Matthieu Zimmer

Article implementation
'''

from multilayerp import MultilayerPerceptron
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile
from utils import index_max
from articles.digit_reco.discretize import discretis


if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbEpoch = 1000
    nbTry = 10
    nbDiscre = 2
    
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 16 * 4, 10, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(16 * 4, 16 * 4 * 2, 16 * 16 + 16 * 4 + 10, learning_rate=lrate, momentum=momentum, grid=mode)

        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10}

    #create inputs/outputs to learn
    examples = DataFile("digit_handwritten_16.txt", mode)

    dis = [[0 for _ in range(nbEpoch)] for _ in range(10)]
    dis2 = [[0 for _ in range(nbEpoch)] for _ in range(10)]
    div = [[0 for _ in range(nbEpoch)] for _ in range(10)]
    valid = [[] for _ in range(10)]

    #learning
    for epoch in range(nbEpoch):
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                #RMS
                network['first_order'].calc_output(
                                            examples.inputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                
                network['high_order_10'].calc_output(
                                            network['first_order'].stateHiddenNeurons)

                im = index_max(examples.outputs[ex])
                div[im][epoch] += 1
                dis[im][epoch] += discretis(network['first_order'].stateHiddenNeurons, nbDiscre)
                dis2[im][epoch] += index_max(network['first_order'].stateOutputNeurons)
                
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
    
    colors = [(0.2, 0.8, 0.88), 'b', 'g', 'r', 'c', 'm', 'y', 'k', (0.8, 0.1, 0.8), (0.8, 0.2, 0.5)]
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
    
    
    for j in range(10):
        plt.plot(valid[j], [dis[j][k] for k in valid[j]], '.', color=colors[j])

    plt.show()
    
    
    
    plt.title('Discretize hidden layer')
    plt.ylabel('DISCRETIZED VALUE')
    plt.xlabel("EPOCHS")
    
    for j in range(10):
        plt.title('shape :%d' % j)
        plt.plot(valid[j], [dis[j][k] for k in valid[j]],  '.', label="hidden")
#        plt.plot(valid[j], [dis2[j][k]*(nbDiscre**(16*4))/10 for k in valid[j]], '.', label="output")
        
        plt.legend(loc='best', frameon=False)
        plt.show()
