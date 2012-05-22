# -*- coding: UTF-8 -*-
'''
Created on 22 May 2012

@author: Matthieu Zimmer

'''
from multilayerp import MultilayerPerceptron
from data import DataFile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from simulation import Simulation
import numpy as np

def build_matrice(weightl, nlist, width):
    weights = [0. for _ in range(len(nlist[0].weights))]
    
    for i in range(len(weights)):
        wsum = 0
        for j in range(len(weightl)):
            wsum += nlist[j].weights[i] * weightl[j]
        weights[i] += wsum
    
    show_repr(weights, width)

def show_repr(weightl, width):
    matrice = [[] for i in range(len(weightl)//width)]
    
    vmax = max(weightl)
    vmin = min(weightl)
    
    for j in range(len(weightl)//width):
        for i in range(width):
            nw = (weightl[j*width + i] - vmin) * (1./(vmax - vmin))
            matrice[j].append(round(nw, 2))
            
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(matrice), cmap=cm.gist_gray, 
                    interpolation='nearest')
    
    width = len(matrice)
    height = len(matrice[0])
    
    for x in range(width):
        for y in range(height):
            ax.annotate(str(matrice[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    cb = fig.colorbar(res)
    alphabet = '012345'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])

    
    plt.show()

    
    pass
    
    
if __name__ == '__main__':    
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbr_try = 10
    nbr_epoch = 1000
    
#   Data Sample Declaration
    def data():
#        return DataFile("digit_handwritten_16.txt", mode)
        return DataFile("digit_shape.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        return MultilayerPerceptron(inputs, inputs//4, outputs, learning_rate=lrate, momentum=momentum, grid=mode)

#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        #rms
        pass

    def step_learn(network, inputs, outputs):
        #Learning
        network['FoN'].train(inputs, outputs)
        
    sim = Simulation(data, nbr_network, [FoN])
    sim.dgraph(['FoN_rms'])
    sim.launch(nbr_epoch, nbr_try, step_propagation, step_statictics, step_learn)
    
    #show_repr(sim.networks[0]['FoN'].hiddenNeurons[0].weights, 4)
    build_matrice(sim.networks[0]['FoN'].outputNeurons[2].weights, sim.networks[0]['FoN'].hiddenNeurons, 4)
