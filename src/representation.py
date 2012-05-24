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
from perceptron import PerceptronR0to1, Perceptron
from utils import index_max


def graph_network(neurons_top, neurons_down, width):
    
    fig = plt.figure()
    plt.clf()
    
    for i in range(len(neurons_top[0])):
        w = []
        for j in range(len(neurons_top)):
            w.append(build_matrice(neurons_top[j][i].weights, neurons_down[j]))
        r = [0. for _ in range(len(w[0]))]
        for l in w:
            for k in range(len(r)):
                r[k] += l[k]
        show_repr(r, width, fig, 250 + i, i)

    plt.show()



def build_matrice(weightl, nlist):
    weights = [0. for _ in range(len(nlist[0].weights))]
    
    for i in range(len(weights)):
        wsum = 0
        for j in range(len(nlist)):
            wsum += weightl[j] * (nlist[j].weights[i] - nlist[j].bias )
        weights[i] += wsum
    
    return weights

def show_repr(weightl, width, fig, num, title):
    matrice = [[] for i in range(len(weightl)//width)]
    
    vmax = max(weightl)
    vmin = min(weightl)
    
    for j in range(len(weightl)//width):
        for i in range(width):
            nw =  (weightl[j*width + i] - vmin) * (1./(vmax - vmin))
            matrice[j].append(round(nw, 2))
    
    ax = fig.add_subplot(num, title=title)
    ax.set_aspect(1)
    res = ax.imshow(np.array(matrice), cmap=cm.gist_gray_r, 
                    interpolation='nearest')
    
    plt.xticks([])
    plt.yticks([])

    
if __name__ == '__main__':    
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbr_try = 50
    nbr_epoch = 200
    
#   Data Sample Declaration
    def data():
        return DataFile("digit_handwritten_16.txt", mode)
#        return DataFile("digit_shape.txt", mode)
    
#   Network Declaration
    def FoN2(inputs, outputs):
        return MultilayerPerceptron(inputs, inputs//4, outputs, learning_rate=lrate, momentum=momentum, grid=mode)

    def FoN(inputs, outputs):
        return [PerceptronR0to1(inputs, learning_rate=0.2, momentum=0.3,
                 temperature=1., ntype=Perceptron.OUTPUT, init_w_randomly=True, enable_bias=True) for _ in range(outputs)]

#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN2'].calc_output(inputs)
        pass
        
    def step_statictics(simu, network, plot, inputs, outputs):
        #rms
        pass

    def step_learn(network, inputs, outputs):
        #Learning
        network['FoN2'].train(inputs, outputs)
        
        for i in range(len(outputs)):
            network['FoN'][i].train(inputs, outputs[i])
        
        
    sim = Simulation(data, nbr_network, [FoN, FoN2])
    sim.dgraph(['FoN_rms'])
    sim.launch(nbr_epoch, nbr_try, step_propagation, step_statictics, step_learn)
    
    #show_repr(sim.networks[0]['FoN'].hiddenNeurons[0].weights, 4)

#    for i in range(10):
#        build_matrice(sim.networks[0]['FoN'].outputNeurons[i].weights, sim.networks[0]['FoN'].hiddenNeurons, 4)

    width = 16

#    MLP
    on = []
    hn = []
    for net in sim.networks:
        on.append(net['FoN2'].outputNeurons)
        hn.append(net['FoN2'].hiddenNeurons)
    graph_network(on, hn, width)

#    Perceptron
    fig = plt.figure()
    plt.clf()
    
    for i in range(10):
        show_repr(sim.networks[0]['FoN'][i].weights, width, fig, 250 + i, i)
    
    plt.show()
