# -*- coding: UTF-8 -*-
'''
Created on 24 fevr. 2012

@author: matthieu637
'''

from data import DataFile
from random import shuffle
from network import MultilayerNetwork
from neuron import Neuron
from utils import findMax

if __name__ == '__main__':
    r = {'x':(0, 1),
         '.': (0,),
         '?': (0, 0.02),
         '!': (1,)}
    samples = DataFile("../data/blindslight.txt", rules=r)
    
    first_order = MultilayerNetwork(100, 60, 100, 0, 0.9, 0, 1., False, True)
    first_order.init_random_weights(-1, 1)
    
    high_order = [Neuron(100, 0.1, 0., 1., Neuron.Output, False, True) for _ in range(2)]
    high_order[0].init_random_weights(0, 0.1)
    high_order[1].init_random_weights(0, 0.1)

    #pre-training
    for epoch in range(150):
        l_exx = list(range(200))
        shuffle(l_exx)
        for ex in l_exx:
            first_order.train(samples.inputs[ex], samples.outputs[ex])
        
            compara = []
            for i in range(100):
                compara.append(samples.inputs[ex][i] - first_order.stateOutputNeurons[i])
            
            i = findMax(first_order.stateOutputNeurons)
            j = findMax(samples.inputs[ex])
            if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
                or(max(first_order.stateOutputNeurons) <= 0.5 and max(samples.inputs[ex]) <= 0.5)) :
                high_order[0].train(compara, 1)
                high_order[1].train(compara, 0)
            else:
                high_order[0].train(compara, 0)
                high_order[1].train(compara, 1)
            
        print(epoch)


    #testing
    #Suprathreshold
    
    pourc = {'hg_co' : 0,
             'hg_in' : 0,
             'lw_co' : 0,
             'lw_in' : 0}
    
    for ex in range(200):
        first_order.calc_output(samples.inputs[ex])
        
        compara = []
        for i in range(100):
            compara.append(samples.inputs[ex][i] - first_order.stateOutputNeurons[i])
            
        res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
        i = findMax(first_order.stateOutputNeurons)
        j = findMax(samples.inputs[ex])
        if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
                or(max(first_order.stateOutputNeurons) <= 0.5 and max(samples.inputs[ex]) <= 0.5)) :
#        if ((max(samples.outputs[ex]) > 0.5 and max(samples.inputs[ex]) > 0.5) \
#            or(max(samples.outputs[ex]) <= 0.5 and max(samples.inputs[ex]) <= 0.5)) :
            if findMax(res) == 0:
                pourc['hg_co'] += 1
            else:
                pourc['lw_in'] += 1
        else:
            if findMax(res) == 1:
                pourc['lw_co'] += 1
            else:
                pourc['hg_in'] += 1
            
    print(pourc)
    
