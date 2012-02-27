# -*- coding: UTF-8 -*-
'''
Created on 24 fevr. 2012

@author: matthieu637
'''

from data import DataFile
from random import shuffle
from network import MultilayerNetwork
from neuron import Neuron, NeuronR0to1
from utils import index_max

if __name__ == '__main__':
    r = {'x':(0, 1),
         '.': (0,),
         '?': (0, 0.02),
         '!': (1,)}
    samples = DataFile("../data/blindslight.txt", rules=r)
    
    first_order = MultilayerNetwork(100, 60, 100, 0, 0.9, 0, 1., False, True)
    first_order.init_random_weights(-1, 1)
    
    high_order = [NeuronR0to1(100, 0.1, 0., 1., Neuron.Output, False, True) for _ in range(2)]
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
            
            i = index_max(first_order.stateOutputNeurons)
            j = index_max(samples.inputs[ex])
            if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.inputs[ex][i] <= 0.5)) :
#            i = index_max(samples.outputs[ex])
#            j = index_max(samples.inputs[ex])
#            if ((samples.outputs[ex][i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
#                    or(samples.outputs[ex][i] <= 0.5 and samples.inputs[ex][i] <= 0.5)) :
                high_order[0].train(compara, 1)
                high_order[1].train(compara, 0)
            else:
                high_order[0].train(compara, 0)
                high_order[1].train(compara, 1)
            
        print(epoch)


    #testing
    #Suprathreshold stimuli
    
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
            
        i = index_max(first_order.stateOutputNeurons)
        j = index_max(samples.inputs[ex])
        if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.inputs[ex][j] <= 0.5)) :
#        
#        i = index_max(samples.outputs[ex])
#        j = index_max(samples.inputs[ex])
#        if ((samples.outputs[ex][i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
#                or(samples.outputs[ex][i] <= 0.5 and samples.inputs[ex][i] <= 0.5)) :
            if index_max(res) == 0:
                pourc['hg_co'] += 1
            else:
                pourc['lw_in'] += 1
        else:
            if index_max(res) == 1:
                pourc['lw_co'] += 1
            else:
                pourc['hg_in'] += 1
            
    print(pourc)
    

    #Subthreshold stimuli
    
    #add +0.0012
    print(samples.inputs[0])
    for input in samples.inputs:
        i_fix = index_max(input)
        for i in range(len(input)):
            if(i != i_fix):
                input[i]+= 0.0012 
        
    print(samples.inputs[0])
    
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
            
        i = index_max(first_order.stateOutputNeurons)
        j = index_max(samples.inputs[ex])
        if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.inputs[ex][j] <= 0.5)) :
        
#        i = index_max(samples.outputs[ex])
#        j = index_max(samples.inputs[ex])
#        if ((samples.outputs[ex][i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
#                or(samples.outputs[ex][i] <= 0.5 and samples.inputs[ex][i] <= 0.5)) :
            if index_max(res) == 0:
                pourc['hg_co'] += 1
            else:
                pourc['lw_in'] += 1
        else:
            if index_max(res) == 0:
                pourc['hg_in'] += 1
            else:
                pourc['lw_co'] += 1
            

    print(pourc)
    print('first')
    
    