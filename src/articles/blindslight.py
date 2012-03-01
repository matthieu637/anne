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
import matplotlib.pyplot as plt

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

#    print(samples.inputs[101])
#    print(samples.outputs[101])

    err = 0.
    the = 0.
    rms = []
    all_cmp = []
    
    #pre-training
    for epoch in range(30):
        l_exx = list(range(200))
        shuffle(l_exx)
        rms_ss = 0.
        for ex in l_exx:
#        for ex in l_exx[1:150]:
            first_order.train(samples.inputs[ex], samples.outputs[ex])
#            first_order.calc_output(samples.inputs[ex])
        
            compara = []
            for i in range(100):
                compara.append((samples.inputs[ex][i] - first_order.stateOutputNeurons[i] ))
#            all_cmp.append(compara)
            
            i = index_max(first_order.stateOutputNeurons)
            j = index_max(samples.inputs[ex])
            if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.inputs[ex][j] <= 0.5)) :
#            i = index_max(samples.outputs[ex])
#            j = index_max(samples.inputs[ex])
#            if ((samples.outputs[ex][i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
#                    or(samples.outputs[ex][i] <= 0.5 and samples.inputs[ex][j] <= 0.5)) :
                high_order[0].train(compara, 1)
                high_order[1].train(compara, -1)
                
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 1):
                    rms_ss +=1
            else:
                the += 1
                high_order[0].train(compara, -1)
                high_order[1].train(compara, 1)
                
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 0):
                    rms_ss +=1
#                print()
        
#            if(min(compara) >= 1.):
#                print(min(compara))
#                exit()
        rms.append(rms_ss/200)
#            if((index_max(first_order.calc_output(samples.inputs[ex]))!=index_max(samples.inputs[ex]) and  ( max(samples.inputs[ex]) > .5 or max(first_order.calc_output(samples.inputs[ex])) > .5 )) \
#                  or (max(first_order.calc_output(samples.inputs[ex])) <= 0.5 and max(samples.inputs[ex]) > 0.5 ) \
#                  or (max(first_order.calc_output(samples.inputs[ex])) > 0.5 and max(samples.inputs[ex]) < 0.5 )):
#                err += 1
                
#            first_order.train(samples.inputs[ex], samples.outputs[ex])
        print(epoch, err, the)
        err = 0.
        the = 0.

#    print(all_cmp)

    plt.plot(rms,
             label="first-order network")
    plt.legend(loc='best', frameon=False)
    plt.show()

    #testing
    #Suprathreshold stimuli
    
    pourc = {'hg_co' : 0,
             'hg_in' : 0,
             'lw_co' : 0,
             'lw_in' : 0,
             'co': 0,
             'in': 0}
    err = 0
    
    for ex in range(200):
        first_order.calc_output(samples.inputs[ex])
        
        if((index_max(first_order.calc_output(samples.inputs[ex]))==index_max(samples.outputs[ex]) and max(samples.outputs[ex]) > 0.5) \
            or(max(first_order.stateOutputNeurons) <= 0.5 and max(samples.inputs[ex]) <= 0.5)):
            pourc['co'] += 1
        else :
            pourc['in'] += 1
        
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
#                or(samples.outputs[ex][i] <= 0.5 and samples.inputs[ex][j] <= 0.5)) :
            err += 1
            if index_max(res) == 0:#high wager
                pourc['hg_co'] += 1
            else:
                pourc['lw_co'] += 1
        else:
            
            if index_max(res) == 0:#high wager
                pourc['hg_in'] += 1
            else:
                pourc['lw_in'] += 1
            
    print(pourc, err, pourc['lw_co'], pourc['lw_in'])
    

#    #Subthreshold stimuli
#    
#    #add +0.0012
#    print(samples.inputs[0])
#    for input in samples.inputs:
#        i_fix = index_max(input)
#        for i in range(len(input)):
#            if(i != i_fix):
#                input[i]+= 0.0012 
#        
#    print(samples.inputs[0])
#    
#    pourc = {'hg_co' : 0,
#             'hg_in' : 0,
#             'lw_co' : 0,
#             'lw_in' : 0}
#    
#    for ex in range(200):
#        first_order.calc_output(samples.inputs[ex])
#        
#        compara = []
#        for i in range(100):
#            compara.append(samples.inputs[ex][i] - first_order.stateOutputNeurons[i])
#            
#        res = [high_order[i].calc_output(compara)
#                    for i in range(2)]
#            
#        i = index_max(first_order.stateOutputNeurons)
#        j = index_max(samples.inputs[ex])
#        if ((first_order.stateOutputNeurons[i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
#                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.inputs[ex][j] <= 0.5)) :
#        
##        i = index_max(samples.outputs[ex])
##        j = index_max(samples.inputs[ex])
##        if ((samples.outputs[ex][i] > 0.5 and samples.inputs[ex][j] > 0.5 and i == j) \
##                or(samples.outputs[ex][i] <= 0.5 and samples.inputs[ex][i] <= 0.5)) :
#            if index_max(res) == 0:
#                pourc['hg_co'] += 1
#            else:
#                pourc['lw_in'] += 1
#        else:
#            if index_max(res) == 0:
#                pourc['hg_in'] += 1
#            else:
#                pourc['lw_co'] += 1
#            
#
#    print(pourc)
#    print('first')
    
    