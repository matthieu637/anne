# -*- coding: UTF-8 -*-
'''
Created on 24 fevr. 2012

@author: matthieu637
'''

from data import DataFile
from random import shuffle
from network import MultilayerNetwork
from neuron import Neuron, NeuronN0to1, NeuronR0to1
from utils import index_max
import matplotlib.pyplot as plt

if __name__ == '__main__':
   
    r = {'x':(0, 1),
         '.': (0,),
         '?': (0, 0.02),
         '!': (1,)}
    samples = DataFile("../data/blindslight.txt", rules=r)
#    samples.inputs = samples.inputs[0:100]
    
    
    first_order = MultilayerNetwork(100, 60, 100, 0, 0.9, 0., 2, False, True)
    first_order.init_random_weights(-0.6, 0.6)
    
    
    high_order = [NeuronR0to1(100, 0.1, 0., 2., Neuron.Output, False, True) for _ in range(2)]
#    high_order = [NeuronN0to1(100, 0.1, 0., False) for _ in range(2)]
    high_order[0].init_random_weights(0., 0.1)
    high_order[1].init_random_weights(0., 0.1)

    err = 0.
    the = 0.
    rms = []
    rms2 = []
    
    nbEpoch = 150
    #training
    for epoch in range(nbEpoch):
        l_exx = list(range(len(samples.inputs)))
        shuffle(l_exx)
        rms_ss = 0.
        rms_ss2 = 0.
        for ex in l_exx:
#            first_order.train(samples.inputs[ex], samples.outputs[ex])
            first_order.calc_output(samples.inputs[ex])
        
            #compara
            compara=[]
            for i in range(100):
                compara.append((samples.inputs[ex][i] - first_order.stateOutputNeurons[i]))

            res2 = [high_order[i].calc_output(list(compara))
                    for i in range(2)]
            if(index_max(res2) == 0):
                err += 1
            
            
            i = index_max(first_order.stateOutputNeurons)
            j = index_max(samples.outputs[ex])
            if ((first_order.stateOutputNeurons[i] > 0.5 and samples.outputs[ex][j] > 0.5 and i==j ) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.outputs[ex][j] <= 0.5)) :

                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 0):
                    rms_ss +=1
                    
                high_order[0].train(compara, 1.)
                high_order[1].train(compara, -1.)
            else:
                the += 1
                
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 1):
                    rms_ss +=1
                    
                high_order[0].train(compara, -1.)
                high_order[1].train(compara, 1.)

            first_order.train(samples.inputs[ex], samples.outputs[ex])
            
        rms.append(rms_ss/200)
        rms2.append((200-the)/200)

            
        print(epoch, "perf 1st : ", (200 - the), " | perf 2nd : ",  rms_ss, " | high wag %d" % err)
        err = 0.
        the = 0.


    plt.plot(rms,
             label="high-order network")
    plt.plot(rms2,
             label="first-order network")
    plt.axis((0, nbEpoch, 0, 1.))
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
    
    for ex in range(200):
        first_order.calc_output(samples.inputs[ex])
        
        #compara
        compara=[]
        for i in range(100):
            compara.append((samples.inputs[ex][i] - first_order.stateOutputNeurons[i]))

        res33 = [high_order[i].calc_output(compara) for i in range(2)]
        
        i = index_max(first_order.stateOutputNeurons)
        j = index_max(samples.outputs[ex])
        if ((first_order.stateOutputNeurons[i] > 0.5 and samples.outputs[ex][j] > 0.5 and i == j) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.outputs[ex][j] <= 0.5)) :
            pourc['co'] += 1
            if index_max(res33) == 0:#high wager
                pourc['hg_co'] += 1
            else:
                pourc['lw_co'] += 1
        else:
            pourc['in'] += 1
            if index_max(res33) == 0:#high wager
                pourc['hg_in'] += 1
            else:
                pourc['lw_in'] += 1
            
    print(pourc, pourc['lw_co'], pourc['lw_in'])
    

#    #Subthreshold stimuli
#    
    #add +0.0012
    print(samples.inputs[0])
    for inpts in samples.inputs:
        i_fix = index_max(inpts)
        for i in range(len(inpts)):
            if(i != i_fix):
                inpts[i]+= 0.0012 
        
    print(samples.inputs[0])
    
    pourc = {'hg_co' : 0,
             'hg_in' : 0,
             'lw_co' : 0,
             'lw_in' : 0,
             'co': 0,
             'in': 0}
    
    for ex in range(200):
        first_order.calc_output(samples.inputs[ex])
        
        #compara
        compara=[]
        for i in range(100):
            compara.append((samples.inputs[ex][i] - first_order.stateOutputNeurons[i]))

        res33 = [high_order[i].calc_output(compara) for i in range(2)]
        
        i = index_max(first_order.stateOutputNeurons)
        j = index_max(samples.outputs[ex])
        if ((first_order.stateOutputNeurons[i] > 0.5 and samples.outputs[ex][j] > 0.5 and i == j) \
                or(first_order.stateOutputNeurons[i] <= 0.5 and samples.outputs[ex][j] <= 0.5)) :
            pourc['co'] += 1
            if index_max(res33) == 0:#high wager
                pourc['hg_co'] += 1
            else:
                pourc['lw_co'] += 1
        else:
            pourc['in'] += 1
            if index_max(res33) == 0:#high wager
                pourc['hg_in'] += 1
            else:
                pourc['lw_in'] += 1
            
    print(pourc, pourc['lw_co'], pourc['lw_in'])
    
    