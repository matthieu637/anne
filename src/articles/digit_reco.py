# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637

Article test
'''

from digit import Factory as DigitsFactory
from network import MultilayerNetwork
from utils import findMax
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    mode = MultilayerNetwork.R0to1
    digits = [DigitsFactory.digitToMatrix(k, (5, 4), mode) for k in range(10)]
    nbr_network = 1
    
    
    mn = [MultilayerNetwork(20, 5, 10, learning_rate=0.1, momentum=.9, grid=mode) for _ in range(nbr_network)]

#    mn2 = MultilayerNetwork(5, 10, 35, learning_rate=0.1, momentum=.9, grid=mode)

    #create example
    examples = [{} for _ in range(10)]
    for ex in range(10):
        examples[ex]["inputs"] = digits[ex].ravel().tolist()
        examples[ex]["outputs"] = [mode] * 10
        examples[ex]["outputs"][ex] = 1


    nbEpoch = 1000
    x = np.linspace(0, nbEpoch, nbEpoch )
    y = [[] for _ in range(13)]
    

    #learning
    for epoch in range(nbEpoch):
#        sse2 = 0.
        rms = []
        rms3 = []
        
        for network in mn:
            sse = 0.
#            print network.calc_output(examples[ex]["inputs"])
#            for ex in range(10):
            for ex in np.random.randint(0,10,10):
#            ex = np.random.randint(1,10)

            
                _sse = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y,2), \
                    network.calc_output(examples[ex]["inputs"]), examples[ex]["outputs"]))
                sse += _sse
    #            _sse2 = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y,2), \
    #                        mn2.calc_output(mn.stateHiddenNeurons), examples[ex]["inputs"]+mn.stateHiddenNeurons+mn.stateOutputNeurons))
    #            sse2 += _sse2
                network.train(examples[ex]["inputs"], examples[ex]["outputs"])
    #            mn2.train(mn.stateHiddenNeurons, examples[ex]["inputs"]+mn.stateHiddenNeurons+mn.stateOutputNeurons)

            rms.append(sse/10)
            
#            print sqrt(sse/100)
        
        
        #print epoch, rms, rms/max(errs), rms/max(sses), rms/max(errr)
        #print epoch, rms2, rms2/max(errs), rms2/max(sses), rms2/max(errr)
        y[0].append(sum(rms)/nbr_network)

        
    #for i in range(12):
    for i in [0]:
        plt.plot(x, y[i],"-",label="line %d" % i)
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend()
    plt.show()
    

    #testing
    for ex in range(10):
        for network in mn:
            print digits[ex]
            print network.calc_output(digits[ex].ravel().tolist())
            print findMax(network.calc_output(digits[ex].ravel().tolist()))