# -*- coding: UTF-8 -*-
'''
Created on 13 fevr. 2012

@author: matthieu637

Article test
'''

from digit import Factory as DigitsFactory
from network import MultilayerNetwork
from utils import findMax, RMS
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    mode = MultilayerNetwork.R0to1
    digits = [DigitsFactory.digitToMatrix(k, (5, 4), mode) for k in range(10)]
    mn = MultilayerNetwork(20, 5, 10, learning_rate=0.1, momentum=.9, grid=mode)
    mn2 = MultilayerNetwork(5, 10, 35, learning_rate=0.1, momentum=.9, grid=mode);

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
        errs = []
        sses = []
        errr = []
        sse = 0.
        sse2 = 0.
        for ex in range(10):
            
            errr.extend(map(lambda x, y: abs(x - y), \
                        mn.calc_output(examples[ex]["inputs"]), examples[ex]["outputs"]))
            
            err = reduce(lambda x, y:x + y, map(lambda x, y: abs(x - y), \
                        mn.calc_output(examples[ex]["inputs"]), examples[ex]["outputs"]))
            errs.append(err/10)
            
            _sse = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y,2), \
                        mn.calc_output(examples[ex]["inputs"]), examples[ex]["outputs"]))
            sses.append(_sse)
            sse += _sse
            
            
            
            _sse2 = reduce(lambda x, y:x + y, map(lambda x, y: pow(x - y,2), \
                        mn2.calc_output(mn.stateHiddenNeurons), examples[ex]["inputs"]+mn.stateHiddenNeurons+mn.stateOutputNeurons))
            sse2 += _sse2
            mn.train(examples[ex]["inputs"], examples[ex]["outputs"])
            mn2.train(mn.stateHiddenNeurons, examples[ex]["inputs"]+mn.stateHiddenNeurons+mn.stateOutputNeurons)
            
        rms = sqrt(sse/10)
        rms2 = pow(sse/10, 2)
        rms3 = sse/10
        #print epoch, rms, rms/max(errs), rms/max(sses), rms/max(errr)
        #print epoch, rms2, rms2/max(errs), rms2/max(sses), rms2/max(errr)
#        y[0].append(rms)
#        y[1].append(rms/max(errs))
#        y[2].append(rms/max(sses))
#        y[3].append(rms/max(errr))
#        y[4].append(rms2)
#        y[5].append(rms2/max(errs))
#        y[6].append(rms2/max(sses))
#        y[7].append(rms2/max(errr))
        y[8].append(rms3)
#        y[9].append(rms3/max(errs))
#        y[10].append(rms3/max(sses))
#        y[11].append(rms3/max(errr))
        y[12].append(sse2/10)
        
    #for i in range(12):
    for i in [8,12]:
        plt.plot(x, y[i],"-",label="line %d" % i)
    plt.ylabel('ERROR')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend()
    plt.show()
    

    #testing
    for ex in range(10):
        print digits[ex]
        print mn.calc_output(digits[ex].ravel().tolist())
        print findMax(mn.calc_output(digits[ex].ravel().tolist()))