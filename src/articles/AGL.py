'''
Created on 6 mars 2012

@author: matthieu637

Artificial Grammar Learning
'''

from data import DataFile
from random import shuffle
from network import MultilayerNetwork
from neuron import Neuron, NeuronN0to1, NeuronR0to1
from utils import index_max
from random import randint
import matplotlib.pyplot as plt



def random_pattern():
    result = []
    nb_letter = randint(3, 8)
    for i in range(8):
        if(i < nb_letter):
            letter = randint(1, 5)
            for k in range(6):
                if(k == letter):
                    result.append(1.)
                else:
                    result.append(0.)
        else:
            for k in range(6):
                if(k == 0):
                    result.append(1.)
                else:
                    result.append(0.)
    return (result, result)

def pattern_to_list(pat):
    res = []
    for i in range(8):
        res.append(index_max(pat[i * 6:i * 6 + 6]))
    return res

if __name__ == '__main__':
   
    ptrain_pattern = [random_pattern() for _ in range(80)]
    
#    print(ptrain_pattern[0][0])
#    print(pattern_to_list(ptrain_pattern[0][0]))
#    exit()
    
    first_order = MultilayerNetwork(48, 40, 48, 0, 0.4, 0.5, 1., False, True)
    first_order.init_random_weights(-1, 1)
    
    
    high_order = [NeuronR0to1(48, 0.4, 0.5, 1., Neuron.Output, False, True) for _ in range(2)]
#    high_order = [NeuronN0to1(48, 0.4, 0.5, False) for _ in range(2)]
    high_order[0].init_random_weights(0., 0.1)
    high_order[1].init_random_weights(0., 0.1)


    print("pre-training")
    err = 0.
    the = 0.
    rms = []
    rms2 = []
    
    nbEpoch = 60
    
    l_ex = list(range(80))
    shuffle(l_ex)
    
    ptrain_pattern_l = l_ex[0:40]
    #pre-training
    for epoch in range(nbEpoch):
        l_exx = list(range(80))
        shuffle(l_exx)
        rms_ss = 0.
        rms_ss2 = 0.
        for ex in l_exx:
            first_order.calc_output(ptrain_pattern[ex][0])
        
            #compara
            compara = []
            for i in range(48):
                compara.append(ptrain_pattern[ex][0][i] - first_order.stateOutputNeurons[i])

            res2 = [high_order[i].calc_output(compara)
                    for i in range(2)]
            if(index_max(res2) == 0):
                err += 1
            
            
            if (pattern_to_list(first_order.stateOutputNeurons) == pattern_to_list(ptrain_pattern[ex][1])) :
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 0):
                    rms_ss += 1
                    
                high_order[0].train(compara, 1.)
                high_order[1].train(compara, 0.)
            else:
                the += 1
                
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 1):
                    rms_ss += 1
                    
                high_order[0].train(compara, 0.)
                high_order[1].train(compara, 1.)

            if(ex in ptrain_pattern_l):
                first_order.train(ptrain_pattern[ex][0], ptrain_pattern[ex][1])
            
        rms.append(rms_ss / 80)
        rms2.append((80 - the) / 80)

            
        print(epoch, "perf 1st : ", (80 - the), " | perf 2nd : ", rms_ss, " | high wag %d" % err)
        err = 0.
        the = 0.

    first_order.init_random_weights(-1, 1)

    ptrain_pattern = [random_pattern() for _ in range(45)]
    
    print("training")
    nbEpoch = 12
    #training
    for epoch in range(nbEpoch):
        l_exx = list(range(45))
        shuffle(l_exx)
        rms_ss = 0.
        rms_ss2 = 0.
        for ex in l_exx:
            first_order.calc_output(ptrain_pattern[ex][0])
        
            #compara
            compara = []
            for i in range(48):
                compara.append(ptrain_pattern[ex][0][i] - first_order.stateOutputNeurons[i])

            res2 = [high_order[i].calc_output(list(compara))
                    for i in range(2)]
            if(index_max(res2) == 0):
                err += 1
            
            
            if (pattern_to_list(first_order.stateOutputNeurons) == pattern_to_list(ptrain_pattern[ex][1])) :
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 0):
                    rms_ss += 1

            else:
                the += 1
                
                res = [high_order[i].calc_output(compara)
                    for i in range(2)]
            
                if(index_max(res) == 1):
                    rms_ss += 1

            first_order.train(ptrain_pattern[ex][0], ptrain_pattern[ex][1])
            
        rms.append(rms_ss / 45)
        rms2.append((45 - the) / 45)

            
        print(epoch, "perf 1st : ", (45 - the), " | perf 2nd : ", rms_ss, " | high wag %d" % err)
        err = 0.
        the = 0.


    
    
    plt.plot(rms,
             label="high-order network")
    plt.plot(rms2,
             label="first-order network")
    plt.axis((0, 60 + nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    

    #testing
    
    pourc = {'hg_co' : 0,
             'hg_in' : 0,
             'lw_co' : 0,
             'lw_in' : 0,
             'co': 0,
             'in': 0}
    
    l_ex = list(range(45))
    shuffle(l_ex)
    for ex in l_ex[0:30]:
        first_order.calc_output(ptrain_pattern[ex][0])
        
        #compara
        compara = []
        for i in range(48):
            compara.append(ptrain_pattern[ex][0][i] - first_order.stateOutputNeurons[i])

        res33 = [high_order[i].calc_output(compara) for i in range(2)]
        
        if (pattern_to_list(first_order.stateOutputNeurons) == pattern_to_list(ptrain_pattern[ex][1])) :
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
    
    for key in pourc.keys():
        pourc[key] /= 0.30
    
    print(pourc, pourc['lw_co'], pourc['lw_in'])
    
    
    
    
    
    ptrain_pattern = [random_pattern() for _ in range(30)]
    
    pourc = {'hg_co' : 0,
             'hg_in' : 0,
             'lw_co' : 0,
             'lw_in' : 0,
             'co': 0,
             'in': 0}
    
    for ex in range(30):
        first_order.calc_output(ptrain_pattern[ex][0])
        
        #compara
        compara = []
        for i in range(48):
            compara.append(ptrain_pattern[ex][0][i] - first_order.stateOutputNeurons[i])

        res33 = [high_order[i].calc_output(compara) for i in range(2)]
        
        if (pattern_to_list(first_order.stateOutputNeurons) == pattern_to_list(ptrain_pattern[ex][1])) :
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
            
    for key in pourc.keys():
        pourc[key] /= 0.30
    
    print(pourc, pourc['lw_co'], pourc['lw_in'])
    
