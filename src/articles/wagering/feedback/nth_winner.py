# -*- coding: UTF-8 -*-
'''
Created on 19 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from utils import index_max, index_max_nth
from random import shuffle, seed
import matplotlib.pyplot as plt
from data import DataFile

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    nbEpoch = 200
    nbTry = 50
    display_interval = range(nbEpoch)[1::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        seed(i)
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        first_order.init_weights_randomly(-1, 1)
        
        high_order_h = MultilayerPerceptron(100, 100, 10, learning_rate=0.1, momentum=0.5, grid=mode)
        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'feedback' : [],
              'max': []}
    
    stats = []
    stats2 = []
    max2 = [0 for _ in range(nbEpoch)]
    div = [0 for _ in range(nbEpoch)]
    
    seed(100)
    
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'feedback' : 0., 
                 'max': 0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(examples.inputs[ex])
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
                
                cell = [0 for _ in range(10)]
                

                for k in range(10):
                    if index_max_nth(network['first_order'].stateOutputNeurons, k) == index_max(examples.outputs[ex]):
                        cell[k] = 1
                        break

                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                    max2[epoch] += max(network['first_order'].stateOutputNeurons)
#                    print(max(network['first_order'].stateOutputNeurons))
                    div[epoch] += 1
                perfo['max'] += max(network['first_order'].stateOutputNeurons)

#                res = [ network['high_order_h'][i].calc_output(network['first_order'].stateHiddenNeurons) for i in range(10)]
                res = network['high_order_h'].stateOutputNeurons

                if(index_max(res) == 0):
                    perfo['wager_proportion'] += 1
                
                if(index_max(cell) == index_max(res)):
                    perfo['high_order_h'] += 1
                    for k in range(10):
                        if(index_max(res) == k and index_max_nth(network['first_order'].stateOutputNeurons, k) == index_max(examples.outputs[ex])):
                            perfo['feedback'] += 1
                            break
                                
                
                stats.append(index_max(cell))
                #learn
                
#                for i in range(10):
#                    network['high_order_h'][i].train(network['first_order'].stateHiddenNeurons
#                                                     , cell[i])
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
        
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    
    iu = 0
    for i in range(10):
        t = 0
        for j in range(len(stats)):
            if(i == stats[j]):
                t += 1
        if(i == 0):
            iu = t
        print(i , ' -> ', t / len(stats))
        
    
    print()
    for i in range(10):
        t = 0
        for j in range(len(stats)):
            if(i == stats[j]):
                t += 1
        print(i , ' -> ', t / (len(stats) - iu))
        
   

    #testing
    for network in networks:
        for ex in range(len(examples.inputs)):
            network['first_order'].calc_output(examples.inputs[ex])
            network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)
            
            cell = [0 for _ in range(10)]
            

            for k in range(10):
                if index_max_nth(network['first_order'].stateOutputNeurons, k) == index_max(examples.outputs[ex]):
                    cell[k] = 1
                    break
                
            stats2.append(index_max(cell))

        
    iu = 0
    for i in range(10):
        t = 0
        for j in range(len(stats2)):
            if(i == stats2[j]):
                t += 1
        if(i == 0):
            iu = t
        print(i , ' -> ', t / len(stats2))
        
    print()
    for i in range(10):
        t = 0
        for j in range(len(stats2)):
            if(i == stats2[j]):
                t += 1
        print(i , ' -> ', t / (len(stats2) - iu))


    for j in range(nbEpoch):
        if(div[j] != 0):
            max2[j] /= div[j]

    plt.title("Performance of first-order and higher-order networks with feedback ( nth W-T-A )")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
#    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
#    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
#    plt.plot(display_interval , y_perfo['max'][3::5], label="most active neuron", linewidth=2)
#    plt.plot(display_interval , max2[3::5], label="most active neuron (good answer)", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    


#    x =  [ [1 for x in range(1935)] + 
#          [2 for x in range(1060)] +
#          [3 for x in range(932)] +
#          [4 for x in range(876)] +
#          [5 for x in range(865)] +
#          [6 for x in range(879)] +
#          [7 for x in range(971)] +
#          [8 for x in range(1138)] +
#          [9 for x in range(1338)] ]
#
#    h, b, c = plt.hist(x, bins=9, histtype='bar')
#    
#    h = 100*h/float(len(x))
#    
#    plt.title("Distribution of the answer when the first-order is wrong")
#    plt.ylabel("frequency")
#    plt.xlabel("n th best active neuron")
#    plt.show()