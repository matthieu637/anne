# -*- coding: UTF-8 -*-
'''
Created on 19 March 2012

@author: Matthieu Zimmer

'''

from multilayerp import MultilayerPerceptron
from perceptron import PerceptronR0to1
from utils import index_max
from random import shuffle
import matplotlib.pyplot as plt
from data import DataFile
from articles.digital_reco.feedback.merging import ampli

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    nbEpoch = 201
    nbTry = 50
    display_interval = range(nbEpoch)[3::5]
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(16 * 16, 100, 10, learning_rate=0.15, momentum=momentum, grid=mode)
        high_order_h = MultilayerPerceptron(100, 20, 2, learning_rate=0.1, momentum=0., grid=mode)
        feedback = [PerceptronR0to1(100 + 8, 0.1, 0., True) for _ in range(10)]
        
        first_order.init_weights_randomly(-1, 1)

        networks[i] = {'first_order' : first_order,
                    'high_order_h' : high_order_h,
                    'feedback': feedback}

    #create example
    examples = DataFile("digit_handwritten_16.txt", mode)

    #3 curves
    y_perfo = {'first_order' : [] ,
              'high_order_h' : [],
              'wager_proportion': [],
              'feedback' : [], 
              'diff' : []}
    
    corrections = [[0 for _ in range(10)] for _ in range(10)]
    
    
    #learning
    for epoch in range(nbEpoch):
        perfo = {'first_order' : 0. ,
                 'high_order_h' : 0.,
                 'wager_proportion': 0.,
                 'feedback' : 0.,
                 'diff': 0.}
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(examples.inputs[ex])
                cell = [mode, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, mode]
                
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)

                if(index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex])):
                    perfo['first_order'] += 1
                if(index_max(network['high_order_h'].stateOutputNeurons) == index_max(cell)):
                    perfo['high_order_h'] += 1
                    
                
                res = [ network['feedback'][i].calc_output(network['first_order'].stateHiddenNeurons + 
                                                           ampli(network['high_order_h'].stateOutputNeurons, 8)) for i in range(10)]
                if(index_max(res) == index_max(examples.outputs[ex])):
                    perfo['feedback'] += 1
                    if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                        corrections[index_max(network['first_order'].stateOutputNeurons)][index_max(res)] += 1
                    
                if(index_max(network['high_order_h'].stateOutputNeurons) == 1):
                    perfo['wager_proportion'] += 1
                
                
                #learn
                for i in range(10):
                    network['feedback'][i].train(network['first_order'].stateHiddenNeurons + 
                                                 ampli(network['high_order_h'].stateOutputNeurons, 8)
                                                 , examples.outputs[ex][i])
                network['high_order_h'].train(network['first_order'].stateHiddenNeurons,
                                               cell)
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])

        perfo['diff'] = (perfo['feedback'] - perfo['first_order'])
        for k in y_perfo.keys():
            y_perfo[k].append(perfo[k] / (nbTry * nbr_network))

        print(epoch)
    
    print("score : ", sum(y_perfo['diff']) / len(y_perfo['diff']))
    print(corrections)
    
    corrections2 = [[0 for _ in range(10)] for _ in range(10)]
    
    #testing
    for epoch in range(nbEpoch):
        for network in networks:
            for ex in range(len(examples.inputs)):
                network['first_order'].calc_output(examples.inputs[ex])
                cell = [mode, 1] \
                        if index_max(network['first_order'].stateOutputNeurons) == index_max(examples.outputs[ex]) \
                        else [1, mode]
                
                network['high_order_h'].calc_output(network['first_order'].stateHiddenNeurons)

                res = [ network['feedback'][i].calc_output(network['first_order'].stateHiddenNeurons + 
                                                           ampli(network['high_order_h'].stateOutputNeurons, 8)) for i in range(10)]
                if(index_max(res) == index_max(examples.outputs[ex])):
                    if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                        corrections2[index_max(network['first_order'].stateOutputNeurons)][index_max(res)] += 1
                    
    plt.title("Feedback with a third perceptron network ( on hidden layer )")
    plt.plot(display_interval , y_perfo['first_order'][3::5], label="first-order network", linewidth=2)
    plt.plot(display_interval , y_perfo['high_order_h'][3::5], label="high-order network (high learning rate)")
    plt.plot(display_interval , y_perfo['wager_proportion'][3::5], label="proportion of high wagers")
    plt.plot(display_interval , y_perfo['feedback'][3::5], label="feedback", linewidth=2)
    plt.ylabel('SUCCESS RATIO')
    plt.xlabel("EPOCHS")
    plt.axis((0, nbEpoch, 0, 1.))
    plt.legend(loc='best', frameon=False)
    plt.show()
    
    
    colors =[(0.2, 0.8, 0.88), 'b', 'g', 'r', 'c', 'm', 'y', 'w', (0.8,0.1,0.8), (0.,0.2,0.5)]

    for i in range(10)[0::]:
        plt.bar(range(i*12+10)[i*12::], corrections[i], color=colors[i])
    
    plt.ylabel("Number of corrections")
    plt.xlabel("Number to correct")
    plt.title("Distribution corrections")
    plt.xticks([5+i*12 for i in range(10)], range(10))
    
    plt.show()
    
    for i in range(10)[0::]:
        plt.bar(range(i*12+10)[i*12::], corrections2[i], color=colors[i])
    
    plt.ylabel("Number of corrections")
    plt.xlabel("Number to correct")
    plt.title("Distribution corrections")
    plt.xticks([5+i*12 for i in range(10)], range(10))
    
    plt.show()
    
