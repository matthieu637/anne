# -*- coding: UTF-8 -*-
'''
Created on 22 March 2012

@author: Matthieu Zimmer

Article implementation
'''

from multilayerp import MultilayerPerceptron
from random import shuffle
from perceptron import PerceptronR0to1
import matplotlib.pyplot as plt
from data import DataFile, DataFileR
from utils import index_max
from copy import deepcopy
from random import seed




if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 10
    momentum = 0.5
    lrate = 0.1
    nbTry = 10
    seed(10)
    
    #create all networks
    networks = [{} for _ in range(nbr_network)]
    
        #create inputs/outputs to learn
#    examples = DataFile("digit_handwritten_16.txt", mode)
    examples = DataFileR("iris.txt")
    
    nbInputs = len(examples.inputs[0])
    nbHidden =  3 + nbInputs//4
    nbOutputs = len(examples.outputs[0])
    
    def newtask(l):
        imax = index_max(l)
        l[imax] = 0.
        l[nbOutputs - 1 - imax] = 1.
    
    def newtask2(l):
        imax = index_max(l)
        l[imax] = 0.
        imax = imax + 1 if imax != nbOutputs - 1 else 0
        l[imax] = 1
    
    def newtask3(l):
        imax = index_max(l)
        l[imax] = 0.
        imax = imax - 1 if imax != 0 else nbOutputs - 1
        l[imax] = 1
    
    for i in range(nbr_network):
        first_order = MultilayerPerceptron(nbInputs, nbHidden, nbOutputs, learning_rate=lrate, momentum=momentum, grid=mode)
        high_order_10 = MultilayerPerceptron(nbHidden, nbHidden * 2, nbInputs+ nbHidden+ nbOutputs, learning_rate=lrate, momentum=momentum, grid=mode)
        control1 = deepcopy(first_order)
        control2 = deepcopy(high_order_10)
        perceptron = [PerceptronR0to1(nbInputs, lrate, momentum) for _ in range(nbOutputs)]

        networks[i] = {'first_order' : first_order,
                        'high_order_10' : high_order_10,
                        'first_order_control': control1,
                        'high_order_control':control2,
                        'perceptron' : perceptron}

    #3 curves
    err_plot = {'first_order' : [] ,
              'high_order_10' : [],
              'first_order_control': [],
              'high_order_control': [],
              'perceptron' : []}

    #learning
    for epoch in range(1000):
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.,
                           'first_order_control': 0.,
                           'high_order_control': 0.,
                           'perceptron': 0.}
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(
                                            examples.inputs[ex])
                network['first_order_control'].calc_output(
                                            examples.inputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                                     
                entire_first_order2 = examples.inputs[ex] + \
                                     network['first_order_control'].stateHiddenNeurons + \
                                     network['first_order_control'].stateOutputNeurons
                
                network['high_order_10'].calc_output(
                                            network['first_order'].stateHiddenNeurons)
                
                network['high_order_control'].calc_output(
                                            network['first_order_control'].stateHiddenNeurons)
                
                res = [network['perceptron'][i].calc_output(examples.inputs[ex]) for i in range(nbOutputs)]

                if(index_max(res) != index_max(examples.outputs[ex])):
                    err_one_network['perceptron'] += 1

                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
                    
                if(index_max(network['first_order_control'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order_control'] += 1

                if(index_max(network['high_order_10'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order'].stateOutputNeurons)):
                    err_one_network['high_order_10'] += 1
                    
                if(index_max(network['high_order_control'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order_control'].stateOutputNeurons)):
                    err_one_network['high_order_control'] += 1

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['high_order_control'].train(network['first_order_control'].stateHiddenNeurons,
                                               entire_first_order2)
                
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                network['first_order_control'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                
                [network['perceptron'][i].train(examples.inputs[ex], examples.outputs[ex][i]) for i in range(nbOutputs)]
            

        #add plot
        for k in err_plot.keys() :
            err_plot[k].append(err_one_network[k] / (nbTry * nbr_network))
        
        print(epoch, " err : ", err_plot['first_order'][epoch])
        
        
    print(examples.outputs[0])
    
    for k in range(len(examples.outputs)):
        newtask(examples.outputs[k])
        
    print(examples.outputs[0])
    
        
    #learning
    for epoch in range(1000):
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.,
                           'first_order_control': 0.,
                           'high_order_control': 0.,
                           'perceptron': 0.}
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(
                                            examples.inputs[ex])
                network['first_order_control'].calc_output(
                                            examples.inputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                                     
                entire_first_order2 = examples.inputs[ex] + \
                                     network['first_order_control'].stateHiddenNeurons + \
                                     network['first_order_control'].stateOutputNeurons
                
                network['high_order_10'].calc_output(
                                            network['first_order'].stateHiddenNeurons)
                
                network['high_order_control'].calc_output(
                                            network['first_order_control'].stateHiddenNeurons)
                
                res = [network['perceptron'][i].calc_output(examples.inputs[ex]) for i in range(nbOutputs)]

                if(index_max(res) != index_max(examples.outputs[ex])):
                    err_one_network['perceptron'] += 1

                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
                    
                if(index_max(network['first_order_control'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order_control'] += 1

                if(index_max(network['high_order_10'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order'].stateOutputNeurons)):
                    err_one_network['high_order_10'] += 1
                    
                if(index_max(network['high_order_control'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order_control'].stateOutputNeurons)):
                    err_one_network['high_order_control'] += 1

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['high_order_control'].train(network['first_order_control'].stateHiddenNeurons,
                                               entire_first_order2)
                
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                network['first_order_control'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                
                [network['perceptron'][i].train(examples.inputs[ex], examples.outputs[ex][i]) for i in range(nbOutputs)]
            

        #add plot
        for k in err_plot.keys() :
            err_plot[k].append(err_one_network[k] / (nbTry * nbr_network))
        
        print(epoch, " err : ", err_plot['first_order'][epoch])
            

    print(examples.outputs[0])
    
    for k in range(len(examples.outputs)):
        newtask2(examples.outputs[k])
        
    print(examples.outputs[0])
    
        
    #learning
    for epoch in range(1000):
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.,
                           'first_order_control': 0.,
                           'high_order_control': 0.,
                           'perceptron': 0.}
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(
                                            examples.inputs[ex])
                network['first_order_control'].calc_output(
                                            examples.inputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                                     
                entire_first_order2 = examples.inputs[ex] + \
                                     network['first_order_control'].stateHiddenNeurons + \
                                     network['first_order_control'].stateOutputNeurons
                
                network['high_order_10'].calc_output(
                                            network['first_order'].stateHiddenNeurons)
                
                network['high_order_control'].calc_output(
                                            network['first_order_control'].stateHiddenNeurons)
                
                res = [network['perceptron'][i].calc_output(examples.inputs[ex]) for i in range(nbOutputs)]

                if(index_max(res) != index_max(examples.outputs[ex])):
                    err_one_network['perceptron'] += 1

                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
                    
                if(index_max(network['first_order_control'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order_control'] += 1

                if(index_max(network['high_order_10'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order'].stateOutputNeurons)):
                    err_one_network['high_order_10'] += 1
                    
                if(index_max(network['high_order_control'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order_control'].stateOutputNeurons)):
                    err_one_network['high_order_control'] += 1

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['high_order_control'].train(network['first_order_control'].stateHiddenNeurons,
                                               entire_first_order2)
                
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                network['first_order_control'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                
                [network['perceptron'][i].train(examples.inputs[ex], examples.outputs[ex][i]) for i in range(nbOutputs)]
            

        #add plot
        for k in err_plot.keys() :
            err_plot[k].append(err_one_network[k] / (nbTry * nbr_network))
        
        print(epoch, " err : ", err_plot['first_order'][epoch])


    print(examples.outputs[0])
    
    for k in range(len(examples.outputs)):
        newtask3(examples.outputs[k])
        
    print(examples.outputs[0])
    
        
    #learning
    for epoch in range(1000):
        err_one_network = {'first_order' : 0. ,
                           'high_order_10' : 0.,
                           'first_order_control': 0.,
                           'high_order_control': 0.,
                           'perceptron': 0.}
        
        for network in networks:
            l_exx = list(range(len(examples.inputs)))
            shuffle(l_exx)
            for ex in l_exx[0:nbTry]:
                network['first_order'].calc_output(
                                            examples.inputs[ex])
                network['first_order_control'].calc_output(
                                            examples.inputs[ex])
                
                entire_first_order = examples.inputs[ex] + \
                                     network['first_order'].stateHiddenNeurons + \
                                     network['first_order'].stateOutputNeurons
                                     
                entire_first_order2 = examples.inputs[ex] + \
                                     network['first_order_control'].stateHiddenNeurons + \
                                     network['first_order_control'].stateOutputNeurons
                
                network['high_order_10'].calc_output(
                                            network['first_order'].stateHiddenNeurons)
                
                network['high_order_control'].calc_output(
                                            network['first_order_control'].stateHiddenNeurons)
                
                res = [network['perceptron'][i].calc_output(examples.inputs[ex]) for i in range(nbOutputs)]

                if(index_max(res) != index_max(examples.outputs[ex])):
                    err_one_network['perceptron'] += 1

                if(index_max(network['first_order'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order'] += 1
                    
                if(index_max(network['first_order_control'].stateOutputNeurons) != index_max(examples.outputs[ex])):
                    err_one_network['first_order_control'] += 1

                if(index_max(network['high_order_10'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order'].stateOutputNeurons)):
                    err_one_network['high_order_10'] += 1
                    
                if(index_max(network['high_order_control'].stateOutputNeurons[nbInputs + nbHidden:nbInputs + nbHidden+ nbOutputs]) != 
                    index_max(network['first_order_control'].stateOutputNeurons)):
                    err_one_network['high_order_control'] += 1

                #learn
                network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                               entire_first_order)
                network['high_order_control'].train(network['first_order_control'].stateHiddenNeurons,
                                               entire_first_order2)
                
                network['first_order'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                network['first_order_control'].train(examples.inputs[ex],
                                             examples.outputs[ex])
                
                [network['perceptron'][i].train(examples.inputs[ex], examples.outputs[ex][i]) for i in range(nbOutputs)]
            

        #add plot
        for k in err_plot.keys() :
            err_plot[k].append(err_one_network[k] / (nbTry * nbr_network))
        
        print(epoch, " err : ", err_plot['first_order'][epoch])
        

    display_interval = range(len(err_plot['first_order']))[::50]
    #displays errors
    plt.plot(display_interval, [err_plot['first_order'][i] for i in display_interval],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval, [err_plot['high_order_10'][i] for i in display_interval],
             label="output layer ( winner take all )")
    
    plt.title('Error ratio of first-order and high-order networks')
    plt.ylabel('ERROR RATIO')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()



    plt.plot(display_interval, [err_plot['first_order_control'][i] for i in display_interval],
             label="first-order network",
             linewidth=2)
    
    plt.plot(display_interval, [err_plot['high_order_control'][i] for i in display_interval],
             label="output layer ( winner take all )")
    
    plt.plot(display_interval, [err_plot['perceptron'][i] for i in display_interval],
             label="perceptron",
             linewidth=2)
    
    plt.title('Error ratio of first-order and high-order networks ( control network )')
    plt.ylabel('ERROR RATIO')
    plt.xlabel("EPOCHS")
    plt.legend(loc='best', frameon=False)
    plt.show()
    