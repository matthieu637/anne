'''
Created on 14 mai 2012

@author: matthieu
'''


from multilayerp import MultilayerPerceptron
import matplotlib.pyplot as plt
from data import DataFile
from utils import index_max, compare, compare_f

from simulation import *

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.9
    lrate = 0.1
    nbr_try = 10
    nbr_epoch = 100
    
    def fo():
        return MultilayerPerceptron(20, 5, 10, learning_rate=lrate, momentum=momentum, grid=mode)
    def ho():
        return MultilayerPerceptron(5, 10, 35, learning_rate=lrate, momentum=momentum, grid=mode)
    
    sim = Simulation(nbr_network, {'first_order' : fo, 'high_order_10' : ho})
    
    
    def data():
        return DataFile("digit_shape.txt", mode)
    
    sim.onData(data)
    
    
    def step(network, plot, inputs, outputs):
        network['first_order'].calc_output(inputs)
        network['high_order_10'].calc_output(network['first_order'].stateHiddenNeurons)

        entire_first_order = inputs + \
                             network['first_order'].stateHiddenNeurons + \
                             network['first_order'].stateOutputNeurons
    
        if(index_max(network['first_order'].stateOutputNeurons) != index_max(outputs)):
            plot['first_order'] += 1
        if(index_max(network['high_order_10'].stateOutputNeurons[25:35]) != index_max(network['first_order'].stateOutputNeurons)):
            plot['high_order_10'] += 1
        
        #learn
        network['high_order_10'].train(network['first_order'].stateHiddenNeurons,
                                       entire_first_order)
        network['first_order'].train(inputs, outputs)
        
    
    sim.launch(nbr_epoch, nbr_try, step)
    
    def moregraph(plt):
        plt.title('Error RMS error of first-order and high-order networks')
        plt.ylabel('RMS ERROR')
        plt.xlabel("EPOCHS")
    
    sim.plot(6, ["first-order network ( winner take all )" , "high-order network ( winner take all )"], [2, 2], moregraph)
    
    
    