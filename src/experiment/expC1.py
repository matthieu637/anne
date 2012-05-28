'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron
from data import DataFile
from utils import index_max
from representation import graph_network


from simulation import Simulation

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.5
    lrate = 0.15
    nbr_try = 50
    nbr_epoch = 204
    width = 4
    discretize = 4
    point = 4
    
#   Data Sample Declaration
    def data():
        return DataFile("digit_shape.txt", mode)
#        return DataFile("digit_handwritten_16.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        m = MultilayerPerceptron(inputs, 100, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
        m.init_weights_randomly(-1, 1)
        return m
    def SoN(inputs, outputs):
        return MultilayerPerceptron(100, 20, 2, learning_rate=0.1, momentum=0, grid=mode)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]

        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, cell)
        
        #err
        simu.perf('FoN', outputs)
        simu.perf('SoN', cell)
        
        #wager ratio
        if(index_max(network['SoN'].stateOutputNeurons) == 1):
            plot['high_wager'] += 1

    def step_learn(network, inputs, outputs):
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]
        #Learning
        network['SoN'].train(network['FoN'].stateHiddenNeurons, cell)
        network['FoN'].train(inputs, outputs)
        
    
    def moregraph1(plt):
        plt.title('Error RMS error of first-order and high-order networks')
        plt.ylabel('RMS ERROR')
        plt.xlabel("EPOCHS")
        
    def moregraph2(plt):
        plt.title('Classification performance of first-order and high-order networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
    
    
    sim = Simulation(nbr_epoch, width, data, nbr_network, [FoN, SoN])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'FoN_perf', 'SoN_perf', 'high_wager'], [])
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(point, 'FoN_rms', ['FoN_rms', 'SoN_rms'],
             ["FoN" , "SoN"],
             [3, 3 ], moregraph1)
    
    sim.plot(point, 'FoN_rms', ['FoN_perf', 'SoN_perf', 'high_wager'],
             ["FoN ( winner take all )" , "SoN (wagering) ", "High wager"],
             [3, 3, 2, ], moregraph2)
        
