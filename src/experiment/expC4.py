'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron, MultilayerPerceptronM
from data import DataFileR
from utils import index_max
import representation
from simulation import Simulation
from random import seed

class tmpObject:
    i = 0

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.5
    lrate = 0.15
    nbr_try = 30
    nbr_epoch = 300
    point = 3
    nbHidden = 160
    
#   Data Sample Declaration
    def data():
        return DataFileR("iris.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        seed(tmpObject.i)
        tmpObject.i=tmpObject.i+1
        m = MultilayerPerceptronM(inputs, nbHidden, outputs, 3, learning_rate=lrate, momentum=momentum, grid=mode)
        m.init_weights_randomly(-1, 1)
        return m
    def SoN(inputs, outputs):
        n = MultilayerPerceptron(nbHidden+inputs, 20, 2, learning_rate=0.1, momentum=0.5, grid=mode)
        n.init_weights_randomly(-1, 1)
        return n
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons[1]+inputs)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]

        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons[1]+inputs, cell)
        
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
        network['SoN'].train(network['FoN'].stateHiddenNeurons[1]+inputs, cell)
        network['FoN'].train(inputs, outputs)
        
    
    def moregraph1(plt):
        plt.title('Error RMS error of first-order and high-order networks')
        plt.ylabel('RMS ERROR')
        plt.xlabel("EPOCHS")
        
    def moregraph2(plt):
        plt.title('Classification performance of first-order and high-order networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch, 0, 1.01))
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [FoN, SoN])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'FoN_perf', 'SoN_perf', 'high_wager'], [])
    seed(100)
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(point, 'FoN_rms', ['FoN_rms', 'SoN_rms'],
             ["FoN" , "SoN"],
             [3, 3 ], moregraph1)
    
    sim.plot(point, 'FoN_rms', ['FoN_perf', 'SoN_perf', 'high_wager'],
             ["FoN ( winner take all )" , "SoN (wagering) ", "High wager"],
             [3, 3, 2, ], moregraph2)
        
