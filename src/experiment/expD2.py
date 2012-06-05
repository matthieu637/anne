'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron
from data import DataFile
from utils import index_max, index_max_nth
import representation
from simulation import Simulation
from random import seed

class tmpObject:
    i = 0

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.5
    lrate = 0.15
    nbr_try = 50
    nbr_epoch = 300
    point = 3
    nbHidden = 100
    
#   Data Sample Declaration
    def data():
        return DataFile("digit_handwritten_16.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        seed(tmpObject.i)
        tmpObject.i=tmpObject.i+1
        m = MultilayerPerceptron(inputs, nbHidden, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
        m.init_weights_randomly(-1, 1)
        return m
    def SoN(inputs, outputs):
        return MultilayerPerceptron(nbHidden, 20, 10, learning_rate=0.1, momentum=0., grid=mode)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        cell = [0 for _ in range(10)]
        for k in range(10):
            if index_max_nth(network['FoN'].stateOutputNeurons, k) == index_max(outputs):
                cell[k] = 1
                break

        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, cell)
        
        #err
        simu.perf('FoN', outputs)
        simu.perf('SoN', cell)
        
        #wager ratio
        if(index_max(network['SoN'].stateOutputNeurons) == 0):
            plot['high_wager'] += 1
        
        #feedback
        if(index_max_nth(network['FoN'].stateOutputNeurons, index_max(network['SoN'].stateOutputNeurons)) == index_max(outputs)):
            plot['feedback'] += 1

    def step_learn(network, inputs, outputs):
        cell = [0 for _ in range(10)]
        for k in range(10):
            if index_max_nth(network['FoN'].stateOutputNeurons, k) == index_max(outputs):
                cell[k] = 1
                break
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
        plt.axis((0, nbr_epoch, 0, 1.01))
        
    def moregraph3(plt):
        plt.title('Classification performance of first-order and feedback networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch, 0, 1.01))
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [FoN, SoN])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'FoN_perf', 'SoN_perf', 'high_wager', 'feedback'], [])
    seed(100)
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(point, 'FoN_rms', ['FoN_rms', 'SoN_rms'],
             ["FoN" , "SoN"],
             [3, 3 ], moregraph1)
    
    sim.plot(point, 'FoN_rms', ['FoN_perf', 'SoN_perf', 'high_wager'],
             ["FoN ( winner take all )" , "SoN (wagering) ", "High wager"],
             [3, 3, 2 ], moregraph2)
    
    sim.plot(point, 'FoN_perf', ['FoN_perf', 'feedback'],
             ["FoN" , "Feedback"],
             [3, 3 ], moregraph3)
        
