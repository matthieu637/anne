'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron
from perceptron import PerceptronR0to1
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
        return MultilayerPerceptron(nbHidden, 20, 2, learning_rate=0.1, momentum=0., grid=mode)
    
    def feedback(inputs, outputs):
        return [PerceptronR0to1(nbHidden + 20, 0.1, 0., True) for _ in range(10)]
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        
        
    def step_statictics(simu, network, plot, inputs, outputs):
        res = [ network['feedback'][i].calc_output(network['FoN'].stateHiddenNeurons + 
                                           network['SoN'].stateHiddenNeurons) for i in range(10)]
        
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
            
        #feedback
        if(index_max(res) == index_max(outputs)):
            plot['feedback'] += 1

    def step_learn(network, inputs, outputs):
        for i in range(10):
            network['feedback'][i].train(network['FoN'].stateHiddenNeurons + 
                                         network['SoN'].stateHiddenNeurons
                                         , outputs[i])
        
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
        plt.axis((0, nbr_epoch, 0, 1.01))
        
    def moregraph3(plt):
        plt.title('Classification performance of first-order and feedback networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch, 0, 1.01))
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [FoN, SoN, feedback])
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
        
