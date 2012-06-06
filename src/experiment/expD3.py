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
from copy import deepcopy

class tmpObject:
    i = 0
    net = None

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
        tmpObject.net = m
        return m
    def SoN(inputs, outputs):
        return MultilayerPerceptron(nbHidden, 20, 2, learning_rate=0.1, momentum=0., grid=mode)
    
    def control(inputs, outputs):
        return deepcopy(tmpObject.net)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        network['control'].calc_output(inputs)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]

        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, cell)
        simu.rms('control', inputs, outputs)
        
        #err
        simu.perf('FoN', outputs)
        simu.perf('SoN', cell)
        simu.perf('control', outputs)
        
        #wager ratio
        if(index_max(network['SoN'].stateOutputNeurons) == 1):
            plot['high_wager'] += 1
            
        #momentum & lrate
        tmp = list(network['SoN'].stateOutputNeurons)
        if(index_max(tmp) == 0):
            plot['lrate'] += (0.15 + (0.15 - 0.025))
            plot['momentum'] += 0.2
        elif(abs(tmp[0] - tmp[1]) <= 0.3):
            plot['lrate'] +=(0.15)
            plot['momentum'] +=(0.5)
        else:
            plot['lrate'] +=(0.025)
            plot['momentum'] +=(0.7)
            
    def step_learn(network, inputs, outputs):
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]
        #Learning
        tmp = list(network['SoN'].stateOutputNeurons)
        network['SoN'].train(network['FoN'].stateHiddenNeurons, cell)
        if(index_max(tmp) == 0):
            network['FoN'].set_learning_rate(0.15 + (0.15 - 0.025))
            network['FoN'].set_momentum(0.2)
        elif(abs(tmp[0] - tmp[1]) <= 0.3):
            network['FoN'].set_learning_rate(0.15)
            network['FoN'].set_momentum(0.5)
        else:
            network['FoN'].set_learning_rate(0.025)
            network['FoN'].set_momentum(0.7)
        network['FoN'].train(inputs, outputs)
        network['control'].train(inputs, outputs)
        
    
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
        plt.title('Classification performance of first-order and control networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch, 0, 1.01))
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [FoN, SoN, control])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'FoN_perf', 'SoN_perf', 'high_wager', 'control_rms', 'control_perf', 'momentum', 'lrate'], [])
    seed(100)
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(point, 'FoN_rms', ['FoN_rms', 'SoN_rms', 'control_rms'],
             ["FoN" , "SoN", 'control'],
             [3, 2, 3 ], moregraph1)
    
    sim.plot(point, 'FoN_rms', ['FoN_perf', 'SoN_perf', 'high_wager'],
             ["FoN ( winner take all )" , "SoN (wagering) ", "High wager"],
             [3, 3, 2 ], moregraph2)
    
    sim.plot(point, 'FoN_perf', ['FoN_perf', 'control_perf', 'momentum', 'lrate'],
             ["FoN" , "Control", 'momentum', 'learning rate'],
             [3, 3, 2, 2 ], moregraph3)
        
