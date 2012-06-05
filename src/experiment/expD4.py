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
from prenforcement import PRenforcement


def l_to_lr(l):
    w = 0.
    for i in range(3):
        w += l[i] * (2 ** i)
    return w


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
    point = 4
    nbHidden = 20
    
#   Data Sample Declaration
    def data():
        return DataFile("digit_handwritten_16.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        seed(tmpObject.i)
        tmpObject.i = tmpObject.i + 1
        m = MultilayerPerceptron(inputs, nbHidden, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
        m.init_weights_randomly(-1, 1)
        tmpObject.net = m
        return m
    def SoN(inputs, outputs):
        return [PRenforcement(nbHidden, 0.3, 0.03, 2., True) for _ in range(6)]
    
    def control(inputs, outputs):
        return deepcopy(tmpObject.net)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        res = [network['SoN'][i].calc_output(network['FoN'].stateHiddenNeurons) for i in range(6)]
        network['control'].calc_output(inputs)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('control', inputs, outputs)
        
        #err
        simu.perf('FoN', outputs)
        simu.perf('control', outputs)
        
        #momentum & lrate
        res = [network['SoN'][i].calc_output(network['FoN'].stateHiddenNeurons) for i in range(6)]
        l = 0.1 + l_to_lr(res[0:3]) * 2.5 / 70
        m = 0.2 + l_to_lr(res[3:6]) * 6.5 / 70
        
        plot['lrate'] += l
        plot['momentum'] += m
            
    def step_learn(network, inputs, outputs):
        #Learning
        res = [network['SoN'][i].calc_output(network['FoN'].stateHiddenNeurons) for i in range(6)]
        f_success = False
        if(index_max(network['FoN'].stateOutputNeurons) == index_max(outputs)):
            f_success = True
        
        l = 0.1 + l_to_lr(res[0:3]) * 2.5 / 70
        m = 0.2 + l_to_lr(res[3:6]) * 6.5 / 70
        
        [ network['SoN'][i].train(f_success) for i in range(6)]     
        network['FoN'].set_learning_rate(l)
        network['FoN'].set_momentum(m)
        network['FoN'].train(inputs, outputs)
        network['control'].train(inputs, outputs)
    
    def moregraph1(plt):
        plt.title('Error RMS error of first-order and high-order networks')
        plt.ylabel('RMS ERROR')
        plt.xlabel("EPOCHS")
        
    def moregraph3(plt):
        plt.title('Classification performance of first-order and control networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch, 0, 1.01))
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [FoN, SoN, control])
    sim.dgraph(['FoN_rms', 'FoN_perf', 'control_rms', 'control_perf', 'momentum', 'lrate'], [])
    seed(100)
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(point, 'FoN_rms', ['FoN_rms', 'control_rms'],
             ["FoN" , 'control'],
             [3, 2, 3 ], moregraph1)
    
    sim.plot(point, 'FoN_perf', ['FoN_perf', 'control_perf', 'momentum', 'lrate'],
             ["FoN" , "Control", 'momentum', 'learning rate'],
             [3, 3, 2, 2 ], moregraph3)
        
