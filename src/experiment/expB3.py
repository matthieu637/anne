'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron
from data import DataFileR
from utils import index_max, compare, compare_f
from representation import graph_network
from copy import deepcopy
from simulation import Simulation

class tmpObject:
    last = None
    last2 = None

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.3
    lrate = 0.1
    nbr_try = 10
    nbr_epoch = 1000
    nbOutputs = 3
    nbr_hidden = 5
    
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

#   Data Sample Declaration
    def data():
        return DataFileR("iris.txt", mode)
    
#   Network Declaration

    def FoN(inputs, outputs):
        tmpObject.last = MultilayerPerceptron(inputs, nbr_hidden, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
        return tmpObject.last
    def FoN2(inputs, outputs):
        return deepcopy(tmpObject.last)
    def SoN(inputs, outputs):
        tmpObject.last2 = MultilayerPerceptron(nbr_hidden, nbr_hidden * 2, inputs + nbr_hidden + outputs, learning_rate=lrate, momentum=momentum, grid=mode)
        return tmpObject.last2
    def SoN2(inputs, outputs):
        return deepcopy(tmpObject.last2)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        
        network['FoN2'].calc_output(inputs)
        network['SoN2'].calc_output(network['FoN2'].stateHiddenNeurons)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons + network['FoN'].stateOutputNeurons
        entire_FoN2 = inputs + network['FoN2'].stateHiddenNeurons + network['FoN2'].stateOutputNeurons
        
        lin = simu.examples.ninputs
        lhi = lin + nbr_hidden
        lout = lhi + simu.examples.noutputs
        
        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, entire_FoN)
        
        plot['SoN_rms_input'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, 0, lin) / lout * (lin - 0)
        plot['SoN_rms_hidden'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, lin, lhi) / lout * (lhi - lin)
        plot['SoN_rms_output'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, lhi, lout) / lout * (lout - lhi)
        
        simu.rms('FoN2', inputs, outputs)
        simu.rms('SoN2', network['FoN2'].stateHiddenNeurons, entire_FoN2)
        
        plot['SoN2_rms_input'] += network['SoN2'].calc_RMS_range(network['FoN2'].stateHiddenNeurons, entire_FoN2, 0, lin) / lout * (lin - 0)
        plot['SoN2_rms_hidden'] += network['SoN2'].calc_RMS_range(network['FoN2'].stateHiddenNeurons, entire_FoN2, lin, lhi) / lout * (lhi - lin)
        plot['SoN2_rms_output'] += network['SoN2'].calc_RMS_range(network['FoN2'].stateHiddenNeurons, entire_FoN2, lhi, lout) / lout * (lout - lhi)
        
        #err
        simu.err('FoN', outputs)
        plot['SoN_err_input'] += 1 - compare(inputs, network['SoN'].stateOutputNeurons[0:lin])
        if(not compare_f(network['FoN'].stateHiddenNeurons, network['SoN'].stateOutputNeurons[lin:lhi], 0.3)):
            plot['SoN_err_hidden'] += 1
        if(index_max(network['SoN'].stateOutputNeurons[lhi:lout]) != index_max(network['FoN'].stateOutputNeurons)):
            plot['SoN_err_output'] += 1
            
        simu.err('FoN2', outputs)
        plot['SoN2_err_input'] += 1 - compare(inputs, network['SoN2'].stateOutputNeurons[0:lin])
        if(not compare_f(network['FoN2'].stateHiddenNeurons, network['SoN2'].stateOutputNeurons[lin:lhi], 0.3)):
            plot['SoN2_err_hidden'] += 1
        if(index_max(network['SoN2'].stateOutputNeurons[lhi:lout]) != index_max(network['FoN2'].stateOutputNeurons)):
            plot['SoN2_err_output'] += 1
            
    def step_learn(network, inputs, outputs):
        #Learning
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons + network['FoN'].stateOutputNeurons
        network['SoN'].train(network['FoN'].stateHiddenNeurons, entire_FoN)
        network['FoN'].train(inputs, outputs)
        
        entire_FoN2 = inputs + network['FoN2'].stateHiddenNeurons + network['FoN2'].stateOutputNeurons
        network['SoN2'].train(network['FoN2'].stateHiddenNeurons, entire_FoN2)
        network['FoN2'].train(inputs, outputs)
        
    def step_learn2(network, inputs, outputs):
        #Learning
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons + network['FoN'].stateOutputNeurons
        network['SoN'].train(network['FoN'].stateHiddenNeurons, entire_FoN)
        network['FoN'].train(inputs, outputs)
        
        entire_FoN2 = inputs + network['FoN2'].stateHiddenNeurons + network['FoN2'].stateOutputNeurons
        network['SoN2'].train(network['FoN2'].stateHiddenNeurons, entire_FoN2)
        
        for i in range(len(outputs)):
            network['FoN2'].outputNeurons[i].train(network['FoN2'].stateHiddenNeurons, outputs[i])
        
    
    def moregraph1(plt):
        plt.title('Error RMS error of first-order and high-order networks')
        plt.ylabel('RMS ERROR')
        plt.xlabel("EPOCHS")
        
    def moregraph2(plt):
        plt.title('Classification error of first-order and high-order networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch*4, -0.01, 1.01))
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [FoN, SoN, FoN2, SoN2])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'SoN_rms_input', 'SoN_rms_hidden', 'SoN_rms_output', 'FoN_err',
                'SoN_err_input', 'SoN_err_hidden', 'SoN_err_output', 'FoN2_rms', 'SoN2_rms', 'SoN2_rms_input',
                'SoN2_rms_hidden', 'SoN2_rms_output', 'FoN2_err',
                'SoN2_err_input', 'SoN2_err_hidden', 'SoN2_err_output'], [])
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    for k in range(len(sim.examples.outputs)):
        newtask(sim.examples.outputs[k])
        
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn2)
    
    for k in range(len(sim.examples.outputs)):
        newtask2(sim.examples.outputs[k])
        
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn2)
     
    for k in range(len(sim.examples.outputs)):
        newtask3(sim.examples.outputs[k])
        
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn2)
    
    sim.plot(48, 'FoN_rms', ['FoN_rms', 'SoN_rms', 'SoN_rms_input', 'SoN_rms_hidden', 'SoN_rms_output'],
             ["FoN" , "SoN", "SoN input layer", "SoN hidden layer", "SoN output layer"],
             [3, 3, 2, 2 , 2 ], moregraph1)
    
    
    sim.plot(48, 'FoN_rms', ['FoN_err', 'SoN_err_input', 'SoN_err_hidden', 'SoN_err_output'],
             ["FoN ( winner take all )" , "SoN input layer ( | x - o | <= 0.3 )", "SoN hidden layer ( | x - o | <= 0.3 )",
                        "SoN output layer ( winner take all )"],
             [3, 2, 2 , 2 ], moregraph2)
    
    sim.plot(48, 'FoN2_rms', ['FoN2_rms', 'SoN2_rms', 'SoN2_rms_input', 'SoN2_rms_hidden', 'SoN2_rms_output'],
             ["FoN" , "SoN", "SoN input layer", "SoN hidden layer", "SoN output layer"],
             [3, 3, 2, 2 , 2 ], moregraph1)
    
    sim.plot(48, 'FoN_rms', ['FoN2_err', 'SoN2_err_input', 'SoN2_err_hidden', 'SoN2_err_output'],
             ["FoN ( winner take all )" , "SoN input layer ( | x - o | <= 0.3 )", "SoN hidden layer ( | x - o | <= 0.3 )",
                        "SoN output layer ( winner take all )"],
             [3, 2, 2 , 2 ], moregraph2)
