'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron
from data import DataFileR
from utils import index_max, compare, compare_f
from representation import graph_network


from simulation import Simulation

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 5
    momentum = 0.9
    lrate = 0.1
    nbr_try = 10
    nbr_epoch = 1000
    width = 4
    discretize = 4
    nbr_hidden = 5
    
#   Data Sample Declaration
    def data():
        return DataFileR("iris.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        return MultilayerPerceptron(inputs, nbr_hidden, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
    def SoN(inputs, outputs):
        return MultilayerPerceptron(nbr_hidden, nbr_hidden * 2, inputs + nbr_hidden + outputs, learning_rate=lrate, momentum=momentum, grid=mode)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons + network['FoN'].stateOutputNeurons
        
        lin = simu.examples.ninputs
        lhi = lin + nbr_hidden
        lout = lhi + simu.examples.noutputs
        
        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, entire_FoN)
        
        plot['SoN_rms_input'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, 0, lin) / lout * (lin - 0)
        plot['SoN_rms_hidden'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, lin, lhi) / lout * (lhi - lin)
        plot['SoN_rms_output'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, lhi, lout) / lout * (lout - lhi)
        
        #err
        simu.err('FoN', outputs)
        if(not compare_f(inputs, network['SoN'].stateOutputNeurons[0:lin], 0.3)):
            plot['SoN_err_input'] += 1
        if(not compare_f(network['FoN'].stateHiddenNeurons, network['SoN'].stateOutputNeurons[lin:lhi], 0.3)):
            plot['SoN_err_hidden'] += 1
        if(index_max(network['SoN'].stateOutputNeurons[lhi:lout]) != index_max(network['FoN'].stateOutputNeurons)):
            plot['SoN_err_output'] += 1
            
        #discretize
        simu.discretize('FoN', index_max(outputs), discretize)

    def step_learn(network, inputs, outputs):
        #Learning
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons + network['FoN'].stateOutputNeurons
        network['SoN'].train(network['FoN'].stateHiddenNeurons, entire_FoN)
        network['FoN'].train(inputs, outputs)
        
    
    def moregraph1(plt):
        plt.title('Error RMS error of first-order and high-order networks')
        plt.ylabel('RMS ERROR')
        plt.xlabel("EPOCHS")
        
    def moregraph2(plt):
        plt.title('Classification error of first-order and high-order networks')
        plt.ylabel('ERROR')
        plt.xlabel("EPOCHS")
        plt.axis((0, nbr_epoch, 0, 1.05))
    
    
    sim = Simulation(nbr_epoch, width, data, nbr_network, [FoN, SoN])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'SoN_rms_input', 'SoN_rms_hidden', 'SoN_rms_output', 'FoN_err',
                'SoN_err_input', 'SoN_err_hidden', 'SoN_err_output'], [Simulation.DISCRETIZE])
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(12, 'FoN_rms', ['FoN_rms', 'SoN_rms', 'SoN_rms_input', 'SoN_rms_hidden', 'SoN_rms_output'],
             ["FoN" , "SoN", "SoN input layer", "SoN hidden layer", "SoN output layer"],
             [3, 3, 2, 2 , 2 ], moregraph1)
    
    
    sim.plot(12, 'FoN_rms', ['FoN_err', 'SoN_err_input', 'SoN_err_hidden', 'SoN_err_output'],
             ["FoN ( winner take all )" , "SoN input layer ( x > 0.5 => activation )", "SoN hidden layer ( | x - o | <= 0.3 )",
                        "SoN output layer ( winner take all )"],
             [3, 2, 2 , 2 ], moregraph2)
    
    sim.custom_plot([Simulation.DISCRETIZE, Simulation.PROTOTYPE])
    
    #Representations
    on = []
    hn = []
    for net in sim.networks:
        on.append(net['FoN'].outputNeurons)
        hn.append(net['FoN'].hiddenNeurons)
    graph_network(on, hn, width)
    
