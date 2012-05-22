'''
Created on 14 mai 2012

@author: matthieu
'''


from multilayerp import MultilayerPerceptron
from data import DataFile
from utils import index_max, compare, compare_f

from simulation import Simulation

if __name__ == '__main__':
    mode = MultilayerPerceptron.R0to1
    nbr_network = 1
    momentum = 0.9
    lrate = 0.1
    nbr_try = 10
    nbr_epoch = 1000
    
#   Data Sample Declaration
    def data():
        return DataFile("digit_handwritten_16.txt", mode)
#        return DataFile("digit_shape.txt", mode)
    
#   Network Declaration
    def FoN(inputs, outputs):
        return MultilayerPerceptron(inputs, inputs//4, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
    def SoN(inputs, outputs):
        return MultilayerPerceptron(inputs//4, inputs//2, inputs + inputs//4 + outputs, learning_rate=lrate, momentum=momentum, grid=mode)
    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['FoN'].calc_output(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        
    def step_statictics(simu, network, plot, inputs, outputs):
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons +  network['FoN'].stateOutputNeurons
        
        lin = simu.examples.ninputs
        lhi = lin + lin//4
        lout = lhi + simu.examples.noutputs
        
        #rms
        simu.rms('FoN', inputs, outputs)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, entire_FoN)
        
        plot['SoN_rms_input'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, 0, lin) / 35 * 20
        plot['SoN_rms_hidden'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, lin, lhi) / 35 * 5
        plot['SoN_rms_output'] += network['SoN'].calc_RMS_range(network['FoN'].stateHiddenNeurons, entire_FoN, lhi, lout) / 35 * 10
        
        #err
        simu.err('FoN', outputs)
        plot['SoN_err_input'] += 1 - compare(inputs, network['SoN'].stateOutputNeurons[0:lin])
        if( not compare_f(network['FoN'].stateHiddenNeurons, network['SoN'].stateOutputNeurons[lin:lhi], 0.3) ):
            plot['SoN_err_hidden'] += 1
        if(index_max(network['SoN'].stateOutputNeurons[lhi:lout]) != index_max(network['FoN'].stateOutputNeurons)):
            plot['SoN_err_output'] += 1

    def step_learn(network, inputs, outputs):
        #Learning
        entire_FoN = inputs + network['FoN'].stateHiddenNeurons +  network['FoN'].stateOutputNeurons
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
    
    
    sim = Simulation(data, nbr_network, [FoN, SoN])
    sim.dgraph(['FoN_rms','SoN_rms', 'SoN_rms_input', 'SoN_rms_hidden', 'SoN_rms_output', 'FoN_err', 
                'SoN_err_input', 'SoN_err_hidden', 'SoN_err_output'])
    sim.launch(nbr_epoch, nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(6, ['FoN_rms','SoN_rms', 'SoN_rms_input', 'SoN_rms_hidden', 'SoN_rms_output'],
             ["FoN" , "SoN", "SoN input layer", "SoN hidden layer", "SoN output layer"], 
             [2, 2, 1, 1 ,1 ], moregraph1)
    
    
    sim.plot(6, ['FoN_err', 'SoN_err_input', 'SoN_err_hidden', 'SoN_err_output'],
             ["FoN ( winner take all )" , "SoN input layer ( x > 0.5 => activation )", "SoN hidden layer ( | x - o | <= 0.3 )", 
                        "SoN output layer ( winner take all )"], 
             [2, 1, 1 ,1 ], moregraph2)
    
    