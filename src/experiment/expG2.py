'''
Created on 27 May 2012

@author: Matthieu Zimmer
'''

from multilayerp import MultilayerPerceptron
from perceptron import PerceptronR0to1
from data import DataFile
from utils import index_max, randmm
import representation
from simulation import Simulation
from random import seed
from copy import deepcopy






class AdHockP(PerceptronR0to1):
    def __init__(self, control):
        self.nbr_input = control.nbr_input
        self.learning_rate = control.learning_rate
        self.momentum = control.momentum
        self.temperature = control.temperature
        self._ntype = control._ntype
        self._enable_bias = control._enable_bias
        

        self.weights = []        
        for w in control.weights:
            self.weights.append(w)
        
        for _ in range(20):
            self.weights.append(randmm(-1, 1))
        
        self.bias = control.bias
        self._last_bias = control._last_bias
        
        self._last_weights = list(self.weights)
        
        self._weights_updated = True
        self._last_inputs = []
    
    def update_weights_1(self, error, inputs, addition):
        self.calc_output(inputs + addition)
        
        ii = len(inputs)
        tmp_weights = list(self.weights[0:ii])
        
        for j in range(len(inputs)):
            dw = self.weights[j] - self._last_weights[j] 
#            print(dw)
            self.weights[j] += self.learning_rate * error * inputs[j] + self.momentum * dw
        
        if self._enable_bias:
            tmp_bias = self.bias 
            self.bias += self.learning_rate * error + self.momentum * (self.bias - self._last_bias)
            self._last_bias = tmp_bias
        
        
        self._last_weights[0:ii] = tmp_weights
        self._weights_updated = True
        
    def update_weights_2(self, output, beg, inputs, hel):
        self.calc_output(hel + inputs)
        
        ii = len(inputs)
        tmp_weights = list(self.weights[beg:beg + ii])
        

        for j in range(ii):
            dw = self.weights[beg + j] - self._last_weights[beg + j] 
            self.weights[beg + j] += (self.learning_rate * (output - self._state) * inputs[j] + self.momentum * dw) / (len(inputs))

        
        self._last_weights[beg:beg + ii] = tmp_weights
        self._weights_updated = True


class AdHock(MultilayerPerceptron):
    def __init__(self, control):
        self.hiddenNeurons = []
        self.outputNeurons = []
        
        nbr_hidden = len(control.hiddenNeurons)
        nbr_output = len(control.outputNeurons)
    
        for i in range(nbr_hidden):
            ph = deepcopy(control.hiddenNeurons[i])
#            PerceptronR0to1(nbr_input, learning_rate, momentum, temperature, Perceptron.HIDDEN, random, enable_bias)
            self.hiddenNeurons.append(ph)
            
        for j in range(nbr_output):
            po = AdHockP(control.outputNeurons[j])
            self.outputNeurons.append(po)
                
        self.stateOutputNeurons = []
        self.stateHiddenNeurons = []
        self._last_inputs = []
        self._network_updated = True
    def calc_SE_range(self, inputs, outputs, imin, imax):
        #self.calc_output(inputs)
        
        s = 0.
        for i in range(imin, imax):
            s += (self.stateOutputNeurons[i] - outputs[i]) ** 2
        return s

    def calc_hidden(self, inputs):
        #determine the state of hidden neurons
        stateHidden = []
        for neuron in self.hiddenNeurons :
            stateHidden.append(neuron.calc_output(inputs))
        self.stateHiddenNeurons = stateHidden

    def calc_output(self, addition):
        #then the output layer
        stateOutputs = []
        for neuron in self.outputNeurons :
            stateOutputs.append(neuron.calc_output(self.stateHiddenNeurons + addition))
        self.stateOutputNeurons = stateOutputs
        return stateOutputs
    
    def train(self, inputs, outputs, addition):
        #build y the error vector to propagate
        y = []
        for i in range(len(self.outputNeurons)) :
            y.append(self.outputNeurons[i].calc_error_propagation(outputs[i]))
        
        yy = []
        for i in range(len(self.hiddenNeurons)):
            w_sum = 0.
            for j in range(len(self.outputNeurons)) :
                w_sum += self.outputNeurons[j].weights[i] * y[j]
            yy.append(self.hiddenNeurons[i].calc_error_propagation(w_sum))
            
        #updates all weights of the network
        for i in range(len(self.hiddenNeurons)) :
            self.hiddenNeurons[i].update_weights(yy[i] , inputs)
 
        for i in range(len(self.outputNeurons)) :
            self.outputNeurons[i].update_weights_1(y[i] , self.stateHiddenNeurons, addition)
            self.outputNeurons[i].update_weights_2(outputs[i], len(self.stateHiddenNeurons), addition, self.stateHiddenNeurons)
            
        self._network_updated = True
        





class tmpObject:
    i = 0
    control = None

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
    def control(inputs, outputs):
        seed(tmpObject.i)
        tmpObject.i = tmpObject.i + 1
        tmpObject.control = MultilayerPerceptron(inputs, nbHidden, outputs, learning_rate=lrate, momentum=momentum, grid=mode)
        tmpObject.control.init_weights_randomly(-1, 1)
        return tmpObject.control
    def FoN(inputs, outputs):
        return AdHock(tmpObject.control)
    def SoN(inputs, outputs):
        return MultilayerPerceptron(nbHidden, 20, 2, learning_rate=0.1, momentum=0., grid=mode)

    
#   Work on one step
    def step_propagation(network, inputs, outputs):
        network['control'].calc_output(inputs)
        network['FoN'].calc_hidden(inputs)
        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
        network['FoN'].calc_output([0] * 20)
        
        
    def step_statictics(simu, network, plot, inputs, outputs):


        #rms
        simu.rms('control', inputs, outputs)
        
        
        #err
        simu.perf('control', outputs)
        

        network['FoN'].calc_output(network['SoN'].stateHiddenNeurons)
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]
        simu.perf('SoN', cell)
        simu.rms('SoN', network['FoN'].stateHiddenNeurons, cell)
        #wager ratio
        if(index_max(network['SoN'].stateOutputNeurons) == 1):
            plot['high_wager'] += 1

       
        simu.rms('FoN', None, outputs)
        simu.perf('FoN', outputs)

    def step_learn(network, inputs, outputs):
        cell = [0., 0.]
        if index_max(network['FoN'].stateOutputNeurons) == index_max(outputs):
            cell = [0., 1]
        else:
            cell = [1, 0.]
        #Learning
        tmp = list(network['FoN'].stateHiddenNeurons)
        network['FoN'].train(inputs, outputs, network['SoN'].stateHiddenNeurons)
        
        network['SoN'].train(tmp, cell)
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
    
    
    sim = Simulation(nbr_epoch, 0, data, nbr_network, [control, FoN, SoN])
    sim.dgraph(['FoN_rms', 'SoN_rms', 'FoN_perf', 'SoN_perf', 'high_wager', 'control_rms', 'control_perf'], [])
    seed(100)
    sim.launch(nbr_try, step_propagation, step_statictics, step_learn)
    
    sim.plot(point, 'FoN_rms', ['FoN_rms', 'SoN_rms', 'control_rms'],
             ["FoN" , "SoN", "Control"],
             [3, 3, 3 ], moregraph1)
    
    sim.plot(point, 'FoN_rms', ['FoN_perf', 'SoN_perf', 'high_wager'],
             ["FoN ( winner take all )" , "SoN (wagering) ", "High wager"],
             [3, 3, 2 ], moregraph2)
    
    sim.plot(point, 'FoN_perf', ['FoN_perf', 'control_perf'],
             ["FoN" , "Control"],
             [3, 3 ], moregraph3)
        
