'''
Created on 14 May 2012

@author: Matthieu Zimmer
'''

from random import shuffle
import matplotlib.pyplot as plt
from utils import index_max
import os, sys, time

class Simulation():
    '''
    classdocs
    '''


    def __init__(self, dataf, nbr_network, build):
        '''
        Constructor
        '''
        self.examples = dataf()
        self.networks = [{} for _ in range(nbr_network)]
        self.plots = {}
        self.nbr_network = nbr_network
        
        for k in build:
            for i in range(nbr_network):
                self.networks[i][k.__name__] = k(self.examples.ninputs, self.examples.noutputs )
        
    def dgraph(self, glist):
        for k in glist: 
            self.plots[k] = []
    
    def launch(self, nbr_epoch, nbr_try, step_propagation, step_statictics, step_learn):
        
        for epoch in range(nbr_epoch):
            avg_plot = {}
            for k in self.plots.keys():
                avg_plot[k] = 0.
                
            for network in self.networks:
                full_example = list(range(len(self.examples.inputs)))
                shuffle(full_example)
                for i_ex in full_example[0:nbr_try]:
                    self.lnetwork = network
                    self.lavg_plot = avg_plot
                    
                    step_propagation(network,
                         self.examples.inputs[i_ex], 
                         self.examples.outputs[i_ex])
                    
                    step_statictics(self, network, avg_plot, 
                         self.examples.inputs[i_ex], 
                         self.examples.outputs[i_ex])
                    
                    step_learn(network,
                         self.examples.inputs[i_ex], 
                         self.examples.outputs[i_ex])
                    
                    
            for k in avg_plot.keys():
                self.plots[k].append(avg_plot[k] / (nbr_try * self.nbr_network))
                    
            print("[%d] %d" % (os.getpid(),epoch))
    
    
    def plot(self, decoup, lplots, titles, linewidths, more):
        
        k = list(self.plots.keys())[0]
        display_interval=range(len(self.plots[k]))[::decoup]
        
        for i in range(len(lplots)):
            plt.plot(display_interval, [self.plots[lplots[i]][j] for j in display_interval],
             label=titles[i],
             linewidth=linewidths[i])
        
        more(plt)

        plt.legend(loc='best', frameon=False)
        
        path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()) )
        plt.savefig(path)
        plt.show()
        
        
    def rms(self, key, inputs, outputs):
        self.lavg_plot[key + '_rms'] += self.lnetwork[key].calc_RMS(inputs, outputs)
        
    def err(self, key, outputs):
        if(index_max(self.lnetwork[key].stateOutputNeurons) != index_max(outputs)):
            self.lavg_plot[key + '_err'] += 1
        
        
        
    