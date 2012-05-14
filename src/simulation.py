'''
Created on 14 May 2012

@author: Matthieu Zimmer
'''

from random import shuffle
import matplotlib.pyplot as plt

class Simulation():
    '''
    classdocs
    '''


    def __init__(self, nbr_network, build):
        '''
        Constructor
        '''
        
        self.networks = [{} for _ in range(nbr_network)]
        self.plots = {}
        self.nbr_network = nbr_network
        
        for k in build.keys():
            for i in range(nbr_network):
                self.networks[i][k] = build[k]()
            self.plots[k] = []
              
    def onData(self, dataf):
        self.examples = dataf()
        
    
    def launch(self, nbr_epoch, nbr_try, func):
        
        for epoch in range(nbr_epoch):
            avg_plot = {}
            for k in self.plots.keys():
                avg_plot[k] = 0.
                
            for network in self.networks:
                full_example = list(range(len(self.examples.inputs)))
                shuffle(full_example)
                for i_ex in full_example[0:nbr_try]:
                    func(network, avg_plot, 
                         self.examples.inputs[i_ex], 
                         self.examples.outputs[i_ex])
                    
                    
            for k in avg_plot.keys():
                self.plots[k].append(avg_plot[k] / (nbr_try * self.nbr_network))
                    
            print(epoch)
    
    
    def plot(self, decoup, titles, linewidths, more):
        
        k = list(self.plots.keys())[0]
        display_interval=range(len(self.plots[k]))[::decoup]
        
        for i in range(len(self.plots)):
            key = list(self.plots.keys())[i]
            plt.plot(display_interval, [self.plots[key][j] for j in display_interval],
             label=titles[i],
             linewidth=linewidths[i])
        
        more(plt)

        plt.legend(loc='best', frameon=False)
        plt.show()
        
        
        
        
        
    