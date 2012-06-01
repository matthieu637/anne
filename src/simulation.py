'''
Created on 14 May 2012

@author: Matthieu Zimmer
'''

from random import shuffle
import matplotlib.pyplot as plt
from utils import index_max
import os, sys, time
from mpl_toolkits.mplot3d import Axes3D as ax
import representation as rpr

class Simulation():
    '''
    classdocs
    '''
    (DISCRETIZE, PROTOTYPE) = range(2)

    def __init__(self, nbr_epoch, width, dataf, nbr_network, build):
        '''
        Constructor
        '''
        self.examples = dataf()
        self.networks = [{} for _ in range(nbr_network)]
        self.plots = {}
        self.nbr_network = nbr_network
        self.nbr_epoch = nbr_epoch
        self.width = width
        
        for k in build:
            for i in range(nbr_network):
                self.networks[i][k.__name__] = k(self.examples.ninputs, self.examples.noutputs)
        
    def dgraph(self, glist, additional):
        for k in glist: 
            self.plots[k] = []
        for k in additional:
            if(k == Simulation.DISCRETIZE):
                self.plots['discretize'] = [[0 for _ in range(self.nbr_epoch)] for _ in range(self.examples.noutputs)]
                self.plots['discretize_div'] = [[0 for _ in range(self.nbr_epoch)] for _ in range(self.examples.noutputs)]
                self.plots['discretize_valid'] = [[] for _ in range(self.examples.noutputs)]
                
    
    def launch(self, nbr_try, step_propagation, step_statictics, step_learn):
        
        for epoch in range(self.nbr_epoch):
            avg_plot = {}
            for k in self.plots.keys():
                avg_plot[k] = 0.
                
            for network in self.networks:
                full_example = list(range(len(self.examples.inputs)))
                shuffle(full_example)
                for i_ex in full_example[0:nbr_try]:
                    self.lnetwork = network
                    self.lavg_plot = avg_plot
                    self.lepoch = epoch
                    
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
                    
            print("[%d] %d" % (os.getpid(), epoch))
    
    
    def plot(self, decoup, init_key, lplots, titles, linewidths, more):
        
        display_interval = range(len(self.plots[init_key]))[::decoup]
        for i in range(len(lplots)):
            if(lplots[i]=='feedback'):
                plt.plot(display_interval, [self.plots[lplots[i]][j] for j in display_interval],
                 label=titles[i], linewidth=linewidths[i], color='red')
            else :
                plt.plot(display_interval, [self.plots[lplots[i]][j] for j in display_interval],
                 label=titles[i],
                 linewidth=linewidths[i])
        
        more(plt)

        plt.legend(loc='best', frameon=False)
        
        path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
        plt.savefig(path)
        plt.show()
    
    def custom_plot(self, additional):
        for k in additional:
            if(k == Simulation.DISCRETIZE):
                for i in range(self.examples.noutputs):
                    for j in range(self.nbr_epoch):
                        if(self.plots['discretize_div'][i][j] != 0):
                            self.plots['discretize'][i][j] /= (self.plots['discretize_div'][i][j] * (self.nbDiscre ** self.nbDiscre))
                
                colors = [(0.2, 0.8, 0.88), 'b', 'g', 'r', 'c', 'm', 'y', 'k', (0.8, 0.1, 0.8), (0., 0.2, 0.5)]
        
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for j in range(self.examples.noutputs):
                    ax.scatter([self.plots['discretize'][j][k] for k in self.plots['discretize_valid'][j]], [j] * 
                                len(self.plots['discretize_valid'][j]), self.plots['discretize_valid'][j], color=colors[j], marker='x')
            
                ax.set_xlabel('DISCRETIZED VALUE')
                ax.set_ylabel('SHAPE')
                ax.set_zlabel('EPOCH')
                
                path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
                plt.savefig(path)
                plt.show()
                
                
                plt.title('Discretize hidden layer')
                plt.ylabel('DISCRETIZED VALUE')
                plt.xlabel("EPOCHS")
                for j in range(self.examples.noutputs):
                    plt.plot(self.plots['discretize_valid'][j], [self.plots['discretize'][j][k] 
                                                                 for k in self.plots['discretize_valid'][j]], '.', color=colors[j])
                path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
                try:
                    plt.savefig(path)
                except ValueError:
                    print('Cannot save discretize_cloud')
                try:
                    plt.show()
                except ValueError:
                    print('Cannot display discretize_cloud')
            elif(k == Simulation.PROTOTYPE):
                lplot = [[0. for _ in range(self.examples.ninputs)] for _ in range(self.examples.noutputs)]
                for network in self.networks:
                    for i in range(len(self.examples.inputs)):
                        network['FoN'].calc_output(self.examples.inputs[i])
                        network['SoN'].calc_output(network['FoN'].stateHiddenNeurons)
                        
                        im = index_max(self.examples.outputs[i])
                        
                        for j in range(self.examples.ninputs):
                            lplot[im][j] += network['SoN'].stateOutputNeurons[j]
                
                fig = plt.figure()
                plt.clf()
                for i in range(self.examples.noutputs):
                    rpr.show_repr(lplot[i], self.width, fig, 250 + i, i)
                path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
                plt.savefig(path)
                plt.show()
                
        
    def rms(self, key, inputs, outputs):
        self.lavg_plot[key + '_rms'] += self.lnetwork[key].calc_RMS(inputs, outputs)
        
    def err(self, key, outputs):
        if(index_max(self.lnetwork[key].stateOutputNeurons) != index_max(outputs)):
            self.lavg_plot[key + '_err'] += 1
    
    def perf(self, key, outputs):
        if(index_max(self.lnetwork[key].stateOutputNeurons) == index_max(outputs)):
            self.lavg_plot[key + '_perf'] += 1
    
    def discretize(self, key, ishape, nbDiscre):
        self.nbDiscre = nbDiscre
        def nbdis(nb, nbDiscretized=nbDiscre):
            for i in range(1, nbDiscretized + 1):
                if(nb <= i / nbDiscretized):
                    return i - 1
            return nbDiscretized - 1
        
            
        def discretis(ll, nbDiscretized=nbDiscre):
            s = 0
            for i in range(len(ll)):
                s += (nbDiscretized ** i) * nbdis(ll[i], nbDiscretized)
            return s

        process = True
        try:
            if(self.ldiscretize != self.lnetwork):
                process = False
        except AttributeError:
            self.ldiscretize = self.lnetwork
        
        if(process):
            self.plots['discretize_div'][ishape][self.lepoch] += 1
            self.plots['discretize'][ishape][self.lepoch] += discretis(self.lnetwork[key].stateHiddenNeurons)
            
            if(len(self.plots['discretize_valid'][ishape]) == 0):
                self.plots['discretize_valid'][ishape].append(self.lepoch)
            elif(self.plots['discretize_valid'][ishape][len(self.plots['discretize_valid'][ishape]) - 1] != self.lepoch):
                self.plots['discretize_valid'][ishape].append(self.lepoch)
                
    def prototypes(self):
        lplot = [[0. for _ in range(self.examples.ninputs)] for _ in range(self.examples.noutputs)]
        for i in range(self.examples.noutputs):
            for j in range(self.examples.ninputs):
                for k in range(len(self.examples.inputs)):
                    if(i == index_max(self.examples.outputs[k])):
                        lplot[i][j] += self.examples.inputs[k][j]
                    
        fig = plt.figure()
        plt.clf()
        for i in range(self.examples.noutputs):
            rpr.show_repr(lplot[i], 16, fig, 250 + i, i)
        path = "/tmp/pyplot.%s.%s.png" % (sys.argv[0], time.strftime("%m-%d-%H-%M-%S", time.localtime()))
        plt.savefig(path)
        plt.show()
    
